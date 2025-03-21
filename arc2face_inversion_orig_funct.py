import argparse
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)
from diffusers import DDIMScheduler
from models.arc2face.arc2face import CLIPTextModelWrapper, project_face_embs

from prompt_to_prompt.ptp_classes import AttentionStore, AttentionReplace, AttentionRefine, EmptyControl,load_512
from prompt_to_prompt.ptp_utils import register_attention_control, text2image_ldm_stable, view_images
from ddm_inversion.inversion_utils import  inversion_forward_process, inversion_reverse_process
from ddm_inversion.utils import image_grid,dataset_from_yaml

from torch import autocast, inference_mode
from ddm_inversion.ddim_inversion import ddim_inversion

import calendar
import time
import os
import torch
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--cfg_src", type=float, default=3.5)
    parser.add_argument("--cfg_tar", type=float, default=15)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--dataset_yaml",  default="test.yaml")
    parser.add_argument("--eta", type=float, default=1)
    parser.add_argument("--mode",  default="our_inv", help="modes: our_inv,p2pinv,p2pddim,ddim")
    parser.add_argument("--skip",  type=int, default=36)
    parser.add_argument("--xa", type=float, default=0.6)
    parser.add_argument("--sa", type=float, default=0.2)
    
    args = parser.parse_args()
    full_data = dataset_from_yaml(args.dataset_yaml)

    # create scheduler
    # load diffusion model
    model_id = 'stable-diffusion-v1-5/stable-diffusion-v1-5'

    encoder = CLIPTextModelWrapper.from_pretrained(
    'models', subfolder="arc2face/models/encoder", torch_dtype=torch.float
    )

    unet = UNet2DConditionModel.from_pretrained(
        'models', subfolder="arc2face/models/arc2face", torch_dtype=torch.float
    )

    device = f"cuda:{args.device_num}"

    cfg_scale_src = args.cfg_src
    cfg_scale_tar_list = [args.cfg_tar]
    eta = args.eta # = 1
    skip_zs = [args.skip]
    xa_sa_string = f'_xa_{args.xa}_sa{args.sa}_' if args.mode=='p2pinv' else '_'

    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)

    # load/reload model:
    ldm_stable = StableDiffusionPipeline.from_pretrained(
        model_id,
        text_encoder=encoder,
        unet=unet,
        torch_dtype=torch.float,
        safety_checker=None
    ).to(device)

    # Extract image ID embedding
    app = FaceAnalysis(name='antelopev2', root='models/arc2face/', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    img_pil = Image.open('models/arc2face/assets/examples/pose1.png').convert("RGB")

    # Resize using PIL's resize method with anti-aliasing
    img_resized = img_pil.resize((512, 512), Image.ANTIALIAS)

    # Convert to numpy array and reverse color channels if needed
    img = np.array(img_resized)
    
    x0 = torch.from_numpy(img).float() / 127.5 - 1
    x0 = x0.permute(2, 0, 1).unsqueeze(0).to(device)

    faces = app.get(img)
    faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)

    id_emb = torch.tensor(faces['embedding'], dtype=torch.float)[None].cuda()
    id_emb = id_emb/torch.norm(id_emb, dim=1, keepdim=True)   # normalize embedding
    id_emb = project_face_embs(ldm_stable, id_emb)    # pass through the encoder

    if args.mode=="p2pddim" or args.mode=="ddim":
            scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
            ldm_stable.scheduler = scheduler
    else:
        ldm_stable.scheduler = DDIMScheduler.from_config(model_id, subfolder = "scheduler")
        
    ldm_stable.scheduler.set_timesteps(args.num_diffusion_steps)

    # vae encode image
    with autocast("cuda"), inference_mode():
        w0 = (ldm_stable.vae.encode(x0).latent_dist.mode() * 0.18215).float()

    # find Zs and wts - forward process
    if args.mode=="p2pddim" or args.mode=="ddim":
        wT = ddim_inversion(ldm_stable, w0, "", cfg_scale_src)
    else:
        wt, zs, wts = inversion_forward_process(ldm_stable, w0, etas=eta, prompt="", id_emb=id_emb, cfg_scale=cfg_scale_src, prog_bar=True, num_inference_steps=args.num_diffusion_steps)


    # Inversion reverse process
    for cfg_scale_tar in cfg_scale_tar_list:
        for skip in skip_zs:    
            if args.mode=="our_inv":
                # reverse process (via Zs and wT)
                controller = AttentionStore()
                register_attention_control(ldm_stable, controller)
                w0, _ = inversion_reverse_process(ldm_stable, xT=wts[args.num_diffusion_steps-skip], etas=eta, prompts=[""], id_emb=id_emb, cfg_scales=[cfg_scale_tar], prog_bar=True, zs=zs[:(args.num_diffusion_steps-skip)], controller=controller)

            # vae decode image
            with autocast("cuda"), inference_mode():
                x0_dec = ldm_stable.vae.decode(1 / 0.18215 * w0).sample
            if x0_dec.dim()<4:
                x0_dec = x0_dec[None,:,:,:]
            deoded_image = image_grid(x0_dec)
                
            # same output
            save_path = "./outputs"
            os.makedirs(save_path, exist_ok=True)
            current_GMT = time.gmtime()
            time_stamp_name = calendar.timegm(current_GMT)
            image_name_png = f'pose1_{time_stamp_name}_{skip}' + ".png"

            save_full_path = os.path.join(save_path, image_name_png)
            deoded_image.save(save_full_path)

    import pdb
    pdb.set_trace()

    # for i in range(len(full_data)):
    #     current_image_data = full_data[i]
    #     image_path = current_image_data['init_img']
    #     image_path = '.' + image_path 
    #     image_folder = image_path.split('/')[1] # after '.'
    #     prompt_src = current_image_data.get('source_prompt', "") # default empty string
    #     prompt_tar_list = current_image_data['target_prompts']

    #     if args.mode=="p2pddim" or args.mode=="ddim":
    #         scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    #         ldm_stable.scheduler = scheduler
    #     else:
    #         ldm_stable.scheduler = DDIMScheduler.from_config(model_id, subfolder = "scheduler")
            
    #     ldm_stable.scheduler.set_timesteps(args.num_diffusion_steps)

    #     # load image
    #     offsets=(0,0,0,0)
    #     x0 = load_512(image_path, *offsets, device)

    #     # vae encode image
    #     with autocast("cuda"), inference_mode():
    #         w0 = (ldm_stable.vae.encode(x0).latent_dist.mode() * 0.18215).float()

    #     # find Zs and wts - forward process
    #     if args.mode=="p2pddim" or args.mode=="ddim":
    #         wT = ddim_inversion(ldm_stable, w0, prompt_src, cfg_scale_src)
    #     else:
    #         wt, zs, wts = inversion_forward_process(ldm_stable, w0, etas=eta, prompt=prompt_src, cfg_scale=cfg_scale_src, prog_bar=True, num_inference_steps=args.num_diffusion_steps)

    #     import pdb
    #     pdb.set_trace()


    #     # iterate over decoder prompts
    #     for k in range(len(prompt_tar_list)):
    #         prompt_tar = prompt_tar_list[k]
    #         save_path = os.path.join(f'./results/', args.mode+xa_sa_string+str(time_stamp), image_path.split(sep='.')[0], 'src_' + prompt_src.replace(" ", "_"), 'dec_' + prompt_tar.replace(" ", "_"))
    #         os.makedirs(save_path, exist_ok=True)

    #         # Check if number of words in encoder and decoder text are equal
    #         src_tar_len_eq = (len(prompt_src.split(" ")) == len(prompt_tar.split(" ")))

    #         for cfg_scale_tar in cfg_scale_tar_list:
    #             for skip in skip_zs:    
    #                 if args.mode=="our_inv":
    #                     # reverse process (via Zs and wT)
    #                     controller = AttentionStore()
    #                     register_attention_control(ldm_stable, controller)
    #                     w0, _ = inversion_reverse_process(ldm_stable, xT=wts[args.num_diffusion_steps-skip], etas=eta, prompts=[prompt_tar], cfg_scales=[cfg_scale_tar], prog_bar=True, zs=zs[:(args.num_diffusion_steps-skip)], controller=controller)

    #                 elif args.mode=="p2pinv":
    #                     # inversion with attention replace
    #                     cfg_scale_list = [cfg_scale_src, cfg_scale_tar]
    #                     prompts = [prompt_src, prompt_tar]
    #                     if src_tar_len_eq:
    #                         controller = AttentionReplace(prompts, args.num_diffusion_steps, cross_replace_steps=args.xa, self_replace_steps=args.sa, model=ldm_stable)
    #                     else:
    #                         # Should use Refine for target prompts with different number of tokens
    #                         controller = AttentionRefine(prompts, args.num_diffusion_steps, cross_replace_steps=args.xa, self_replace_steps=args.sa, model=ldm_stable)

    #                     register_attention_control(ldm_stable, controller)
    #                     w0, _ = inversion_reverse_process(ldm_stable, xT=wts[args.num_diffusion_steps-skip], etas=eta, prompts=prompts, cfg_scales=cfg_scale_list, prog_bar=True, zs=zs[:(args.num_diffusion_steps-skip)], controller=controller)
    #                     w0 = w0[1].unsqueeze(0)

    #                 elif args.mode=="p2pddim" or args.mode=="ddim":
    #                     # only z=0
    #                     if skip != 0:
    #                         continue
    #                     prompts = [prompt_src, prompt_tar]
    #                     if args.mode=="p2pddim":
    #                         if src_tar_len_eq:
    #                             controller = AttentionReplace(prompts, args.num_diffusion_steps, cross_replace_steps=.8, self_replace_steps=0.4, model=ldm_stable)
    #                         # Should use Refine for target prompts with different number of tokens
    #                         else:
    #                             controller = AttentionRefine(prompts, args.num_diffusion_steps, cross_replace_steps=.8, self_replace_steps=0.4, model=ldm_stable)
    #                     else:
    #                         controller = EmptyControl()

    #                     register_attention_control(ldm_stable, controller)
    #                     # perform ddim inversion
    #                     cfg_scale_list = [cfg_scale_src, cfg_scale_tar]
    #                     w0, latent = text2image_ldm_stable(ldm_stable, prompts, controller, args.num_diffusion_steps, cfg_scale_list, None, wT)
    #                     w0 = w0[1:2]
    #                 else:
    #                     raise NotImplementedError
                    
    #                 # vae decode image
    #                 with autocast("cuda"), inference_mode():
    #                     x0_dec = ldm_stable.vae.decode(1 / 0.18215 * w0).sample
    #                 if x0_dec.dim()<4:
    #                     x0_dec = x0_dec[None,:,:,:]
    #                 img = image_grid(x0_dec)
                       
    #                 # same output
    #                 current_GMT = time.gmtime()
    #                 time_stamp_name = calendar.timegm(current_GMT)
    #                 image_name_png = f'cfg_d_{cfg_scale_tar}_' + f'skip_{skip}_{time_stamp_name}' + ".png"

    #                 save_full_path = os.path.join(save_path, image_name_png)
    #                 img.save(save_full_path)