import argparse
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)
from diffusers import DDIMScheduler
from models.arc2face.arc2face import CLIPTextModelWrapper, project_face_embs

from ddm_inversion.utils import image_grid

from torch import autocast, inference_mode

import calendar
import time
import os
import torch
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from ddm_inversion.inversion_utils_adapted import (
        inversion_forward_process_uncond,
        inversion_reverse_process_uncond
    )

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--eta", type=float, default=1)
    args = parser.parse_args()
    
    # Load your model (here, using a StableDiffusionPipeline-like interface)
    model_id = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
    # Load encoder and unet as per your Arc2Face setup. (Adjust paths as needed.)
    encoder = CLIPTextModelWrapper.from_pretrained(
        'models', subfolder="arc2face/models/encoder", torch_dtype=torch.float
    )
    unet = UNet2DConditionModel.from_pretrained(
        'models', subfolder="arc2face/models/arc2face", torch_dtype=torch.float
    )
    
    device = f"cuda:{args.device_num}"
    ldm_stable = StableDiffusionPipeline.from_pretrained(
        model_id,
        text_encoder=encoder,
        unet=unet,
        torch_dtype=torch.float,
        safety_checker=None
    ).to(device)
    
    # Set up the scheduler
    ldm_stable.scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
    ldm_stable.scheduler.set_timesteps(args.num_diffusion_steps)
    
    # Load and process image
    image_path = 'models/arc2face/assets/examples/freddie.png'
    img_pil = Image.open(image_path).convert("RGB")
    img_resized = img_pil.resize((512, 512), Image.LANCZOS)  # Using LANCZOS as ANTIALIAS is deprecated.
    img_np = np.array(img_resized)
    x0 = torch.from_numpy(img_np).float() / 127.5 - 1
    x0 = x0.permute(2, 0, 1).unsqueeze(0).to(device)
    
    # VAE encode the image to obtain the initial latent (w0)
    with autocast("cuda"), inference_mode():
        w0 = (ldm_stable.vae.encode(x0).latent_dist.mode() * 0.18215).float()
    
    # --- Forward Inversion Process ---
    # This builds the noisy trajectory and extracts the noise maps.
    w_final, zs, wts = inversion_forward_process_uncond(
        ldm_stable, 
        w0, 
        etas=args.eta, 
        num_inference_steps=args.num_diffusion_steps, 
        prog_bar=True
    )
    
    # --- Reverse Inversion Process ---
    # Starting from the final latent in the trajectory, reconstruct x0.
    w0_reconstructed, _ = inversion_reverse_process_uncond(
        ldm_stable,
        xT=wts[-1], 
        etas=args.eta, 
        zs=zs,
        num_inference_steps=args.num_diffusion_steps, 
        prog_bar=True
    )
    
    # Decode the reconstructed latent back to an image for verification.
    with autocast("cuda"), inference_mode():
        x0_dec = ldm_stable.vae.decode(1 / 0.18215 * w0_reconstructed).sample
    if x0_dec.dim() < 4:
        x0_dec = x0_dec.unsqueeze(0)
    final_image = image_grid(x0_dec)
    
    # Save the output image.
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    time_stamp = calendar.timegm(time.gmtime())
    output_path = os.path.join(output_dir, f'freddie_reconstructed_{time_stamp}.png')
    final_image.save(output_path)
    
    print(f"Inversion complete. Output saved to {output_path}")

    ################################# TEST ################################
    
    ## MSE
    mse = torch.mean((w0 - w0_reconstructed) ** 2)
    print("MSE between original and reconstructed latent:", mse.item())
    
    
    ## PSNR

    # Decode both latents
    with torch.no_grad():
        x0_dec = ldm_stable.vae.decode(1 / 0.18215 * w0).sample
        x0_rec_dec = ldm_stable.vae.decode(1 / 0.18215 * w0_reconstructed).sample

    # Convert to numpy for metric calculation
    x0_dec_np = x0_dec.clamp(-1, 1).cpu().numpy()
    x0_rec_dec_np = x0_rec_dec.clamp(-1, 1).cpu().numpy()

    mse_image = ((x0_dec_np - x0_rec_dec_np) ** 2).mean()
    psnr_value = psnr(x0_dec_np, x0_rec_dec_np, data_range=2.0)  # since images are in [-1,1]
    print("Image MSE:", mse_image)
    print("Image PSNR:", psnr_value)

    import pdb
    pdb.set_trace()
