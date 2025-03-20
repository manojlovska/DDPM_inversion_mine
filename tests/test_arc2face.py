from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)

from models.arc2face.arc2face import CLIPTextModelWrapper, project_face_embs

import os
import torch
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np

# Assuming 'images' is a list of PIL Image objects
def save_images_grid(images, grid_size=None, save_path="combined_image.png"):
    num_images = len(images)
    img_width, img_height = images[0].size  # Assuming all images are the same size

    # Determine grid size (auto if not provided)
    if grid_size is None:
        grid_cols = int(num_images ** 0.5)  # Square root for square-like grid
        grid_rows = (num_images + grid_cols - 1) // grid_cols  # Round up
    else:
        grid_cols, grid_rows = grid_size

    # Create blank canvas
    combined_img = Image.new("RGB", (grid_cols * img_width, grid_rows * img_height))

    # Paste images into the grid
    for idx, img in enumerate(images):
        x_offset = (idx % grid_cols) * img_width
        y_offset = (idx // grid_cols) * img_height
        combined_img.paste(img, (x_offset, y_offset))

    # Save the combined image
    combined_img.save(save_path)
    print(f"Combined image saved as {save_path}")

# Arc2Face is built upon SD1.5
# The repo below can be used instead of the now deprecated 'runwayml/stable-diffusion-v1-5'
base_model = 'stable-diffusion-v1-5/stable-diffusion-v1-5'

encoder = CLIPTextModelWrapper.from_pretrained(
    'models', subfolder="arc2face/models/encoder", torch_dtype=torch.float16
)

unet = UNet2DConditionModel.from_pretrained(
    'models', subfolder="arc2face/models/arc2face", torch_dtype=torch.float16
)

pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        text_encoder=encoder,
        unet=unet,
        torch_dtype=torch.float16,
        safety_checker=None
    )

# Scheduler
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to('cuda:0')

# Extract image ID embedding
app = FaceAnalysis(name='antelopev2', root='models/arc2face/', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

img = np.array(Image.open('models/arc2face/assets/examples/jackie.png'))[:,:,::-1]

faces = app.get(img)
faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
id_emb = torch.tensor(faces['embedding'], dtype=torch.float16)[None].cuda()
id_emb = id_emb/torch.norm(id_emb, dim=1, keepdim=True)   # normalize embedding
id_emb = project_face_embs(pipeline, id_emb)    # pass through the encoder

# Generate images
num_images = 4
images = pipeline(prompt_embeds=id_emb, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=num_images).images

os.makedirs("tests/images", exist_ok=True)
save_images_grid(images, save_path="tests/images/output_grid.png")

import pdb
pdb.set_trace()