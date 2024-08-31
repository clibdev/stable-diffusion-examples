from diffusers import DiffusionPipeline
import torch
import os
import re

output_path = 'images'
if not os.path.exists(output_path):
    os.makedirs(output_path)

files = os.listdir(output_path)
if files:
    ids = [int(re.search('(\\d+)', name)[0]) for name in files]
    filename = 'test_' + str(max(ids) + 1) + '.jpg'
else:
    filename = 'test_1.jpg'

base = DiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype=torch.float16,
    variant='fp16',
    use_safetensors=True
)
base.to('cuda')
refiner = DiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-refiner-1.0',
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant='fp16',
)
refiner.to('cuda')

n_steps = 40
high_noise_frac = 0.8

prompt = 'High-fashion photography. Old man face'

image = base(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type='latent',
).images
image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]

image.save(os.path.join(output_path, filename))
