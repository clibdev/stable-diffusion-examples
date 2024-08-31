from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
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

base = 'stabilityai/stable-diffusion-xl-base-1.0'
repo = 'ByteDance/SDXL-Lightning'
ckpt = 'sdxl_lightning_4step_unet.safetensors'

config = UNet2DConditionModel.load_config(base, subfolder='unet')
unet = UNet2DConditionModel.from_config(config).to('cuda', torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device='cuda'))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant='fp16').to('cuda')

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing='trailing')

prompt = 'High-fashion photography. Old man face'

image = pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0]
image.save(os.path.join(output_path, filename))
