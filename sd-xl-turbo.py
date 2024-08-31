from diffusers import AutoPipelineForText2Image
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

pipe = AutoPipelineForText2Image.from_pretrained(
    'stabilityai/sdxl-turbo',
    torch_dtype=torch.float16,
    variant='fp16'
)
pipe.to('cuda')

prompt = 'High-fashion photography. Old man face'

image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
image.save(os.path.join(output_path, filename))
