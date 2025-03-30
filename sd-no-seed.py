from diffusers import StableDiffusionPipeline
import torch
import numpy as np

pipe = StableDiffusionPipeline.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5')
pipe = pipe.to('cuda')

np.random.seed(0)
latents = np.random.randn(1, 4, 64, 64)
latents = torch.tensor(latents, dtype=torch.float32)

prompt = 'a photo of an astronaut riding a horse on mars'
image = pipe(prompt, latents=latents, num_inference_steps=20).images[0]

image.save('astronaut_rides_horse.png')
