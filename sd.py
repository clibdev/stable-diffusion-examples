from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_single_file('https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors')
pipe = pipe.to('cuda')

prompt = 'a photo of an astronaut riding a horse on mars'
image = pipe(prompt).images[0]

image.save('astronaut_rides_horse.png')
