from transformers import CLIPTokenizer

# vocab.json and merges.txt download from:
# https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer

args = {
    'do_lower_case': True,
    'strip_accents': False,
    'do_split_on_punc': False,
}

prompt = 'a photo of an astronaut riding a horse on mars'

clip = CLIPTokenizer('vocab.json', 'merges.txt', **args)
encodings = clip.encode(prompt)

print(encodings)
