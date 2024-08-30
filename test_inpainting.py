from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
import sys, os
import numpy as np

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    # torch_dtype=torch.float16,
)
pipe = pipe.to('cuda')
# prompt = "a new york night skyline background of a studio"
prompt = "background of a news studio"
#image and mask_image should be PIL images.
image = Image.open('./data/john-oliver-SHOW1/JO_SHOW1/images/00001.png')
mask = np.array(Image.open('./data/john-oliver-SHOW1/JO_SHOW1/matted/00001.png'))[:,:,3]
mask[:,100:400] = np.ones_like(mask[:,100:400]) * 255
# mask = Image.fromarray(((1 - mask / 255.)* 255).astype(np.uint8))
mask = Image.fromarray(((mask)).astype(np.uint8))
mask.save('./john-oliver-mask.png')
#The mask structure is white for inpainting and black for keeping as is
image = pipe(prompt=prompt, image=image, mask_image=mask).images[0]

image.save("./john-oliver-inpainted.png")


# os.system(f'yt-dlp https://www.youtube.com/watch?v=1K1XKClRb38 -f mp4 --download-sections *00:00:00-00:08:40 -o fish_tank_part1.mp4')