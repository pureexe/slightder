import torchvision 
import torch 
from PIL import Image 
from multiprocessing import Pool
import os 
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf", 32)
import numpy as np
from tqdm.auto import tqdm
import skimage
from tqdm.auto import tqdm

OBJECTS = [
    'shoe',
]
SCALES = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
LORA_CHKPT = range(100, 12100, 100)
SEEDS = [807, 200, 201, 202, 800]
LEARNING_RATE = "5e-5"
ROOT_DIR = "../output/chkpt100/512_unsplash250_cast_singlescale_chkpt100_lr1e-4_shoescale/"
COLOR_TYPE="gray"

def add_text(content):
    prompt_id = 0
    real_scale_id = 0

    seed_id = content["seed_id"]
    scale_id = content["scale_id"]
    lora_id = content["lora_id"]
    lora_name = content["lora_name"]
    prompt_name = content["prompt_name"]
    scale_name = content["scale_name"]
    seed_name = content["seed_name"]

    folder_out = f"{ROOT_DIR}/scale_{SCALES[scale_id]:.2f}/{COLOR_TYPE}_with_text"

    os.makedirs(folder_out, exist_ok=True)

    
    out_path = f"{folder_out}/{lora_id:04d}_{prompt_id:04d}_{real_scale_id:04d}_{seed_id:04d}.png"
    if os.path.exists(out_path):
        return None
    
    folder_in = f"{ROOT_DIR}/scale_{SCALES[scale_id]:.2f}/{COLOR_TYPE}"
    
    fpath = f"{folder_in}/{lora_id:04d}_{prompt_id:04d}_{real_scale_id:04d}_{seed_id:04d}.png"
    image = Image.open(fpath)
    draw = ImageDraw.Draw(image)
    draw.text((10, 10),f"step: {lora_name:5d}\nprompt: {prompt_name}\nscale: {scale_name:.2f}\nseed: {seed_name:3d}\nLR: {LEARNING_RATE}",(255,255,255),font=font)
    image.save(out_path)
    return None

def build_row(image_id):
    os.makedirs(f"{ROOT_DIR}/gray_row", exist_ok=True)
    lora_id = image_id
    images = []
    for seed_id, seed_name in enumerate(SEEDS):
        for scale_id, scale_name in enumerate(SCALES):
            fpath = os.path.join(ROOT_DIR, f"scale_{scale_name:.2f}","gray_with_text",f'{lora_id:04d}_0000_0000_{seed_id:04d}.png')
            image = torchvision.io.read_image(fpath)
            image = torchvision.transforms.functional.resize(image, (256,256))
            images.append(image)
    images = torch.stack(images)
    grid = torchvision.utils.make_grid(images, nrow=len(SCALES))
    grid = torchvision.transforms.ToPILImage()(grid)
    output_name = f"{ROOT_DIR}/gray_row/{lora_id:04d}.png"
    grid.save(output_name)

def main():
    # create text 
    if False:

        jobs = []
        for idx in range(0, len(SCALES)):
            for lora_id, lora_name in enumerate((LORA_CHKPT)):
                for prompt_id, prompt_name in enumerate(OBJECTS):
                    for scale_id, scale_name in enumerate(SCALES): #enumerate(SCALES):
                        for seed_id, seed_name in enumerate(SEEDS):
                            content = {
                                "lora_id": lora_id,
                                "lora_name": lora_name,
                                "prompt_id": prompt_id,
                                "prompt_name": prompt_name,
                                "scale_id": scale_id,
                                "scale_name": scale_name,
                                "seed_id": seed_id,
                                "seed_name": seed_name
                            }
                            jobs.append(content)
        with Pool(16) as p:
            r = list(tqdm(p.imap(add_text, jobs), total=len(jobs)))
    if True:
        with Pool(16) as p:
            r = list(tqdm(p.imap(build_row, range(len(LORA_CHKPT))), total=len(LORA_CHKPT)))

    

    # create grid

if __name__ == "__main__":
    main()