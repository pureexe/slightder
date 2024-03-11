import torchvision 
import torch 
from PIL import Image 

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
#SCALES = [-1.0, 1.0]
SCALES = [-1.0]
LORA_CHKPT = [0] #range(100, 40000, 100)
#SEEDS = [807, 200, 201, 202, 800]
SEEDS = range(0,10000, 100)
LEARNING_RATE = "5e-5"
#ROOT_DIR = "../output/chkpt100/512_unsplash250_cast_learning_rate/"
ROOT_DIR = "../output/chkpt100/512_unsplash250_cast_learning_rate/ckpt10000_5e-5_bottle_minus1"
# COLOR_TYPE="gray"
# FOLDER_OUT = f"{ROOT_DIR}/{LEARNING_RATE}/{COLOR_TYPE}_with_text"
# FOLDER_IN = f"{ROOT_DIR}/{LEARNING_RATE}/{COLOR_TYPE}"
# FOLDER_VID = f"{ROOT_DIR}/{LEARNING_RATE}/{COLOR_TYPE}_video"
# FOLDER_ROW = f"{ROOT_DIR}/{LEARNING_RATE}/{COLOR_TYPE}_row"

COLOR_TYPE="gray"
FOLDER_OUT = f"{ROOT_DIR}/{COLOR_TYPE}_with_text"
FOLDER_IN = f"{ROOT_DIR}/{COLOR_TYPE}"
FOLDER_VID = f"{ROOT_DIR}/{COLOR_TYPE}_video"
FOLDER_ROW = f"{ROOT_DIR}/{COLOR_TYPE}_row"
os.makedirs(FOLDER_OUT, exist_ok=True)
os.makedirs(FOLDER_VID, exist_ok=True)
os.makedirs(FOLDER_ROW, exist_ok=True)



for lora_id, lora_name in enumerate(tqdm(LORA_CHKPT)):
    for prompt_id, prompt_name in enumerate(OBJECTS):
        for scale_id, scale_name in enumerate(SCALES):
            for seed_id, seed_name in enumerate(SEEDS):
                fpath = f"{FOLDER_IN}/{prompt_id:04d}_{scale_id:04d}_{seed_id:04d}.png"
                #print(fpath)
                image = Image.open(fpath)
                draw = ImageDraw.Draw(image)
                draw.text((10, 10),f"prompt: {prompt_name}\nscale: {scale_name:.2f}\nseed: {seed_name:3d}\nLR: {LEARNING_RATE}",(255,255,255),font=font)
                image.save(f"{FOLDER_OUT}/{lora_id:04d}_{prompt_id:04d}_{scale_id:04d}_{seed_id:04d}.png")


for lora_id, lora_name in enumerate(tqdm(LORA_CHKPT)):
    for scale_id, scale_name in enumerate(SCALES):
        images = []
        for prompt_id, prompt_name in enumerate(OBJECTS):
            for seed_id, seed_name in enumerate(SEEDS):
                fpath = os.path.join(FOLDER_OUT, f'{lora_id:04d}_{prompt_id:04d}_{scale_id:04d}_{seed_id:04d}.png')
                image = torchvision.io.read_image(fpath)
                image = torchvision.transforms.functional.resize(image, (256,256))
                images.append(image)
        images = torch.stack(images)
        #grid = torchvision.utils.make_grid(images, nrow=len(SEEDS))
        grid = torchvision.utils.make_grid(images, nrow=10)
        grid = torchvision.transforms.ToPILImage()(grid)
        output_name = f"{FOLDER_ROW}/{lora_id:04d}_{scale_id:04d}.png"
        grid.save(output_name)
