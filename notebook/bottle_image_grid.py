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
SCALES = [-1.0, 1.0, 0.0]
LORA_CHKPT = range(100, 25000, 100)
#SEEDS = [807, 200, 201, 202, 800]
SEEDS = range(0, 10000, 100)
LEARNING_RATE = "5e-5"
#MASTER_DIR = "../output/chkpt100/512_unsplash250_cast_learning_rate/"
MASTER_DIR = ""
ROOT_DIR = ""
COLOR_TYPE="gray"

def add_text(content):
    prompt_id = 0
    ROOT_DIR = content["root_dir"]
    seed_id = content["seed_id"]
    scale_id = content["scale_id"]
    #lora_id = content["lora_id"]
    #lora_name = content["lora_name"]
    prompt_name = content["prompt_name"]
    scale_name = content["scale_name"]
    seed_name = content["seed_name"]
    real_scale_id = scale_id

    folder_out = f"{ROOT_DIR}/{COLOR_TYPE}_with_text"

    os.makedirs(folder_out, exist_ok=True)

    
    out_path = f"{folder_out}/{prompt_id:04d}_{real_scale_id:04d}_{seed_id:04d}.png"
    if os.path.exists(out_path):
        return None
    
    folder_in = f"{ROOT_DIR}/{COLOR_TYPE}"
    
    fpath = f"{folder_in}/{prompt_id:04d}_{real_scale_id:04d}_{seed_id:04d}.png"
    #print(fpath)
    image = Image.open(fpath)
    draw = ImageDraw.Draw(image)
    draw.text((10, 10),f"prompt: {prompt_name}\nscale: {scale_name:.2f}\nseed: {seed_name:3d}\nLR: {LEARNING_RATE}",(255,255,255),font=font)
    image.save(out_path)
    return None

def build_row(ROOT_DIR):
    os.makedirs(f"{ROOT_DIR}/{COLOR_TYPE}_row", exist_ok=True)
    #lora_id = image_id
    for scale_id, scale_name in enumerate(SCALES):
        images = []
        for seed_id, seed_name in enumerate(SEEDS):
        
            fpath = os.path.join(ROOT_DIR, f"{COLOR_TYPE}_with_text",f'0000_{scale_id:04d}_{seed_id:04d}.png')
            image = torchvision.io.read_image(fpath)
            image = torchvision.transforms.functional.resize(image, (256,256))
            #image = torch.zeros(3, 256, 256).to(torch.uint8)
            images.append(image)
        images = torch.stack(images)
        grid = torchvision.utils.make_grid(images, nrow=10)
        grid = torchvision.transforms.ToPILImage()(grid)
        output_name = f"{ROOT_DIR}/{COLOR_TYPE}_row/{scale_id:04d}.png"
        grid.save(output_name)

def main():
    # dir_data = [
    #     "ckpt10000_5e-5_bottle_start199_stop0",
    #     "ckpt10000_5e-5_bottle_start199_stop100",
    #     "ckpt10000_5e-5_bottle_start299_stop200",
    #     "ckpt10000_5e-5_bottle_start399_stop200",
    #     "ckpt10000_5e-5_bottle_start399_stop300",
    #     "ckpt10000_5e-5_bottle_start499_stop0",
    #     "ckpt10000_5e-5_bottle_start499_stop400",
    #     "ckpt10000_5e-5_bottle_start599_stop400",
    #     "ckpt10000_5e-5_bottle_start599_stop500",
    #     "ckpt10000_5e-5_bottle_start699_stop600",
    #     "ckpt10000_5e-5_bottle_start799_stop600",
    #     "ckpt10000_5e-5_bottle_start799_stop700",
    #     "ckpt10000_5e-5_bottle_start899_stop800",
    #     "ckpt10000_5e-5_bottle_start999_stop500",
    #     "ckpt10000_5e-5_bottle_start999_stop799",
    #     "ckpt10000_5e-5_bottle_start999_stop900"
    # ]
    # create text 
    #if True:
    for dir_name in dir_data:
        #ROOT_DIR = f"{MASTER_DIR}/{dir_name}"
        ROOT_DIR = "../output/chkpt100/512_unsplash250_cast_singlescale_chkpt100_lr5e-5_all/"
        jobs = []
        for idx in range(0, len(SCALES)):
            #for lora_id, lora_name in enumerate((LORA_CHKPT)):
            if True:
                for prompt_id, prompt_name in enumerate(OBJECTS):
                    for scale_id, scale_name in enumerate(SCALES): #enumerate(SCALES):
                        for seed_id, seed_name in enumerate(SEEDS):
                            content = {
                                # "lora_id": lora_id,
                                # "lora_name": lora_name,
                                "root_dir": ROOT_DIR,
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
        # for job in tqdm(jobs):
        #     add_text(job)
        build_row(ROOT_DIR)
    if False:
        with Pool(16) as p:
            r = list(tqdm(p.imap(build_row, range(len(LORA_CHKPT))), total=len(LORA_CHKPT)))

    

    # create grid

if __name__ == "__main__":
    main()