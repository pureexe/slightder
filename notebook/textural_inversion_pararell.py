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

LORA_CHKPT = range(1000, 40000, 1000)
SEEDS = [807, 200, 201, 202, 800]
#LEARNING_RATE = "1e-3"
#LEARNING_RATE = "1e-4"
#LEARNING_RATE = "5e-4"
LEARNING_RATE = "5e-5"
SCENE = "beach"
IMAGE_ID = 3
# INPUT_DIR = f"../output/textural_inversion/raw/mapnetlearnmatrix_interpolate_chkpt100_{SCENE}_lr{LEARNING_RATE}_alpha1.0_rank4_noxattn"
# ROW_DIR = f"../output/textural_inversion/row/mapnetlearnmatrix_interpolate_chkpt100_{SCENE}_lr{LEARNING_RATE}_alpha1.0_rank4_noxattn"
# OUTPUT_DIR = f"../output/textural_inversion/text/mapnetlearnmatrix_interpolate_chkpt100_{SCENE}_lr{LEARNING_RATE}_alpha1.0_rank4_noxattn"

# INPUT_DIR = f"../output/textural_inversion/raw/mapnetlearnmatrix_interpolate_chkpt100_{SCENE}_i{IMAGE_ID}_lr{LEARNING_RATE}_alpha1.0_rank4_noxattn"
# ROW_DIR = f"../output/textural_inversion/row/mapnetlearnmatrix_interpolate_chkpt100_{SCENE}_i{IMAGE_ID}_lr{LEARNING_RATE}_alpha1.0_rank4_noxattnn"
# OUTPUT_DIR = f"../output/textural_inversion/text/mapnetlearnmatrix_interpolate_chkpt100_{SCENE}_i{IMAGE_ID}_lr{LEARNING_RATE}_alpha1.0_rank4_noxattn"

INPUT_DIR = f"../output/textural_inversion/raw/textural_inversion_classic_{SCENE}_lr{LEARNING_RATE}_alpha1.0_rank4_noxattn"
ROW_DIR = f"../output/textural_inversion/row/textural_inversion_classic_{SCENE}_lr{LEARNING_RATE}_alpha1.0_rank4_noxattn"
OUTPUT_DIR = f"../output/textural_inversion/text/textural_inversion_classic_{SCENE}_lr{LEARNING_RATE}_alpha1.0_rank4_noxattn"

def add_text(content):
    prompt_id = 0
    real_scale_id = 0

    seed_id = content["seed_id"]
   # scale_id = content["scale_id"]
    lora_id = content["lora_id"]
    lora_name = content["lora_name"]
    seed_name = content["seed_name"]

    folder_out = OUTPUT_DIR

    os.makedirs(folder_out, exist_ok=True)

    
    out_path = f"{folder_out}/{lora_id:04d}_{prompt_id:04d}_{real_scale_id:04d}_{seed_id:04d}.png"
    if os.path.exists(out_path):
        return None
    
    folder_in = INPUT_DIR
    
    fpath = f"{folder_in}/{lora_id:04d}_{prompt_id:04d}_{real_scale_id:04d}_{seed_id:04d}.png"
    #print(fpath)
    try:
        image = Image.open(fpath)
    except:
        image = Image.new("RGB", (512,512), (0, 0, 0))

    draw = ImageDraw.Draw(image)
    draw.text((10, 10),f"step: {lora_name:5d}\nseed: {seed_name:3d}\nLR: {LEARNING_RATE}",(255,255,255),font=font)
    image.save(out_path)
    return None

def build_row(image_id):
    os.makedirs(ROW_DIR, exist_ok=True)
    lora_id = image_id
    images = []
    for seed_id, seed_name in enumerate(SEEDS):
        fpath = os.path.join(OUTPUT_DIR,f'{lora_id:04d}_0000_0000_{seed_id:04d}.png')
        try:
            image = torchvision.io.read_image(fpath)
            image = torchvision.transforms.functional.resize(image, (256,256))
        except:
            image = torch.zeros(3, 256, 256).to(torch.uint8)
        images.append(image)
    images = torch.stack(images)
    grid = torchvision.utils.make_grid(images, nrow=len(LORA_CHKPT))
    grid = torchvision.transforms.ToPILImage()(grid)
    output_name = f"{ROW_DIR}/{lora_id:04d}.png"
    grid.save(output_name)

def main():
    # create text 
    if True:
        jobs = []
        for lora_id, lora_name in enumerate((LORA_CHKPT)):
            for seed_id, seed_name in enumerate(SEEDS):
                content = {
                    "lora_id": lora_id,
                    "lora_name": lora_name,
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