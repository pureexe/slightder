# 2girl SCALES:  [tensor(0.2626)]

ACTUAL_SCALE = 1.0
OBJECTS = [
    'shoe',
    # 'cake',
    # 'bottle',
    # 'chair',
    # 'cup',
    # 'laptop',
    # 'cell phone',
    # 'keyboard',
    # 'book',
    # 'scissors',
]

#OBJECTS = ['']


#SCALES = [-1.0, 1.0,  0.0]
#SCALES = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
#SCALES = [1.0]
#SCALES = [-1.0]
#SCALES = [0.0]
#SCALES = [-0.5]
#SCALES = [0.5]
#SCALES = [0.75]
#SCALES = [-0.75]
#SCALES = [0.25]
#SCALES = [-0.25]

#SCALES = [24]
#SCALES = [174]

#SCALES = [158]
#SCALES = [228]
#print("VERSION: ", SCALES)

# From right - 2wWJ--XoTyg.png: 24
# From right - eh_Q3gHA8gM.png: 174
# From left - _UZJN5WmrSI.png: 158
# From left - qsgZMnf0Uyc.png: 228

#SEEDS = range(0, 10000, 100)

SEEDS = [807, 200, 201, 202, 800]
# SEEDS = [807]

import torch
from PIL import Image
import argparse
import os, json, random
import pandas as pd
import matplotlib.pyplot as plt
import glob, re

from tqdm.auto import tqdm
import numpy as np

from safetensors.torch import load_file
import matplotlib.image as mpimg
import copy
import gc
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, LMSDiscreteScheduler, PNDMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor, AttentionProcessor
from typing import Any, Dict, List, Optional, Tuple, Union
from trainscripts.textsliders.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
#from trainscripts.imagesliders.pure_util.lora_global_adapter import LoRAGlobalSingleScaleAdapter
from torch import nn

# from trainscripts.imagesliders.pure_util.lora_global_adapter import FeedForward, GlobalAdapter, GEGLU

from trainscripts.imagesliders.pure_util.lora_global_adapter import LoRAMappingNetwork

from attention_map.utils import (
    cross_attn_init,
    register_cross_attention_hook,
    attn_maps,
    get_net_attn_map,
    resize_net_attn_map,
    save_net_attn_map,
)

#cross_attn_init()
### CONFIGURATION
width = 512
height = 512 
# width = 256
# height = 256 
steps = 50  
cfg_scale = 3
pretrained_sd_model = "CompVis/stable-diffusion-v1-4"
pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"

revision = None
device = 'cuda:0'
rank = 4
weight_dtype = torch.float32


RANK = "4"
CHECKPOINT = "30000"
LEARNING_RATE = "1e-3"
#LEARNING_RATE = "5e-4"
#LEARNING_RATE = "1e-4"
#LEARNING_RATE = "5e-5" 
NUM_BIN = 2
IMAGE_ID = 2
SCENE = "2scenes_no_prompt"
checkpoint = 30000
lora_weights = [
    #f"models/512_unsplash250_mapnet_single_chkpt100_lr{LEARNING_RATE}_alpha1.0_rank4_noxattn/512_unsplash250_mapnet_single_chkpt100_lr{LEARNING_RATE}_alpha1.0_rank4_noxattn_{CHECKPOINT}steps.pt",
    #f"models/512_unsplash250_mapnet_single_chkpt100_lr{LEARNING_RATE}_alpha1.0_rank4_noxattn/512_unsplash250_mapnet_single_chkpt100_lr{LEARNING_RATE}_alpha1.0_rank4_noxattn_{checkpoint}steps.pt" for checkpoint in range(100, 40100, 500)
    #f"models/512_unsplash250_mapnetlearnmatrix_single_chkpt100_lr{LEARNING_RATE}_alpha1.0_rank4_noxattn/512_unsplash250_mapnetlearnmatrix_single_chkpt100_lr{LEARNING_RATE}_alpha1.0_rank4_noxattn_{checkpoint}steps.pt"  for checkpoint in range(500, 40500, 500)
    #f"models/512_unsplash250_mapnetlearnmatrix_interpolate_chkpt100_lr{LEARNING_RATE}_bin{NUM_BIN}_alpha1.0_rank4_noxattn/512_unsplash250_mapnetlearnmatrix_interpolate_chkpt100_lr{LEARNING_RATE}_bin{NUM_BIN}_alpha1.0_rank4_noxattn_{checkpoint}steps.pt"  for checkpoint in range(1000, 41000, 1000)
    #f"models/mapnetlearnmatrix_interpolate_chkpt100_{SCENE}_lr{LEARNING_RATE}_alpha1.0_rank4_noxattn/mapnetlearnmatrix_interpolate_chkpt100_{SCENE}_lr{LEARNING_RATE}_alpha1.0_rank4_noxattn_{checkpoint}steps.pt" for checkpoint in range(1000, 41000, 1000)
    #f"models/v2_mapnetlearnmatrix_interpolate_chkpt100_{SCENE}__lr{LEARNING_RATE}_alpha1.0_rank4_noxattn/v2_mapnetlearnmatrix_interpolate_chkpt100_{SCENE}__lr{LEARNING_RATE}_alpha1.0_rank4_noxattn_{checkpoint}steps.pt"
    f"models/mapnetlearnmatrix_interpolate_chkpt100_{SCENE}_lr{LEARNING_RATE}_alpha1.0_rank4_noxattn/mapnetlearnmatrix_interpolate_chkpt100_{SCENE}_lr{LEARNING_RATE}_alpha1.0_rank4_noxattn_{checkpoint}steps.pt"
]



#output_dir = f"output/chkpt100/512_unsplash250_mapnet_single_chkpt100_lr{LEARNING_RATE}/chkpt{CHECKPOINT}/gray"
#output_dir = f"output/chkpt100/512_unsplash250_mapnetlearnmatrix_interpolate_chkpt100_lr{LEARNING_RATE}_bin{NUM_BIN}_scale/scale_{SCALES[0]:.02f}/gray"
output_dir = f"output/textural_inversion/raw/mapnetlearnmatrix_interpolate_chkpt100_{SCENE}_i{IMAGE_ID}_lr{LEARNING_RATE}_alpha1.0_rank4_noxattn"


PROMPTS = [ 
    #"a photo of {}, close-up, product photography, commercial photography, white lighting, studio lighting, a slightly look down camera, blank gray background, solid background",
    #"{}, gray background",
    #"{}"
]

# timestep during inference when we switch to LoRA scale>0 (this is done to ensure structure in the images)
start_noise = 999
stop_noise = 0 

# seed for random number generator
seed = 0

#number of images per prompt
num_images_per_prompt = 1

torch_device = device
negative_prompt = None
batch_size = 1
ddim_steps = 50

#guidance_scales = [5.0]
#guidance_scales = [7.0]
guidance_scales = [0.0]

ALL_BIN = np.linspace(-1, 1, NUM_BIN)
ALL_BIN_TENSOR = torch.tensor(ALL_BIN).to(device)


def flush():
    torch.cuda.empty_cache()
    gc.collect()
flush()

# LOAD DATASET 
from torch.utils.data import Dataset, DataLoader
from trainscripts.imagesliders.pure_util.datasets.image_axis3 import ImageAxis3Dataset
dataset = ImageAxis3Dataset(root_dir=f"datasets/v2/textural/{SCENE}_0_1", image_size = (512,512), is_normalize=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
data_iter = iter(dataloader)
for i in range(1,IMAGE_ID+1):
    batch = next(data_iter)
    print(batch['text'][0])

# setting new prompt 
PROMPTS = [ batch['text'][0] ]
SCALES = [ batch['coeff'][0][0] ]
print("SCALES: ", SCALES)
#SCALES = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

"""
PROMPTS:  ['a woman walking in the desert near red rocks']
SCALES:  [tensor(0.7600)]
PROMPTS:  ['a field of grass with dew drops on it']
SCALES:  [tensor(-0.0103)]

"""

print("PROMPTS: ", PROMPTS)
print("SCALES: ", SCALES)

def main():
    # sd1.4 default scheduler
    noise_scheduler = PNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        prediction_type="epsilon",
        set_alpha_to_one=False,
        skip_prk_steps=True,
        steps_offset=1,
        trained_betas=None
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", revision=revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision)
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", revision=revision
    )
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)
    vae.requires_grad_(False)

    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.


    # Move unet, vae and text_encoder to device and cast to weight_dtype
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    # prepare prompt
    prompts = []
    for prompt in PROMPTS:
        for obj in OBJECTS:
            prompts.append(prompt.format(obj))
    
    scales = SCALES
    

    # white background latent
    white_image = torch.zeros((1, 3, height, width)).to(device, dtype=weight_dtype)
    white_latents = vae.encode(white_image).latent_dist.sample()
    white_latents = 0.18215 * white_latents

    ## RUN 
    for prompt_id, prompt in enumerate(prompts):
        # for different seeds on same prompt
        for _ in range(num_images_per_prompt):
            #seed = random.randint(0, 5000)
            for lora_id, lora_weight in enumerate(lora_weights):

            
                if 'full' in lora_weight:
                    train_method = 'full'
                elif 'noxattn' in lora_weight:
                    train_method = 'noxattn'
                else:
                    train_method = 'noxattn'

                network_type = "c3lier"
                if train_method == 'xattn':
                    network_type = 'lierla'

                modules = DEFAULT_TARGET_REPLACE
                if network_type == "c3lier":
                    modules += UNET_TARGET_REPLACE_MODULE_CONV
                import os
                model_name = lora_weight

                name = os.path.basename(model_name)
                unet = UNet2DConditionModel.from_pretrained(
                    pretrained_model_name_or_path, subfolder="unet", revision=revision
                )
                # freeze parameters of models to save more memory
                unet.requires_grad_(False)
                unet.to(device, dtype=weight_dtype)
                rank = 4
                alpha = 1
                if 'rank1' in lora_weight:
                    rank = 1
                if 'rank2' in lora_weight:
                    rank = 2
                if 'rank4' in lora_weight:
                    rank = 4
                if 'rank8' in lora_weight:
                    rank = 8
                if 'rank16' in lora_weight:
                    rank = 16
                if 'rank64' in lora_weight:
                    rank = 64
                if 'rank256' in lora_weight:
                    rank = 256
                if 'alpha1' in lora_weight:
                    alpha = 1.0
                #network = LoRANetwork(
                network = LoRAMappingNetwork(
                        unet,
                        learnable_matrix=NUM_BIN,
                        rank=rank,
                        multiplier=1.0,
                        alpha=alpha,
                        train_method=train_method,
                        global_input_dim=1
                    ).to(device, dtype=weight_dtype)
                
                c = torch.load(lora_weight)
                # d = c['learnable_matrix'].clone()
                # a = network.learnable_matrix.clone()
                network.load_state_dict(c)
                # b = network.learnable_matrix.clone()
                # print("======================================")
                # print("IS WEIGHTS THE SAME?")
                # print((torch.abs(a - b) < 1e-8).all())
                # print((torch.abs(b.cpu() - d.cpu()) < 1e-8).all())
                # print((torch.abs(a.cpu() - d.cpu()) < 1e-8).all())
                # generator = torch.Generator().manual_seed(0)
                # e = torch.normal(0, 1, size=(2, 4, 768), generator=generator)
                # print((torch.abs(e.cpu() - d.cpu()) < 1e-8).all())
                # print((torch.abs(e[0].cpu() - e[1].cpu()) < 1e-8).all())
                # print("======================================")
                # exit()

                network.set_lora_slider(scale=1.0)
                images_list = []

                os.makedirs(output_dir, exist_ok=True)

                for scale_id, scale in enumerate(scales):
                    #if len(SCALES) > 1:
                    #output_dir = f"output/chkpt100/512_unsplash250_mapnetlearnmatrix_interpolate_chkpt100_lr{LEARNING_RATE}_bin{NUM_BIN}_overfit{PROMPT_ID:02d}_5seed/scale_0/gray"
                    #os.makedirs(output_dir, exist_ok=True)
                    #global_token = network.get_global_token(torch.tensor([scale]).to(device)) # [1,4,768 ]     
                    #print(global_token[0,:,0])
                    #exit()

                    # Get global_token 
                    bin_id = np.digitize(scale, ALL_BIN, right=True)
                    upper_bin = np.clip(bin_id - 1,0, np.inf) + 1
                    lower_bin = upper_bin - 1 

                    lower_bin = torch.tensor(lower_bin).to(device).long()
                    upper_bin = torch.tensor(upper_bin).to(device).long()
                   
                    scale = torch.tensor(scale).to(device).float()
                    #distance = np.abs((scale - ALL_BIN[lower_bin]) / (ALL_BIN[upper_bin] - ALL_BIN[lower_bin]))
                    distance = (scale - ALL_BIN_TENSOR[lower_bin]) / (ALL_BIN_TENSOR[upper_bin] - ALL_BIN_TENSOR[lower_bin])
                    
                    global_upper = network.get_global_token(torch.tensor(ALL_BIN_TENSOR[upper_bin]).to(device).long()).to(device) #[1,4,768]
                    global_lower = network.get_global_token(torch.tensor(ALL_BIN_TENSOR[lower_bin]).to(device).long()).to(device) #[1,4,768]
                    # print("=============================================")
                    # print("IS upper bin and lower bin the same?")
                    # print((torch.abs(global_upper - global_lower) < 1e-8).all())
                    # exit()
                    global_token = global_lower + distance * (global_upper - global_lower) #[1,4,768]
                    global_token = global_token.float()[None]
                    # print(global_token[0,:4,0])
                    # continue
               
                    for seed_id, seed in enumerate(SEEDS):
                        for guidance_scale in guidance_scales:
                            if len(guidance_scales) > 1:
                                fname = f"{lora_id:04d}_{prompt_id:04d}_{scale_id:04d}_{seed_id:04d}_g{guidance_scale:02.2f}.png"
                            elif len(lora_weights) > 1:
                                fname = f"{lora_id:04d}_{prompt_id:04d}_{scale_id:04d}_{seed_id:04d}.png"
                            else:
                                fname = f"{prompt_id:04d}_{scale_id:04d}_{seed_id:04d}.png"
                            if os.path.exists(os.path.join(output_dir, fname)):
                                continue

                            print(prompt, scale, seed)
                            generator = torch.manual_seed(seed) 
                            text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

                            text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

                            max_length = text_input.input_ids.shape[-1]
                            if negative_prompt is None:
                                uncond_input = tokenizer(
                                    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                                )
                            else:
                                uncond_input = tokenizer(
                                    [negative_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                                )
                            ori_uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
                            ori_text_embeddings = text_embeddings


                           #text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

                            latents = torch.randn(
                                (batch_size, unet.in_channels, height // 8, width // 8),
                                generator=generator,
                            )
                            latents = latents.to(torch_device)
                            
                            noise_scheduler.set_timesteps(ddim_steps)

                            latents = latents * noise_scheduler.init_noise_sigma
                            latents = latents.to(weight_dtype)
                            latent_model_input = torch.cat([latents] * 2)
                            
                            #register_cross_attention_hook(unet)
                            for t in tqdm(noise_scheduler.timesteps):
                                if t> start_noise or t < stop_noise:
                                    lora_scale = torch.tensor([[0.0]])
                                else:
                                    lora_scale = torch.tensor([[scale]])
                                #network.set_lora_slider(scale=lora_scale)
                                    
                                #global_token = network.get_global_token(lora_scale.to(device))
                                new_uncond_embeddings = torch.cat([ori_uncond_embeddings,global_token],axis=-2)
                                new_text_embeddings = torch.cat([ori_text_embeddings,global_token],axis=-2)
                                text_embeddings = torch.cat([new_uncond_embeddings, new_text_embeddings])


                                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                                latent_model_input = torch.cat([latents] * 2)

                                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)
                                # predict the noise residual
                                with network:
                                    with torch.no_grad():
                                        #noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                                        unet_out = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)
                                        noise_pred = unet_out.sample
                                # perform guidance
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                if guidance_scale > 0:
                                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                                else:
                                    #noise_pred = noise_pred_uncond
                                    noise_pred = noise_pred_text

                                # compute the previous noisy sample x_t -> x_t-1
                                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

                            # scale and decode the image latents with vae
                            latents = 1 / 0.18215 * latents
                            with torch.no_grad():
                                image = vae.decode(latents).sample
                            image = (image / 2 + 0.5)
                            image = image.clamp(0, 1)
                            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()

                            images = (image * 255).round().astype("uint8")
                            pil_images = [Image.fromarray(image) for image in images]
                            
                            pil_images[0].save(os.path.join(output_dir, fname))

                            # dir_name = "attn_maps"
                            # net_attn_maps = get_net_attn_map(pil_images[0].size)
                            
                            # net_attn_maps = resize_net_attn_map(net_attn_maps, pil_images[0].size)
                            # save_net_attn_map(net_attn_maps, dir_name, tokenizer, prompt)
                            

                del network, unet
                unet = None
                network = None
                torch.cuda.empty_cache()
                flush()

if __name__ == "__main__":
    main()