OBJECTS = [
    'shoe',
    'cake',
    'bottle',
    'chair',
    'cup',
    'laptop',
    'cell phone',
    'keyboard',
    'book',
    'scissors',
]

#SCALES = [1.0, 2.5]
#SCALES = [-0.5, 0.5]
#SCALES = [1.0, 3.0]
#SCALES = [0.0]
#SCALES = [1.0, 3.0]
SCALES = [-1.0, 1.0]
#SCALES = [1.0, 3.0]


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

### CONFIGURATION
#width = 512
#height = 512 
width = 256
height = 256 
steps = 50  
cfg_scale = 3
pretrained_sd_model = "CompVis/stable-diffusion-v1-4"
pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"

revision = None
device = 'cuda:0'
rank = 4
weight_dtype = torch.float32

lora_weights = [
    #"models/unsplash2000_alpha1.0_rank4_noxattn/unsplash2000_alpha1.0_rank4_noxattn_30000steps.pt"
    #"models/ball20k_latent_alpha1.0_rank4_noxattn/ball20k_latent_alpha1.0_rank4_noxattn_30000steps.pt",
    #"models/unsplash2000_alpha1.0_rank4_noxattn/unsplash2000_alpha1.0_rank4_noxattn_30000steps.pt",
]

#RANK = "4"
#RANK = "16"
#RANK = "64"
RANK = "256"
#STR = "180"
#CHECKPOINT = "19500"
CHECKPOINT = "16000"
#CHECKPOINT = "20000"
#CHECKPOINT = "19000"
lora_weights = [
    #f"models/shoe401_{STR}_1.0_2.0_alpha1.0_rank4_noxattn/shoe401_{STR}_1.0_2.0_alpha1.0_rank4_noxattn_last.pt"
    #f"models/shoe401_{STR}_1.0_2.0_alpha1.0_rank4_noxattn/shoe401_{STR}_1.0_2.0_alpha1.0_rank4_noxattn_{CHECKPOINT}steps.pt"
    #f"models/unsplash2000_with_text_alpha1.0_rank4_noxattn/unsplash2000_with_text_alpha1.0_rank4_noxattn_{CHECKPOINT}steps.pt"
    #"models/ball20k_latent_alpha1.0_rank4_noxattn/ball20k_latent_alpha1.0_rank4_noxattn_1000steps.pt"
    #f"models/unsplash_cast_250_rank{RANK}_alpha1.0_rank{RANK}_noxattn/unsplash_cast_250_rank{RANK}_alpha1.0_rank{RANK}_noxattn_{CHECKPOINT}steps.pt"
    f"models/unsplash2000_-1.0_1.0_rank{RANK}_alpha1.0_rank{RANK}_noxattn/unsplash2000_-1.0_1.0_rank{RANK}_alpha1.0_rank{RANK}_noxattn_{CHECKPOINT}steps.pt"
]

#lora_weights = [f"models/ball20k_latent_alpha1.0_rank4_noxattn/ball20k_latent_alpha1.0_rank4_noxattn_{idx}steps.pt" for idx in range(500, 100500, 500)]

#output_dir = "output/unsplash2000_1_3/chkpt10000_-1_3_standard"
#output_dir = "output/unsplash2000_-1_1/chkpt30000_-1.0_1.0_nosolid_background_noshadow/"
#output_dir = "output/unsplash2000_-1_1/chkpt30000_-1.0_1.0"
#output_dir = "output/unsplash2000_1_3/chkpt30000_1_3_nosolid_background_noshadow/"
#output_dir = "output/unsplash2000/chkpt30000_-0.5_0.5/"

#output_dir = "output/unsplash2000_-1_1/chkpt1000_-1.0_1.0_nosolid_background_noshadow_256"
#output_dir = "output/unsplash2000_1_3/chkpt30000_1_3"
#output_dir = "output/unsplash2000_-1_1/vary_checkpoint_nosolid_background_noshadow"
#output_dir = f"output/unsplash2000_with_text/chkpt{CHECKPOINT}_background"
#output_dir = f"output/rank_experiment/unsplash2000/{RANK}/chkpt{CHECKPOINT}_background"
#output_dir = f"output/rank_experiment/cast250/{RANK}/chkpt{CHECKPOINT}"
output_dir = f"output/rank_experiment/unsplash2000/{RANK}/chkpt{CHECKPOINT}"


PROMPTS = [ 
    #"a photo of {}, blank gray background, solid background, shadow, heavy shadow, cast shadow",
    #"a photo of {}, blank gray background, solid background",
    #"a photo of {}, shadow, heavy shadow, cast shadow",
    "a photo of {}",
    # "a black bottle with a red stripe on a gray background"
]
# SCALES = np.linspace(1.0, 2.0, 180)

# timestep during inference when we switch to LoRA scale>0 (this is done to ensure structure in the images)
start_noise = 999

# seed for random number generator
seed = 0

#number of images per prompt
num_images_per_prompt = 1

torch_device = device
#negative_prompt = None
negative_prompt = "fake, wax, cartoon, shadow, clutter, painting, logo, low quality"
batch_size = 1
ddim_steps = 50
#guidance_scale = 5.0 # OVERFIT TO GUIDANCE SCALE
guidance_scale = 4.0 # OVERFIT TO GUIDANCE SCALE

def flush():
    torch.cuda.empty_cache()
    gc.collect()
flush()

def main():
    # sd1.4 default scheduler
    noise_scheduler = PNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        #clip_sample=False,
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
                network = LoRANetwork(
                        unet,
                        rank=rank,
                        multiplier=1.0,
                        alpha=alpha,
                        train_method=train_method,
                    ).to(device, dtype=weight_dtype)
                network.load_state_dict(torch.load(lora_weight))
                images_list = []

                os.makedirs(output_dir, exist_ok=True)

                for scale_id, scale in enumerate(scales):
                    # if scale_id == 0:
                    #     continue
                    for seed_id, seed in enumerate(SEEDS):
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
                        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

                        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

                        latents = torch.randn(
                            (batch_size, unet.in_channels, height // 8, width // 8),
                            generator=generator,
                        )
                        latents = latents.to(torch_device)
                        
                        #latents = noise_scheduler.add_noise(white_latents, latents, torch.tensor([999]).to(torch_device))
                        



                        noise_scheduler.set_timesteps(ddim_steps)

                        latents = latents * noise_scheduler.init_noise_sigma
                        latents = latents.to(weight_dtype)
                        latent_model_input = torch.cat([latents] * 2)
                        
                        for t in tqdm(noise_scheduler.timesteps):
                            if t>start_noise:
                                network.set_lora_slider(scale=0)
                            else:
                                network.set_lora_slider(scale=scale)
                            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                            latent_model_input = torch.cat([latents] * 2)

                            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)
                            # predict the noise residual
                            with network:
                                with torch.no_grad():
                                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                            # perform guidance
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            if guidance_scale > 0:
                                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                            else:
                                noise_pred = noise_pred_uncond

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
                        #images_list.append(pil_images[0])
                        if len(lora_weights) > 1:
                            fname = f"{lora_id:04d}_{prompt_id:04d}_{scale_id:04d}_{seed_id:04d}.png"
                        else:
                            fname = f"{prompt_id:04d}_{scale_id:04d}_{seed_id:04d}.png"
                        pil_images[0].save(os.path.join(output_dir, fname))

                del network, unet
                unet = None
                network = None
                torch.cuda.empty_cache()
                flush()

if __name__ == "__main__":
    main()