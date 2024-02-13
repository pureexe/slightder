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

CONDITION_IMAGES = [
    "datasets/scale1_2/unsplash2000_ball/1.00000000/image.png",
    "datasets/scale1_2/unsplash2000_ball/2.84577371/image.png"
]
#SCALES = [1.0, 2.0]
SEEDS = [148, 200, 201, 305, 312]
#SEEDS = range(300, 320)

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
from transformers import ConvNextImageProcessor, ConvNextModel

class LoRAWithConvHead(LoRANetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_head = torch.nn.Linear(1024 * 7 * 7, 1)

### CONFIGURATION
width = 512
height = 512 
steps = 50  
cfg_scale = 7.5 
pretrained_sd_model = "CompVis/stable-diffusion-v1-4"
pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"

revision = None
device = 'cuda:0'
rank = 4
weight_dtype = torch.float32

lora_weights = [
    "models/unsplash2000_latent_alpha1.0_rank4_noxattn/unsplash2000_latent_alpha1.0_rank4_noxattn_9500steps.pt"
]
output_dir = "output/unsplash2000_latent/chkpt9500/"
PROMPTS = [ 
   "a photo of {}, blank gray background, solid background, shadow, heavy shadow, cast shadow",
]
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
height = 512
width = 512
ddim_steps = 50
guidance_scale = 5.0 # OVERFIT TO GUIDANCE SCALE

def flush():
    torch.cuda.empty_cache()
    gc.collect()
flush()

def main():
    # loading ConvNext Pipeline
    convnext_processor = ConvNextImageProcessor.from_pretrained("facebook/convnext-base-224-22k")
    convnext_model = ConvNextModel.from_pretrained("facebook/convnext-base-224-22k").to("cuda")

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
    
    # white background latent
    white_image = torch.zeros((1, 3, 512, 512)).to(device, dtype=weight_dtype)
    white_latents = vae.encode(white_image).latent_dist.sample()
    white_latents = 0.18215 * white_latents

    ## RUN 
    for prompt_id, prompt in enumerate(prompts):
        # for different seeds on same prompt
        for _ in range(num_images_per_prompt):
            #seed = random.randint(0, 5000)
            for lora_weight in lora_weights:
            
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
                if 'alpha1' in lora_weight:
                    alpha = 1.0
                network = LoRAWithConvHead(
                        unet,
                        rank=rank,
                        multiplier=1.0,
                        alpha=alpha,
                        train_method=train_method,
                    ).to(device, dtype=weight_dtype)
                
                network.load_state_dict(torch.load(lora_weight))
                images_list = []

                os.makedirs(output_dir, exist_ok=True)

                for scale_id, condition_image_path in enumerate(CONDITION_IMAGES):
                    condition_image = Image.open(condition_image_path)
                    convnext_inputs = convnext_processor(condition_image, return_tensors="pt")
                    convnext_inputs = {k: v.to("cuda") for k, v in convnext_inputs.items()}
                    convnext_outputs = convnext_model(**convnext_inputs)
                    convnext_last_hidden_states = convnext_outputs.last_hidden_state
                    lora_scale = network.conv_head(convnext_last_hidden_states.view(-1, 1024 * 7 * 7))
                    scale = lora_scale
                    for seed_id, seed in enumerate(SEEDS):
                        print(prompt, condition_image_path, scale, seed)
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
                        #backup_latents = latents.clone()
                        latents = noise_scheduler.add_noise(white_latents, latents, torch.tensor([999]).to(torch_device))

                        #latents[:,:,16:48,16:48] = backup_latents[:,:,16:48,16:48]

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
                        pil_images[0].save(os.path.join(output_dir, f"{prompt_id:04d}_{scale_id:04d}_{seed_id:04d}.png"))

                del network, unet
                unet = None
                network = None
                torch.cuda.empty_cache()
                flush()

if __name__ == "__main__":
    main()