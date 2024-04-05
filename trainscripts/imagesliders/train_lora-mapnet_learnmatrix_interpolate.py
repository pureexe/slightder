# Single scale Condition into  LORA Arxhitech

# ref:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L566
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py

from typing import List, Optional
import argparse
import ast
from pathlib import Path
import gc

import torch
from tqdm import tqdm
import os, glob

from lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
from pure_util.lora_global_adapter import LoRAMappingNetwork
from pure_util.datasets.image_axis3 import ImageAxis3Dataset
import train_util
import model_util
import prompt_util
from prompt_util import PromptEmbedsCache, PromptEmbedsPair, PromptSettings
import debug_util
import config_util
from config_util import RootConfig
import random
import numpy as np
import wandb
from PIL import Image
import json
import time
from torch.utils.data import Dataset, DataLoader
import torchvision

def flush():
    torch.cuda.empty_cache()
    gc.collect()

def train(
    config: RootConfig,
    prompts: list[PromptSettings],
    device: int,
    folder_main: str,
    folders,
    scales,
    prompt_file: Optional[str] = None,
    image_size = 512,
    num_of_bin = 4
):
    scales = np.array(scales)
    folders = np.array(folders)
    scales_unique = list(scales)

    metadata = {
        "prompts": ",".join([prompt.json() for prompt in prompts]),
        "config": config.json(),
    }
    save_path = Path(config.save.path)

    modules = DEFAULT_TARGET_REPLACE
    if config.network.type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV

    if config.logging.verbose:
        print(metadata)

    if config.logging.use_wandb:
        wandb.init(project=f"LECO_{config.save.name}", config=metadata)

    weight_dtype = config_util.parse_precision(config.train.precision)
    save_weight_dtype = config_util.parse_precision(config.train.precision)

    tokenizer, text_encoder, unet, noise_scheduler, vae = model_util.load_models(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
        v2=config.pretrained_model.v2,
        v_pred=config.pretrained_model.v_pred,
    )

    text_encoder.to(device, dtype=weight_dtype)
    text_encoder.eval()

    unet.to(device, dtype=weight_dtype)
    unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()
    
    vae.to(device)
    vae.requires_grad_(False)
    vae.eval()

    network = LoRAMappingNetwork(
        unet,
        learnable_matrix=num_of_bin,
        rank=config.network.rank,
        multiplier=1.0,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
        global_input_dim=1
    ).to(device, dtype=weight_dtype)
    ALL_BIN = np.linspace(-1, 1, num_of_bin)
    ALL_BIN_TENSOR = torch.tensor(ALL_BIN).to(device)
    network.set_lora_slider(scale=1.0) #Set LoRA Scale to default

    optimizer_module = train_util.get_optimizer(config.train.optimizer)
    optimizer_kwargs = {}
    if config.train.optimizer_args is not None and len(config.train.optimizer_args) > 0:
        for arg in config.train.optimizer_args.split(" "):
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value
            
    optimizer = optimizer_module(network.prepare_optimizer_params(), lr=config.train.lr, **optimizer_kwargs)
    lr_scheduler = train_util.get_lr_scheduler(
        config.train.lr_scheduler,
        optimizer,
        max_iterations=config.train.iterations,
        lr_min=config.train.lr / 100,
    )
    criteria = torch.nn.MSELoss()

    print("Prompts")
    for settings in prompts:
        print(settings)

    # debug
    debug_util.check_requires_grad(network)
    debug_util.check_training_mode(network)

    cache = PromptEmbedsCache()
    prompt_pairs: list[PromptEmbedsPair] = []

    with torch.no_grad():
        for settings in prompts:
            print(settings)
            for prompt in [
                settings.target,
                settings.positive,
                settings.neutral,
                settings.unconditional,
            ]:
                print(prompt)
                if isinstance(prompt, list):
                    if prompt == settings.positive:
                        key_setting = 'positive'
                    else:
                        key_setting = 'attributes'
                    if len(prompt) == 0:
                        cache[key_setting] = []
                    else:
                        if cache[key_setting] is None:
                            cache[key_setting] = train_util.encode_prompts(
                                tokenizer, text_encoder, prompt
                            )
                else:
                    if cache[prompt] == None:
                        cache[prompt] = train_util.encode_prompts(
                            tokenizer, text_encoder, [prompt]
                        )

            prompt_pairs.append(
                PromptEmbedsPair(
                    criteria,
                    cache[settings.target],
                    cache[settings.positive],
                    cache[settings.unconditional],
                    cache[settings.neutral],
                    settings,
                )
            )

    prompt_pair: PromptEmbedsPair = prompt_pairs[0]
    dataset = ImageAxis3Dataset(root_dir=folder_main, image_size = (image_size,image_size), is_normalize=True)
    dataloader = DataLoader(dataset, batch_size=prompt_pair.batch_size, shuffle=False, num_workers=4)
    data_iter = iter(dataloader)
    pbar = tqdm(range(config.train.iterations))
    for i in pbar:
        
        # Fetching new batch 
        try:
            batch = next(data_iter) 
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader 
            data_iter = iter(dataloader)
            batch = next(data_iter) 
    

        with torch.no_grad():
            positive_embed = train_util.encode_prompts(
                tokenizer, text_encoder, batch['text'][0]
            )

            noise_scheduler.set_timesteps(
                config.train.max_denoising_steps, device=device
            )

            optimizer.zero_grad()

            # 1 ~ 49 からランダム
            timesteps_to = torch.randint(
                1, config.train.max_denoising_steps-1, (1,)
#                 1, 25, (1,)
            ).item()

            height, width = (
                prompt_pair.resolution,
                prompt_pair.resolution,
            )
            if prompt_pair.dynamic_resolution:
                height, width = train_util.get_random_resolution_in_bucket(
                    prompt_pair.resolution
                )

            if config.logging.verbose:
                print("guidance_scale:", prompt_pair.guidance_scale)
                print("resolution:", prompt_pair.resolution)
                print("dynamic_resolution:", prompt_pair.dynamic_resolution)
                if prompt_pair.dynamic_resolution:
                    print("bucketed resolution:", (height, width))
                print("batch_size:", prompt_pair.batch_size)


            img2 = torchvision.transforms.functional.to_pil_image(batch['image'][0])
            seed = random.randint(0,2*15)

            generator = torch.manual_seed(seed)
            denoised_latents_high, high_noise = train_util.get_noisy_image(
                img2,
                vae,
                generator,
                unet,
                noise_scheduler,
                start_timesteps=0,
                total_timesteps=timesteps_to)
            denoised_latents_high = denoised_latents_high.to(device, dtype=weight_dtype)
            high_noise = high_noise.to(device, dtype=weight_dtype)
            noise_scheduler.set_timesteps(1000)

            current_timestep = noise_scheduler.timesteps[
                int(timesteps_to * 1000 / config.train.max_denoising_steps)
            ]

        #network.set_lora_slider()
    
        #TODO: TEMPORARY DISABLE BATCH SIZE 
        scale = batch['coeff'].numpy()[0,:1] #[B,1]
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

        global_token = global_lower + distance * (global_upper - global_lower) #[1,4,768]
        global_token = global_token.float()

        # compute global token
        #global_token = network.get_global_token(batch['id'][...,:1].to(device)) # [1,4,768 ]

        uncond_embed = torch.cat([prompt_pair.unconditional, global_token], axis=-2)
        positive_embed = torch.cat([prompt_pair.positive, global_token], axis=-2)


        with network:
            target_latents_high = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_high,
                train_util.concat_embeddings(
                    uncond_embed,
                    positive_embed, #prompt_pair.positive,
                    prompt_pair.batch_size,
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)
            
            
        
        loss_high = criteria(target_latents_high, high_noise.cpu().to(torch.float32))
        pbar.set_description(f"Loss*1k: {loss_high.item()*1000:.4f}")
        loss_high.backward()
        
        
        optimizer.step()
        lr_scheduler.step()

        if (
            i % config.save.per_steps == 0
            and i != 0
            and i != config.train.iterations - 1
        ):
            print("Saving...")
            save_path.mkdir(parents=True, exist_ok=True)
            network.save_weights(
                save_path / f"{config.save.name}_{i}steps.pt",
                dtype=save_weight_dtype,
            )

    print("Saving...")
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        save_path / f"{config.save.name}_last.pt",
        dtype=save_weight_dtype,
    )

    del (
        unet,
        noise_scheduler,
        optimizer,
        network,
    )

    flush()

    print("Done.")


def main(args):
    config_file = args.config_file

    config = config_util.load_config_from_yaml(config_file)
    if args.name is not None:
        config.save.name = args.name
    attributes = []
    if args.attributes is not None:
        attributes = args.attributes.split(',')
        attributes = [a.strip() for a in attributes]
    
    config.network.alpha = args.alpha
    config.network.rank = args.rank
    config.save.name += f'_alpha{args.alpha}'
    config.save.name += f'_rank{config.network.rank }'
    config.save.name += f'_{config.network.training_method}'
    config.save.path += f'/{config.save.name}'

    prompts = prompt_util.load_prompts_from_yaml(config.prompts_file, attributes)
    device = torch.device(f"cuda:{args.device}")
    
    if args.folders == "_":
        folders = ",".join(os.listdir(args.folder_main))
        args.folders = folders
        args.scales = folders

    folders = args.folders.split(',')
    folders = [f.strip() for f in folders]
    scales = args.scales.split(',')
    scales = [f.strip() for f in scales]
    #scales = [int(s) for s in scales]
    
    print(folders, scales)
    if len(scales) != len(folders):
        raise Exception('the number of folders need to match the number of scales')
    
    if args.stylecheck is not None:
        check = args.stylecheck.split('-')
        
        for i in range(int(check[0]), int(check[1])):
            folder_main = args.folder_main+ f'{i}'
            config.save.name = f'{os.path.basename(folder_main)}'
            config.save.name += f'_alpha{args.alpha}'
            config.save.name += f'_rank{config.network.rank }'
            config.save.path = f'models/{config.save.name}'
            train(config=config, prompts=prompts, device=device, folder_main = folder_main, prompt_file = args.prompt_file, image_size = args.image_size, num_of_bin=args.num_of_bin)
    else:
        train(config=config, prompts=prompts, device=device, folder_main = args.folder_main, folders = folders, scales = scales, prompt_file = args.prompt_file, image_size = args.image_size, num_of_bin=args.num_of_bin)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=False,
        default = 'data/config.yaml',
        help="Config file for training.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="LoRA weight.",
    )
    
    parser.add_argument(
        "--rank",
        type=int,
        required=False,
        help="Rank of LoRA.",
        default=4,
    )
    
    parser.add_argument(
        "--device",
        type=int,
        required=False,
        default=0,
        help="Device to train on.",
    )
    
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default=None,
        help="Device to train on.",
    )
    
    parser.add_argument(
        "--attributes",
        type=str,
        required=False,
        default=None,
        help="attritbutes to disentangle",
    )
    
    parser.add_argument(
        "--folder_main",
        type=str,
        required=True,
        help="The folder to check",
    )
    
    parser.add_argument(
        "--stylecheck",
        type=str,
        required=False,
        default = None,
        help="The folder to check",
    )
    
    parser.add_argument(
        "--folders",
        type=str,
        required=False,
        default = 'verylow, low, high, veryhigh',
        help="folders with different attribute-scaled images",
    )
    parser.add_argument(
        "--scales",
        type=str,
        required=False,
        default = '-2, -1,1, 2',
        help="scales for different attribute-scaled images",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=False,
        default = None,
        help="prompt file for text",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        required=False,
        default = 512,
        help="size of image to resize",
    )
    parser.add_argument(
        "--num_of_bin",
        type=int,
        required=False,
        default = 4,
        help="number of matrix bin",
    )
    
    
    args = parser.parse_args()

    main(args)
