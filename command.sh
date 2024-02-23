CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'unsplash2000' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/scale1_2/unsplash2000_raw' --folders='_' --scales='_'

CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'ball20k_latent' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/unsplash2000_raw' --folders='_' --scales='_'


CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-scale_ball_latent.py --name 'unsplash2000_latent' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/scale1_2/unsplash2000_raw' --folder_condition 'datasets/scale1_2/unsplash2000_ball' --folders='_' --scales='_'


CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'ball20k_latent' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/scale-1_1/unsplash2000_raw' --folders='_' --scales='_'


# CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'ball20k_latent' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/scale-1_1/unsplash2000_raw' --folders='_' --scales='_'


CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora_latents.py --name 'dev_latent' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/scale-1_1/unsplash2000_raw' --folder_latent 'datasets/scale-1_1/unsplash2000_convnext' --folders='_' --scales='_'


CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora_latents.py --name 'unsplash2000_lora_latent/clip_base' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/scale-1_1/unsplash2000_raw' --folder_latent 'datasets/scale-1_1/unsplash2000_clip' --folders='_' --scales='_'

CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora_latents.py --name 'unsplash2000_lora_latent/clip_base' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/scale-1_1/unsplash2000_raw' --folder_latent 'datasets/scale-1_1/unsplash2000_clip' --folders='_' --scales='_'
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora_latents.py --name 'unsplash2000_lora_latent/convnext_base' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/scale-1_1/unsplash2000_raw' --folder_latent 'datasets/scale-1_1/unsplash2000_convnext' --folders='_' --scales='_'


CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'unsplash2000_minus1_1' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent1m.yaml' --folder_main 'datasets/scale-1_1/unsplash2000_raw' --folders='_' --scales='_'


#shoe experiment
CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'shoe401_2_1.0_2.0' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_shoe.yaml' --folder_main 'datasets/scale1_2/shoe401_2_1.0_2.0' --folders='_' --scales='_'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'shoe401_4_1.0_2.0' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_shoe.yaml' --folder_main 'datasets/scale1_2/shoe401_4_1.0_2.0' --folders='_' --scales='_'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'shoe401_8_1.0_2.0' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_shoe.yaml' --folder_main 'datasets/scale1_2/shoe401_8_1.0_2.0' --folders='_' --scales='_'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'shoe401_12_1.0_2.0' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_shoe.yaml' --folder_main 'datasets/scale1_2/shoe401_12_1.0_2.0' --folders='_' --scales='_'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'shoe401_23_1.0_2.0' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_shoe.yaml' --folder_main 'datasets/scale1_2/shoe401_23_1.0_2.0' --folders='_' --scales='_'
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'shoe401_46_1.0_2.0' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_shoe.yaml' --folder_main 'datasets/scale1_2/shoe401_46_1.0_2.0' --folders='_' --scales='_'
CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'shoe401_91_1.0_2.0' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_shoe.yaml' --folder_main 'datasets/scale1_2/shoe401_91_1.0_2.0' --folders='_' --scales='_'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'shoe401_180_1.0_2.0' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_shoe.yaml' --folder_main 'datasets/scale1_2/shoe401_180_1.0_2.0' --folders='_' --scales='_'


# shoe experiment v2


CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_shoe401_2_1.0_2.0' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_shoe.yaml' --folder_main 'datasets/scale1_2/shoe401_2_1.0_2.0' --folders='_' --scales='_'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_shoe401_4_1.0_2.0' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_shoe.yaml' --folder_main 'datasets/scale1_2/shoe401_4_1.0_2.0' --folders='_' --scales='_'
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_shoe401_8_1.0_2.0' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_shoe.yaml' --folder_main 'datasets/scale1_2/shoe401_8_1.0_2.0' --folders='_' --scales='_'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_shoe401_12_1.0_2.0' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_shoe.yaml' --folder_main 'datasets/scale1_2/shoe401_12_1.0_2.0' --folders='_' --scales='_'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_shoe401_23_1.0_2.0' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_shoe.yaml' --folder_main 'datasets/scale1_2/shoe401_23_1.0_2.0' --folders='_' --scales='_'
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_shoe401_46_1.0_2.0' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_shoe.yaml' --folder_main 'datasets/scale1_2/shoe401_46_1.0_2.0' --folders='_' --scales='_'
CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_shoe401_91_1.0_2.0' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_shoe.yaml' --folder_main 'datasets/scale1_2/shoe401_91_1.0_2.0' --folders='_' --scales='_'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_shoe401_180_1.0_2.0' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_shoe.yaml' --folder_main 'datasets/scale1_2/shoe401_180_1.0_2.0' --folders='_' --scales='_'

# train with text

CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'unsplash2000_with_text' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100.yaml' --folder_main 'datasets/scale1_2/unsplash2000_raw' --folders='_' --scales='_' --prompt_file 'datasets/scale1_2/unplash2000_blip2.json'



# Experiment 
# LoRA RANK 4, 16, 64, 256

CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'unsplash_cast_250_rank4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/scale-1_1/unsplash_cast_250' --folders='_' --scales='_' --prompt_file 'datasets/scale-1_1/unplash2000_blip2.json'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'unsplash_cast_250_rank16' --rank 16 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/scale-1_1/unsplash_cast_250' --folders='_' --scales='_' --prompt_file 'datasets/scale-1_1/unplash2000_blip2.json'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'unsplash_cast_250_rank64' --rank 64 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/scale-1_1/unsplash_cast_250' --folders='_' --scales='_' --prompt_file 'datasets/scale-1_1/unplash2000_blip2.json'
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'unsplash_cast_250_rank256' --rank 256 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/scale-1_1/unsplash_cast_250' --folders='_' --scales='_' --prompt_file 'datasets/scale-1_1/unplash2000_blip2.json'



CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'unsplash2000_-1.0_1.0_rank4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/scale-1_1/unsplash2000_raw' --folders='_' --scales='_'  --prompt_file 'datasets/scale-1_1/unplash2000v2_blip2.json'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'unsplash2000_-1.0_1.0_rank16' --rank 16 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/scale-1_1/unsplash2000_raw' --folders='_' --scales='_'  --prompt_file 'datasets/scale-1_1/unplash2000v2_blip2.json'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'unsplash2000_-1.0_1.0_rank64' --rank 64 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/scale-1_1/unsplash2000_raw' --folders='_' --scales='_'  --prompt_file 'datasets/scale-1_1/unplash2000v2_blip2.json'
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'unsplash2000_-1.0_1.0_rank256' --rank 256 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/scale-1_1/unsplash2000_raw' --folders='_' --scales='_'  --prompt_file 'datasets/scale-1_1/unplash2000v2_blip2.json'

