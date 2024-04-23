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




CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'profiling_test' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash_cast_250' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash2000_blip2.json'


### TRAIN BLIP REMASTER
CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'unsplash250_sh1_rank4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash250_sh1' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash250_sh1_blip2.json'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'unsplash250_sh1_rank2' --rank 2 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash250_sh1' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash250_sh1_blip2.json'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'unsplash250_sh1_rank1' --rank 1 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash250_sh1' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash250_sh1_blip2.json'

CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'shoe401_12_1.0_2.0_gc_removed' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_shoe.yaml' --folder_main 'datasets/v1/scale1_2/shoe401_12_1.0_2.0' --folders='_' --scales='_'


# TRAIN WITH MULTIPLE SPLIT
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'unsplash250_sh1_chkpt100' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash250_sh1' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash250_sh1_blip2.json'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'unsplash250_cast250_chkpt100' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash_cast_250' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash2000_blip2.json'

# DIFFERENET LEARNING RATE

CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_unsplash250_sh1_chkpt100_lr1e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-3.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash250_sh1' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash250_sh1_blip2.json'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_unsplash250_sh1_chkpt100_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash250_sh1' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash250_sh1_blip2.json'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_unsplash250_sh1_chkpt100_lr1e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-5.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash250_sh1' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash250_sh1_blip2.json'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_unsplash250_sh1_chkpt100_lr5e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-3.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash250_sh1' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash250_sh1_blip2.json'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_unsplash250_sh1_chkpt100_lr5e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-4.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash250_sh1' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash250_sh1_blip2.json'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_unsplash250_sh1_chkpt100_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash250_sh1' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash250_sh1_blip2.json'



CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_unsplash250_cast_chkpt100_lr1e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-3.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash_cast_250' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash2000_blip2.json'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_unsplash250_cast_chkpt100_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash_cast_250' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash2000_blip2.json'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_unsplash250_cast_chkpt100_lr1e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-5.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash_cast_250' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash2000_blip2.json'
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_unsplash250_cast_chkpt100_lr5e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-3.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash_cast_250' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash2000_blip2.json'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_unsplash250_cast_chkpt100_lr5e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-4.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash_cast_250' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash2000_blip2.json'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_unsplash250_cast_chkpt100_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash_cast_250' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash2000_blip2.json'

# HDR MODE

CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_unsplash250_hdr_chkpt100_lr1e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-3.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash250_hdr' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unsplash250_hdr.json'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_unsplash250_hdr_chkpt100_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash250_hdr' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unsplash250_hdr.json'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_unsplash250_hdr_chkpt100_lr1e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-5.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash250_hdr' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unsplash250_hdr.json'
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_unsplash250_hdr_chkpt100_lr5e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-3.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash250_hdr' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unsplash250_hdr.json'
CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_unsplash250_hdr_chkpt100_lr5e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-4.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash250_hdr' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unsplash250_hdr.json'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-scale_classic.py --name '512_unsplash250_hdr_chkpt100_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash250_hdr' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unsplash250_hdr.json'



## CAST UP 250
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-singlescale_cond.py --name '512_unsplash250_cast_singlescale_chkpt100_lr1e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-3.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash_cast_250' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash2000_blip2.json' --image_size 512
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-singlescale_cond.py --name '512_unsplash250_cast_singlescale_chkpt100_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash_cast_250' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash2000_blip2.json' --image_size 512
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-singlescale_cond.py --name '512_unsplash250_cast_singlescale_chkpt100_lr1e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-5.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash_cast_250' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash2000_blip2.json' --image_size 512
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-singlescale_cond.py --name '512_unsplash250_cast_singlescale_chkpt100_lr5e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-3.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash_cast_250' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash2000_blip2.json' --image_size 512
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-singlescale_cond.py --name '512_unsplash250_cast_singlescale_chkpt100_lr5e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-4.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash_cast_250' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash2000_blip2.json' --image_size 512
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-singlescale_cond.py --name '512_unsplash250_cast_singlescale_chkpt100_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash_cast_250' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash2000_blip2.json' --image_size 512



CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-doublescale_cond.py --name '512_unsplash250_cast_doublescale_chkpt100_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/unsplash250cast' --image_size 512
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-doublescale_cond.py --name '512_unsplash250_cast_doublescale_chkpt100_lr5e-5_singlescale_fallback' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/unsplash250cast' --image_size 512


CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-singlescale_cond.py --name 'v2_512_unsplash250_cast_singlescale_chkpt100_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v1/scale-1_1/unsplash_cast_250' --folders='_' --scales='_' --prompt_file 'datasets/v1/scale-1_1/unplash2000_blip2.json' --image_size 512


CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet.py --name '512_unsplash250_mapnet_single_chkpt100_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v2/unsplash250cast'
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-mapnet.py --name '512_unsplash250_mapnet_single_chkpt100_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/unsplash250cast'


CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name '512_unsplash250_mapnetlearnmatrix_single_chkpt100_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/unsplash250cast'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name '512_unsplash250_mapnetlearnmatrix_single_chkpt100_lr5e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-4.yaml' --folder_main 'datasets/v2/unsplash250cast'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name '512_unsplash250_mapnetlearnmatrix_single_chkpt100_lr5e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-3.yaml' --folder_main 'datasets/v2/unsplash250cast'
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name '512_unsplash250_mapnetlearnmatrix_single_chkpt100_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v2/unsplash250cast'
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name '512_unsplash250_mapnetlearnmatrix_single_chkpt100_lr1e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-5.yaml' --folder_main 'datasets/v2/unsplash250cast'


CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name '512_unsplash250_mapnetlearnmatrix_interpolate_chkpt100_lr1e-4_bin4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v2/unsplash250cast' --num_of_bin 4
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name '512_unsplash250_mapnetlearnmatrix_interpolate_chkpt100_lr1e-4_bin2' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v2/unsplash250cast' --num_of_bin 2
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name '512_unsplash250_mapnetlearnmatrix_interpolate_chkpt100_lr1e-4_bin8' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v2/unsplash250cast' --num_of_bin 8
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name '512_unsplash250_mapnetlearnmatrix_interpolate_chkpt100_lr1e-4_bin16' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v2/unsplash250cast' --num_of_bin 16


CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name '512_unsplash250_mapnetlearnmatrix_interpolate_chkpt100_lr5e-5_bin4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/unsplash250cast' --num_of_bin 4
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name '512_unsplash250_mapnetlearnmatrix_interpolate_chkpt100_lr5e-5_bin2' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/unsplash250cast' --num_of_bin 2
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name '512_unsplash250_mapnetlearnmatrix_interpolate_chkpt100_lr5e-5_bin8' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/unsplash250cast' --num_of_bin 8
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name '512_unsplash250_mapnetlearnmatrix_interpolate_chkpt100_lr5e-5_bin16' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/unsplash250cast' --num_of_bin 16


#  fit the textural inversion
CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-textural_inversion_default.py --name 'textural_inversion_two_gril_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/textural/2girls'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-textural_inversion_default.py --name 'textural_inversion_sprout_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/textural/sprout'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-textural_inversion_default.py --name 'textural_inversion_sprout_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/textural/beach'

CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-textural_inversion_default.py --name 'textural_inversion_2girls_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v2/textural/2girls'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-textural_inversion_default.py --name 'textural_inversion_2girls_lr5e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-4.yaml' --folder_main 'datasets/v2/textural/2girls'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-textural_inversion_default.py --name 'textural_inversion_2girls_lr1e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-3.yaml' --folder_main 'datasets/v2/textural/2girls'
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-textural_inversion_default.py --name 'textural_inversion_2girls_lr5e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-3.yaml' --folder_main 'datasets/v2/textural/2girls'

CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-textural_inversion_default.py --name 'textural_inversion_beach_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v2/textural/beach'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-textural_inversion_default.py --name 'textural_inversion_beach_lr5e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-4.yaml' --folder_main 'datasets/v2/textural/beach'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-textural_inversion_default.py --name 'textural_inversion_beach_lr1e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-3.yaml' --folder_main 'datasets/v2/textural/beach'
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-textural_inversion_default.py --name 'textural_inversion_beach_lr5e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-3.yaml' --folder_main 'datasets/v2/textural/beach'

# FIt single lora interpolate experiment
CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_2girls_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/textural/2girls' --num_of_bin 2
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_2girls_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v2/textural/2girls' --num_of_bin 2
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_2girls_lr5e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-4.yaml' --folder_main 'datasets/v2/textural/2girls' --num_of_bin 2
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_2girls_lr1e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-3.yaml' --folder_main 'datasets/v2/textural/2girls' --num_of_bin 2

CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_beach_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/textural/beach' --num_of_bin 2
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_beach_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v2/textural/beach' --num_of_bin 2
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_beach_lr5e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-4.yaml' --folder_main 'datasets/v2/textural/beach' --num_of_bin 2
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_beach_lr1e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-3.yaml' --folder_main 'datasets/v2/textural/beach' --num_of_bin 2

CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_sprout_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/textural/sprout' --num_of_bin 2
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_sprout_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v2/textural/sprout' --num_of_bin 2
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_sprout_lr5e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-4.yaml' --folder_main 'datasets/v2/textural/sprout' --num_of_bin 2
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_sprout_lr1e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-3.yaml' --folder_main 'datasets/v2/textural/sprout' --num_of_bin 2


CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_sprout_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/textural/sprout' --num_of_bin 2
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_sprout_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v2/textural/sprout' --num_of_bin 2
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_sprout_lr5e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-4.yaml' --folder_main 'datasets/v2/textural/sprout' --num_of_bin 2
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_sprout_lr1e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-3.yaml' --folder_main 'datasets/v2/textural/sprout' --num_of_bin 2


# Experiment for 2scene
CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_2scenes_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/textural/2scenes' --num_of_bin 2
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_2scenes_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v2/textural/2scenes' --num_of_bin 2
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_2scenes_lr5e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-4.yaml' --folder_main 'datasets/v2/textural/2scenes' --num_of_bin 2
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_2scenes_lr1e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-3.yaml' --folder_main 'datasets/v2/textural/2scenes' --num_of_bin 2

# Unsplash 250
CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_unsplash250cast_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/unsplash250cast' --num_of_bin 2
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_unsplash250cast_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v2/unsplash250cast' --num_of_bin 2
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_unsplash250cast_lr5e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-4.yaml' --folder_main 'datasets/v2/unsplash250cast' --num_of_bin 2
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_unsplash250cast_lr1e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-3.yaml' --folder_main 'datasets/v2/unsplash250cast' --num_of_bin 2

# 2scenesvs 

CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'v2_mapnetlearnmatrix_interpolate_chkpt100_2scenes_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/textural/2scenes' --num_of_bin 2
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'v2_mapnetlearnmatrix_interpolate_chkpt100_2scenes_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v2/textural/2scenes' --num_of_bin 2
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'v2_mapnetlearnmatrix_interpolate_chkpt100_2scenes_lr5e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-4.yaml' --folder_main 'datasets/v2/textural/2scenes' --num_of_bin 2
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'v2_mapnetlearnmatrix_interpolate_chkpt100_2scenes_lr1e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-3.yaml' --folder_main 'datasets/v2/textural/2scenes' --num_of_bin 2

# Unsplash 250v2
CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'v2_mapnetlearnmatrix_interpolate_chkpt100_unsplash250cast_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/unsplash250cast' --num_of_bin 2
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'v2_mapnetlearnmatrix_interpolate_chkpt100_unsplash250cast_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v2/unsplash250cast' --num_of_bin 2
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'v2_mapnetlearnmatrix_interpolate_chkpt100_unsplash250cast_lr5e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-4.yaml' --folder_main 'datasets/v2/unsplash250cast' --num_of_bin 2
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'v2_mapnetlearnmatrix_interpolate_chkpt100_unsplash250cast_lr1e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-3.yaml' --folder_main 'datasets/v2/unsplash250cast' --num_of_bin 2


# Baseline of textural inversion 
CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-textural_inversion_classic.py --name 'textural_inversion_classic_beach_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v2/textural/beach'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-textural_inversion_classic.py --name 'textural_inversion_classic_beach_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-3.yaml' --folder_main 'datasets/v2/textural/beach'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-textural_inversion_classic.py --name 'textural_inversion_classic_beach_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-4.yaml' --folder_main 'datasets/v2/textural/beach'
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-textural_inversion_classic.py --name 'textural_inversion_classic_beach_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/textural/beach'

CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-textural_inversion_classic.py --name 'textural_inversion_classic_2girls_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v2/textural/2girls'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-textural_inversion_classic.py --name 'textural_inversion_classic_2girls_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-3.yaml' --folder_main 'datasets/v2/textural/beach'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-textural_inversion_classic.py --name 'textural_inversion_classic_2girls_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-4.yaml' --folder_main 'datasets/v2/textural/beach'
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-textural_inversion_classic.py --name 'textural_inversion_classic_2girls_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/textural/beach'


CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'v2_mapnetlearnmatrix_interpolate_chkpt100_2scenes_no_prompt_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/textural/2scenes_no_prompt' --num_of_bin 2
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'v2_mapnetlearnmatrix_interpolate_chkpt100_2scenes_no_prompt__lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v2/textural/2scenes_no_prompt' --num_of_bin 2
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'v2_mapnetlearnmatrix_interpolate_chkpt100_2scenes_no_prompt__lr5e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-4.yaml' --folder_main 'datasets/v2/textural/2scenes_no_prompt' --num_of_bin 2
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'v2_mapnetlearnmatrix_interpolate_chkpt100_2scenes_no_prompt__lr1e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-3.yaml' --folder_main 'datasets/v2/textural/2scenes_no_prompt' --num_of_bin 2

# NO prompt 0,1
CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_2scenes_no_prompt_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-5.yaml' --folder_main 'datasets/v2/textural/2scenes_no_prompt_0_1' --num_of_bin 2
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_2scenes_no_prompt_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-4.yaml' --folder_main 'datasets/v2/textural/2scenes_no_prompt_0_1' --num_of_bin 2
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_2scenes_no_prompt_lr5e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-4.yaml' --folder_main 'datasets/v2/textural/2scenes_no_prompt_0_1' --num_of_bin 2
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt100_2scenes_no_prompt_lr1e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-3.yaml' --folder_main 'datasets/v2/textural/2scenes_no_prompt_0_1' --num_of_bin 2



CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name 'mapnetlearnmatrix_chkpt100_2scenes_lr2e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr2e-5.yaml' --folder_main 'datasets/v2/textural/2scenes'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name 'mapnetlearnmatrix_chkpt100_2scenes_lr1e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-5.yaml' --folder_main 'datasets/v2/textural/2scenes'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name 'mapnetlearnmatrix_chkpt100_2scenes_lr5e-6' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr5e-6.yaml' --folder_main 'datasets/v2/textural/2scenes'
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name 'mapnetlearnmatrix_chkpt100_2scenes_lr1e-6' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100_lr1e-6.yaml' --folder_main 'datasets/v2/textural/2scenes'


CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name 'mapnetlearnmatrix_100k_2scenes_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr5e-5.yaml' --folder_main 'datasets/v2/textural/2scenes'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name 'mapnetlearnmatrix_100k_2scenes_lr3e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr3e-5.yaml' --folder_main 'datasets/v2/textural/2scenes'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name 'mapnetlearnmatrix_100k_2scenes_lr2e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr2e-5.yaml' --folder_main 'datasets/v2/textural/2scenes'
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name 'mapnetlearnmatrix_100k_2scenes_lr1e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr1e-5.yaml' --folder_main 'datasets/v2/textural/2scenes'

# interpolate on shoe401_360 from 1e-3 to 5e-5
CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt1000_shoe401_360_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr5e-5.yaml' --folder_main 'datasets/v2/shoe401_360' --num_of_bin 2
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt1000_shoe401_360_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr1e-4.yaml' --folder_main 'datasets/v2/shoe401_360' --num_of_bin 2
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt1000_shoe401_360_lr5e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr5e-4.yaml' --folder_main 'datasets/v2/shoe401_360' --num_of_bin 2
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt1000_shoe401_360_lr1e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr1e-3.yaml' --folder_main 'datasets/v2/shoe401_360' --num_of_bin 2
7
# interpolate on shoe401_few from 1e-3 to 5e-5
CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt1000_shoe401_few_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr5e-5.yaml' --folder_main 'datasets/v2/shoe401_few' --num_of_bin 2
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt1000_shoe401_few_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr1e-4.yaml' --folder_main 'datasets/v2/shoe401_few' --num_of_bin 2
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt1000_shoe401_few_lr5e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr5e-4.yaml' --folder_main 'datasets/v2/shoe401_few' --num_of_bin 2
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt1000_shoe401_few_lr1e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr1e-3.yaml' --folder_main 'datasets/v2/shoe401_few' --num_of_bin 2


# TEST shoe with 2 pair to see if thing still working correctly or not 
CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name 'mapnetlearnmatrix_interpolate_chkpt1000_shoe401_two_image_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr5e-5.yaml' --folder_main 'datasets/v2/shoe401_two_image'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name 'mapnetlearnmatrix_interpolate_chkpt1000_shoe401_two_image_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr1e-4.yaml' --folder_main 'datasets/v2/shoe401_two_image'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name 'mapnetlearnmatrix_interpolate_chkpt1000_shoe401_two_image_lr5e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr5e-4.yaml' --folder_main 'datasets/v2/shoe401_two_image'
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name 'mapnetlearnmatrix_interpolate_chkpt1000_shoe401_two_image_lr1e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr1e-3.yaml' --folder_main 'datasets/v2/shoe401_two_image'


# interpolate v2

CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt1000_shoe401_front_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr5e-5.yaml' --folder_main 'datasets/v2/shoe401_front' --num_of_bin 2
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt1000_shoe401_front_lr5e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr5e-4.yaml' --folder_main 'datasets/v2/shoe401_front' --num_of_bin 2
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt1000_shoe401_front_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr1e-4.yaml' --folder_main 'datasets/v2/shoe401_front' --num_of_bin 2
CUDA_VISIBLE_DEVICES=3 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix_interpolate.py --name 'mapnetlearnmatrix_interpolate_chkpt1000_shoe401_front_lr1e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr1e-3.yaml' --folder_main 'datasets/v2/shoe401_front' --num_of_bin 2


# test swap input image
CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name 'mapnetlearnmatrix_chkpt1000_2scenes_no_prompt_swap_image_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr5e-5.yaml' --folder_main 'datasets/v2/textural/2scenes_no_prompt_swap_image'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name 'mapnetlearnmatrix_chkpt1000_2scenes_no_prompt_swap_image_lr3e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr3e-5.yaml' --folder_main 'datasets/v2/textural/2scenes_no_prompt_swap_image'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name 'mapnetlearnmatrix_chkpt1000_2scenes_no_prompt_swap_image_lr2e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_100k_lr2e-5.yaml' --folder_main 'datasets/v2/textural/2scenes_no_prompt_swap_image'

# train with max step 1000
CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name 'mapnetlearnmatrix_denosing1000_2scenes_no_prompt_lr5e-5' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_denosing1000_lr5e-5.yaml' --folder_main 'datasets/v2/textural/2scenes_no_prompt'
CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name 'mapnetlearnmatrix_denosing1000_2scenes_no_prompt_lr5e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_denosing1000_lr5e-4.yaml' --folder_main 'datasets/v2/textural/2scenes_no_prompt'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name 'mapnetlearnmatrix_denosing1000_2scenes_no_prompt_lr1e-4' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_denosing1000_lr1e-4.yaml' --folder_main 'datasets/v2/textural/2scenes_no_prompt'
CUDA_VISIBLE_DEVICES=2 python trainscripts/imagesliders/train_lora-mapnet_learnmatrix.py --name 'mapnetlearnmatrix_denosing1000_2scenes_no_prompt_lr1e-3' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent_denosing1000_lr1e-3.yaml' --folder_main 'datasets/v2/textural/2scenes_no_prompt'