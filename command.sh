CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'unsplash2000' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/scale1_2/unsplash2000_raw' --folders='_' --scales='_'

CUDA_VISIBLE_DEVICES=0 python trainscripts/imagesliders/train_lora-scale_classic.py --name 'ball20k_latent' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/unsplash2000_raw' --folders='_' --scales='_'


CUDA_VISIBLE_DEVICES=1 python trainscripts/imagesliders/train_lora-scale_ball_latent.py --name 'unsplash2000_latent' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config_latent.yaml' --folder_main 'datasets/scale1_2/unsplash2000_raw' --folder_condition 'datasets/scale1_2/unsplash2000_ball' --folders='_' --scales='_'
