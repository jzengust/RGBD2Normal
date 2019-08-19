#!/bin/bash
cd ..
cd ..
project_dir=$(pwd)
log_file=${project_dir}/logs/log_mt_rgb_1.txt

python train_RGB.py --arch_RGB vgg_16_in --input rgb --pretrained --model_num 1 --batch_size 16 --loss l1 --dataset matterport --n_epoch 10 --img_rows 256 --img_cols 320 --num_workers 8 --writer mt_rgb_1 2>&1 | tee ${log_file} &
