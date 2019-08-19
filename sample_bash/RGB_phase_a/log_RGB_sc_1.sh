#!/bin/bash
cd ..
cd ..
project_dir=$(pwd)
log_file=${project_dir}/logs/log_sc_rgb_1.txt

python train_RGB.py --arch_RGB vgg_16_in --input rgb --pretrained --model_num 1 --batch_size 16 --loss l1 --dataset scannet --n_epoch 10 --num_workers 8 --writer sc_rgb_1 2>&1 | tee ${log_file} &
