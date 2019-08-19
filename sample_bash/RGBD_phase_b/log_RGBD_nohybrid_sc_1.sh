#!/bin/bash
cd ..
cd ..
project_dir=$(pwd)
log_file=${project_dir}/logs/log_sc_nohybrid_1.txt

python train_RGBD_ms.py --resume --resume_model_path ./checkpoint/instance_norm/vgg_16_in_scannet_l1_2_resume_RGB_best.pkl --arch_RGB vgg_16_in --arch_map map_conv --model_num 1 --batch_size 4 --loss l1 --no_hybrid_loss --dataset scannet --n_epoch 10 --img_rows 240 --img_cols 320 --num_workers 8 --writer sc_nohy_1 2>&1 | tee ${log_file} &
