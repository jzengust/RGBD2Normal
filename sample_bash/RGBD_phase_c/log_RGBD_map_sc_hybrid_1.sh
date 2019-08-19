#!/bin/bash
cd ..
cd ..
project_dir=$(pwd)
log_file=${project_dir}/logs/log_sc1_hybrid.txt

python train_RGBD_ms.py --resume --resume_model_path ./checkpoint/FCONV_MS/fconv_ms_scannet_l1_2_nohybrid_resume_best.pkl --arch_map map_conv --model_num 1 --batch_size 4 --loss l1 --hybrid_loss --dataset scannet --n_epoch 10 --img_rows 240 --img_cols 320 --num_workers 8 --writer sc1_hy_conv 2>&1 | tee ${log_file} &
