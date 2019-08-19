#!/bin/bash
cd ..
cd ..

project_dir=$(pwd)
log_file=${project_dir}/logs/log_mt1_hybrid.txt

python train_RGBD_ms.py --resume --resume_model_path ./checkpoint/FCONV_MS/fconv_ms_matterport_l1_2_nohybrid_resume_best.pkl --arch_map map_conv --model_num 1 --batch_size 4 --loss l1 --hybrid_loss --dataset matterport --n_epoch 10 --img_rows 256 --img_cols 320 --num_workers 8 --writer mt1_hy_conv 2>&1 | tee ${log_file} &
