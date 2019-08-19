#!/bin/bash
python test_RGBD_ms.py --arch_F fconv_ms --arch_map map_conv --no_testset --img_path ./sample_pic/sc_rgb/ --depth_path ./sample_pic/sc_depth/ --d_scale 10000 --img_rows 240 --img_cols 320 --model_savepath ./checkpoint/FCONV_MS/  --model_full_name fconv_ms_scannet_l1_1_hybrid_best.pkl --out_path ./result/demo_rgbd_sc/
