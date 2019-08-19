Sample bash file
====
Please run at project root

# Test RGB Model

## Test RGB Model with Matterport Dataset
python test_RGB.py --arch_RGB vgg_16_in --model_full_name vgg_16_in_matterport_l1_2_in_RGB_best.pkl --model_savepath ./checkpoint/resume_RGB/ --img_rows 256 --img_cols 320 --test_dataset matterport --test_split testsmall --result_path ./result/ --numerical_result_path vgg_16_in_matterport_l1_2_result.txt

## Test RGB Model with Scannet Dataset
python test_RGB.py --arch_RGB vgg_16_in --model_full_name vgg_16_in_scannet_l1_2_resume_RGB_best.pkl --model_savepath ./checkpoint/resume_RGB/ --img_rows 240 --img_cols 320 --test_dataset scannet --test_split testsmall --result_path ./result/ --numerical_result_path vgg_16_in_scannet_l1_2_result.txt

## Test RGB Model with single image 
python test_RGB.py --arch_RGB vgg_16_in --model_full_name vgg_16_in_matterport_l1_2_in_RGB_best.pkl --model_savepath ./checkpoint/resume_RGB/ --img_rows 256 --img_cols 320 --no_testset --img_path ./sample_pic/mt_rgb/0affd60e9e4249858dc9bf84639fe86f_i1_2.jpg --out_path ./temp/output.jpg

## Test RGB Model with image list
python test_RGB.py --arch_RGB vgg_16_in --model_full_name vgg_16_in_matterport_l1_2_in_RGB_best.pkl --model_savepath ./checkpoint/resume_RGB/ --no_testset --img_dataroot ./sample_pic/mt_rgb/ --img_datalist ./sample_pic/mp_sample_list.txt --out_path ./temp/


# Test RGBD Model

## Test RGBD Model with Matterport Dataset
python test_RGBD_ms.py --arch_F fconv_ms --arch_map map_conv --dataset matterport --model_savepath ./checkpoint/FCONV_MS/ --model_full_name fconv_ms_matterport_l1_2_hybrid_best.pkl --testset_out_path ./result/ --test_split testsmall 

## Test RGBD Model with Scannet Dataset 
python test_RGBD_ms.py --arch_F fconv_ms --arch_map map_conv --dataset scannet --img_rows 240 --img_cols 320 --model_savepath ./checkpoint/FCONV_MS/  --model_full_name fconv_ms_scannet_l1_1_hybrid_best.pkl --testset_out_path ./result/ --test_split testsmall 


## Test RGBD Model with customized images
python test_RGBD_ms.py --arch_F fconv_ms --arch_map map_conv --no_testset --img_path ./sample_pic/mt_rgb/ --depth_path ./sample_pic/mt_depth/ --d_scale 40000 --img_rows 256 --img_cols 320 --model_savepath ./checkpoint/FCONV_MS/  --model_full_name fconv_ms_matterport_l1_2_hybrid_best.pkl --out_path ./result/demo_rgbd_mp/ 


python test_RGBD_ms.py --arch_F fconv_ms --arch_map map_conv --no_testset --img_path ./sample_pic/sc_rgb/ --depth_path ./sample_pic/sc_depth/ --d_scale 10000 --img_rows 240 --img_cols 320 --model_savepath ./checkpoint/FCONV_MS/  --model_full_name fconv_ms_scannet_l1_1_hybrid_best.pkl --out_path ./result/demo_rgbd_sc/


