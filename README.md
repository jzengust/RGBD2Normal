CVPR2019 Deep Surface Normal Estimation with Hierarchical RGB-D Fusion
====
# Paper
**Deep Surface Normal Estimation with Hierarchical RGB-D Fusion** (CVPR2019) [[Project page](https://huangyunmu.github.io/HFMNet_CVPR2019/)]


![Architecture](https://huangyunmu.github.io/HFMNet_CVPR2019/res/framework_v5_cut.png)


# File Organization
    |-- models: model structures
    |-- pre_trained: pretrain models
    |-- loader: dataloaders
    |-- datalist: train/test list for matterport and scannet
	              testsmall: used for fast checking
    |-- checkpoint/
        FCONV_MS: RGBD model for 
	        |-- matterport (fconv_ms_matterport_l1_2_hybrid_best.pkl) 
	        |-- scannet (fconv_ms_scannet_l1_1_hybrid_best.pkl)
        RGB_resume: RGB model for 
            |-- matterport (vgg_16_in_matterport_l1_2_in_RGB_best.pkl) 
	        |-- scannet (vgg_16_in_scannet_l1_2_resume_RGB_best.pkl)
	|-- sample_bash: bash sample for training and testing
	|-- demo_rgbd_mp.sh/demo_rgbd_sc.sh , demo script
	|-- sample_pic: sample pictures for demo
    |-- train_RGB.py, training for RGB model 
    |-- test_RGB.py, testing for RGB model
    |-- train_RGBD_ms.py, training for RGBD model
    |-- test_RGBD_ms.py, testing for RGBD model
    |-- test_RGBD_ms_sc.py, testing for RGBD model for object details
    |-- config.json, config file for dataset filepath.
    |-- utils.py, misc

Note: due to upload size limit, pretrain model and checkpoint are not uploaded. 
Please refer to Model section for model download.


# Requirement 

* python 2.7
* torch 0.4.1
* torchvision 0.2.1
* scikit-learn 0.20.3
* numpy
* matplotlib
* tqdm
* tensorboardX

# Dataset 
Two datasets are used: Matterport and Scannet. Please refer to [[this website](http://deepcompletion.cs.princeton.edu/)] for data download.<br>
After data download, please config the data path in config.json.

# Model 

Download the model and pretrain file from [[Google Drive](https://drive.google.com/drive/folders/1dSdTR_ezhXgEjG7n5hrmku5Mey5ZZJCr)] 

# Demo 
1. Please set up environment and download the model.
2. run demo_rgbd_mp.sh. Result will be saved in ./result/demo_rgbd_mp/
3. run demo_rgbd_sc.sh. Result will be saved in ./result/demo_rgbd_sc/

# Usage
Please set the data path in config.json

Sample bash files can be found in ./sample_bash for training and testing.
You may use them directly or follow the below instruction to make your own
training/testing scripts.  

To train/test on matterport, 

1. training:
   * a. train RGB model with l1 loss (train_RGB.py)
   * b. train fusion network with map_conv (no hybrid, just l1 at final scale)
      use RGB model for RGB encoder
      several epoch will be okay (5~10)
   * c. use the model above as pretrained model, apply hybrid (5~10)
   
2. testing: 
   run test_RGBD.py<br>
   func eval_normal_pixel for evaluation metric computation<br>
   Metric result saved in ./model_path/model_name.csv, as the below sequence: <br>
   
   
   Mean error|Median error|Percentage of error less than 11.25&deg;|22.5&deg;|30&deg; <br>
   
   The average result is stored in the last row. For metric detail meaning, please refer to paper
   
To train/test on scannet
1. training: 
   * a. train RGB model with l1 loss (train_RGB.py)
   * b. train fusion network with map_conv (train_RGBD_ms.py, no hybrid, just l1 at final scale)
      use RGB model for RGB encoder
      several epoch will be okay (5~10)
   * c. use the model above as pretrained model, apply hybrid(train_RGBD_ms.py,5~10 epoch)
   
2. testing: run test_RGBD_ms.py<br>
   func eval_normal_pixel for evaluation metric computation<br>
   Average metric result saved in ./model_path/model_name.csv
   
3. Additionally, run test_RGBD_ms_sc.py <br>
   For object detail normal computation. Pixels belonging to bed, sofa, chair are computed separately<br>

# Citation
        @inproceedings{zeng2019deep,
          title={Deep Surface Normal Estimation with Hierarchical RGB-D Fusion},
          author={Zeng, Jin and Tong, Yanfeng and Huang, Yunmu and Yan, Qiong and Sun, Wenxiu and Chen, Jing and Wang, Yongtian},
          booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
          year={2019}
        } 

# Contact 
Jin Zeng, jzeng2010@gmail.com<br>
Yunmu Huang, hymlearn@gmail.com

