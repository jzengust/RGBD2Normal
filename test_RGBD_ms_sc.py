##########################
# Test normal estimation
# RGBD input
# coupled with train_RGBD_ms.py
# Jin Zeng, 20181109
#########################

import sys, os
import torch
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from os.path import join as pjoin
import scipy.io as io

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from models import get_model, get_lossfun
from loader import get_data_path, get_loader
from pre_trained import get_premodel
from utils import norm_imsave    
from models.eval import eval_normal_pixel, eval_normal_detail, eval_print, eval_mask_resize
from loader.loader_utils import png_reader_32bit, png_reader_uint8

# from sync_batchnorm import DataParallelWithCallback

def test(args):
    
    # Setup Model
    model_name_F = args.arch_F
    model_F = get_model(model_name_F,True) # concat and output
    model_F = torch.nn.DataParallel(model_F, device_ids=range(torch.cuda.device_count()))
    if args.arch_map == 'map_conv' or args.arch_map == 'hybrid':
        model_name_map = 'map_conv'
        model_map = get_model(model_name_map,True) # concat and output
        model_map = torch.nn.DataParallel(model_map, device_ids=range(torch.cuda.device_count()))            

    if args.model_full_name != '':
        # Use the full name of model to load
        print("Load training model: " + args.model_full_name)
        checkpoint = torch.load(pjoin(args.model_savepath, args.model_full_name))
        model_F.load_state_dict(checkpoint['model_F_state'])
        model_map.load_state_dict(checkpoint["model_map_state"])
    
    # Setup image
    if args.imgset:
        print("Test on dataset: {}".format(args.dataset))
        data_loader = get_loader(args.dataset)
        data_path = get_data_path(args.dataset)
        v_loader = data_loader(data_path, split=args.test_split, img_size=(args.img_rows,args.img_cols), img_norm=args.img_norm,mode='seg')
        evalloader = data.DataLoader(v_loader, batch_size=1)
        print("Finish Loader Setup")        
        
        model_F.cuda()
        model_F.eval()
        if args.arch_map == 'map_conv' or args.arch_map == 'hybrid':
            model_map.cuda()    
            model_map.eval()

        sum_mean, sum_median, sum_small, sum_mid, sum_large, sum_num = [], [], [], [], [], []
        sum_mean_b, sum_median_b, sum_small_b, sum_mid_b, sum_large_b, sum_num_b = [], [], [], [], [], []
        sum_mean_s, sum_median_s, sum_small_s, sum_mid_s, sum_large_s, sum_num_s = [], [], [], [], [], []
        sum_mean_c, sum_median_c, sum_small_c, sum_mid_c, sum_large_c, sum_num_c = [], [], [], [], [], []
        evalcount = 0
        with torch.no_grad():
            for i_val, (images_val, labels_val, masks_val, valids_val, depthes_val, segment_val) in tqdm(enumerate(evalloader)):

                # if i_val>10:
                #     break                

                images_val = Variable(images_val.contiguous().cuda())
                labels_val = Variable(labels_val.contiguous().cuda())
                masks_val = Variable(masks_val.contiguous().cuda())
                valids_val = Variable(valids_val.contiguous().cuda())
                depthes_val = Variable(depthes_val.contiguous().cuda()) 
                segment_val = Variable(segment_val.contiguous().cuda())                
                
                # Bed:11 1191 494 786 1349
                # Sofa:6 1313
                # Chair:2 10 23 74 885 1184 1291 1338   
                segment_bed = torch.eq(segment_val,11)+torch.eq(segment_val,1191)+torch.eq(segment_val,494)+torch.eq(segment_val,786)+torch.eq(segment_val,1349) 
                segment_sofa = torch.eq(segment_val,6)+torch.eq(segment_val,1313)
                segment_chair = torch.eq(segment_val,2)+torch.eq(segment_val,10)+torch.eq(segment_val,23)+torch.eq(segment_val,74)+torch.eq(segment_val,885)+torch.eq(segment_val,1184)+torch.eq(segment_val,1291)+torch.eq(segment_val,1338)
                if segment_val.shape != masks_val.shape:                    
                    segment_bed = eval_mask_resize(segment_bed, args.img_rows, args.img_cols)
                    segment_sofa = eval_mask_resize(segment_sofa, args.img_rows, args.img_cols)
                    segment_chair = eval_mask_resize(segment_chair, args.img_rows, args.img_cols)
                
                if args.arch_map == 'map_conv' or  args.arch_map == 'hybrid':
                    outputs_valid = model_map(torch.cat((depthes_val, valids_val[:,np.newaxis,:,:]), dim=1))
                    outputs, outputs1, outputs2, outputs3,output_d = model_F(images_val, depthes_val, outputs_valid.squeeze(1))
                else:
                    outputs, outputs1, outputs2, outputs3,output_d = model_F(images_val, depthes_val, valids_val)

                outputs_n, pixelnum, mean_i, median_i, small_i, mid_i, large_i = eval_normal_pixel(outputs, labels_val, masks_val)

                masks_bed_val = segment_bed.to(torch.float64)*masks_val   
                masks_sofa_val = segment_sofa.to(torch.float64)*masks_val        
                masks_chair_val = segment_chair.to(torch.float64)*masks_val 
                _, pixelnum_b, mean_i_b, median_i_b, small_i_b, mid_i_b, large_i_b = eval_normal_pixel(outputs, labels_val, masks_bed_val)
                _, pixelnum_s, mean_i_s, median_i_s, small_i_s, mid_i_s, large_i_s = eval_normal_pixel(outputs, labels_val, masks_sofa_val) 
                _, pixelnum_c, mean_i_c, median_i_c, small_i_c, mid_i_c, large_i_c = eval_normal_pixel(outputs, labels_val, masks_chair_val) 
   
                   
                # outputs_norm = np.squeeze(outputs_n.data.cpu().numpy(), axis=0)         
                # labels_val_norm = np.squeeze(labels_val.data.cpu().numpy(), axis=0)                
                # images_val = np.squeeze(images_val.data.cpu().numpy(), axis=0)
                # images_val = images_val+0.5
                # images_val = images_val.transpose(1, 2, 0)
                # depthes_val = np.squeeze(depthes_val.data.cpu().numpy(), axis=0)
                # depthes_val = np.transpose(depthes_val, [1,2,0])
                # depthes_val = np.repeat(depthes_val, 3, axis = 2)
                # masks_bed_val = np.squeeze(masks_bed_val.data.cpu().numpy(), axis=0)   
                # masks_sofa_val = np.squeeze(masks_sofa_val.data.cpu().numpy(), axis=0) 
                # masks_chair_val = np.squeeze(masks_chair_val.data.cpu().numpy(), axis=0)  
                # outputs_valid = np.squeeze(outputs_valid.data.cpu().numpy(), axis=0)
                # outputs_valid = np.transpose(outputs_valid, [1,2,0])
                # outputs_valid = np.repeat(outputs_valid, 3, axis = 2) 
                # if (i_val+1)%1 == 0:
                    # misc.imsave(pjoin(args.testset_out_path, "{}_fms_hybrid2.png".format(i_val+1)), outputs_norm)
                    # misc.imsave(pjoin(args.testset_out_path, "{}_fms1_l1.png".format(i_val+1)), outputs_norm1)
                    # misc.imsave(pjoin(args.testset_out_path, "{}_fms2_l1.png".format(i_val+1)), outputs_norm2)
                    # misc.imsave(pjoin(args.testset_out_path, "{}_fms3_l1.png".format(i_val+1)), outputs_norm3)
                    # misc.imsave(pjoin(args.testset_out_path, "{}_fmsd_l1.png".format(i_val+1)), outputs_normd)
                    # misc.imsave(pjoin(args.testset_out_path, "{}_d_l1_imgout.png".format(i_val+1)), outputs_norm)
                    # misc.imsave(pjoin(args.testset_out_path, "{}_gt.png".format(i_val+1)), labels_val_norm)
                    # misc.imsave(pjoin(args.testset_out_path, "{}_in.jpg".format(i_val+1)), images_val)
                    # # misc.imsave(pjoin(args.testset_out_path, "{}_depth.png".format(i_val+1)), depthes_val)
                    # misc.imsave(pjoin(args.testset_out_path, "{}_bed.png".format(i_val+1)), masks_bed_val)
                    # misc.imsave(pjoin(args.testset_out_path, "{}_sofa.png".format(i_val+1)), masks_sofa_val)
                    # misc.imsave(pjoin(args.testset_out_path, "{}_chair.png".format(i_val+1)), masks_chair_val)
                    # misc.imsave(pjoin(args.testset_out_path, "{}_ms_conf.png".format(i_val+1)), outputs_valid)
                    # if i_val == 0:
                    #     outputs_mat = outputs_n.data.cpu().numpy()
                    # else:
                    #     outputs_mat = np.concatenate((outputs_mat,outputs_n.data.cpu().numpy()), axis=0)

                # accumulate the metrics in matrix
                if ((np.isnan(mean_i))|(np.isinf(mean_i)) == False):
                    sum_mean.append(mean_i)
                    sum_median.append(median_i)
                    sum_small.append(small_i)
                    sum_mid.append(mid_i)
                    sum_large.append(large_i)
                    sum_num.append(pixelnum)

                    sum_mean_b.append(mean_i_b)
                    sum_median_b.append(median_i_b)
                    sum_small_b.append(small_i_b)
                    sum_mid_b.append(mid_i_b)
                    sum_large_b.append(large_i_b)
                    sum_num_b.append(pixelnum_b)

                    sum_mean_s.append(mean_i_s)
                    sum_median_s.append(median_i_s)
                    sum_small_s.append(small_i_s)
                    sum_mid_s.append(mid_i_s)
                    sum_large_s.append(large_i_s)
                    sum_num_s.append(pixelnum_s)

                    sum_mean_c.append(mean_i_c)
                    sum_median_c.append(median_i_c)
                    sum_small_c.append(small_i_c)
                    sum_mid_c.append(mid_i_c)
                    sum_large_c.append(large_i_c)
                    sum_num_c.append(pixelnum_c)

                    evalcount+=1
                    if (i_val+1) % 10 == 0:
                        print("Iteration %d Evaluation Loss: mean %.4f, median %.4f, 11.25 %.4f, 22.5 %.4f, 30 %.4f" % (i_val+1, 
                            mean_i, median_i, small_i, mid_i, large_i))            

            # Summarize the result 
            eval_print(sum_mean, sum_median, sum_small, sum_mid, sum_large, sum_num, item='Pixel-Level') 
            eval_print(sum_mean_b, sum_median_b, sum_small_b, sum_mid_b, sum_large_b, sum_num_b, item='Bed-pixel') 
            eval_print(sum_mean_s, sum_median_s, sum_small_s, sum_mid_s, sum_large_s, sum_num_s, item='Sofa-pixel')
            eval_print(sum_mean_c, sum_median_c, sum_small_c, sum_mid_c, sum_large_c, sum_num_c, item='Chair-pixel')
           
            avg_mean = sum(sum_mean)/evalcount
            sum_mean.append(avg_mean)
            avg_median = sum(sum_median)/evalcount
            sum_median.append(avg_median)
            avg_small = sum(sum_small)/evalcount
            sum_small.append(avg_small)
            avg_mid = sum(sum_mid)/evalcount
            sum_mid.append(avg_mid)
            avg_large = sum(sum_large)/evalcount
            sum_large.append(avg_large)
            print("evalnum is %d, Evaluation Image-Level Mean Loss: mean %.4f, median %.4f, 11.25 %.4f, 22.5 %.4f, 30 %.4f" % (evalcount, 
                avg_mean, avg_median, avg_small, avg_mid, avg_large))                        

            sum_matrix = np.transpose([sum_mean,sum_median,sum_small,sum_mid,sum_large])

            if args.model_full_name != '':
                sum_file = args.model_full_name[:-4] + '.csv'
            np.savetxt(pjoin(args.model_savepath,sum_file), sum_matrix, fmt='%.6f', delimiter=',')
            print("Saving to %s" % (sum_file)) 

            # save normal output
            # sio.savemat('./result/scannet/RGBD_map.mat', {'outputs_mat':outputs_mat})           
            # end of dataset test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--arch_RGB', nargs='?', type=str, default='vgg_16_in', 
                        help='Architecture for RGB to use [\'vgg_16,vgg_16_in etc\']')
    parser.add_argument('--arch_D', nargs='?', type=str, default='unet_3_mask_in', 
                        help='Architecture for Depth to use [\'unet_3, unet_3_mask, unet_3_mask_in etc\']')
    parser.add_argument('--arch_F', nargs='?', type=str, default='fconv_ms', 
                        help='Architecture for Fusion to use [\'fconv,fconv_in, fconv_ms etc\']')
    parser.add_argument('--arch_map', nargs='?', type=str, default='hybrid', 
                        help='Architecture for confidence map to use [\'mask, map_conv, hybrid etc\']')
    parser.add_argument('--model_savepath', nargs='?', type=str, default='./checkpoint/FCONV_MS', 
                        help='Path for model saving [\'checkpoint etc\']')
    parser.add_argument('--model_full_name', nargs='?', type=str, default='',
                        help='The full name of the model to be tested.')
    parser.add_argument('--dataset', nargs='?', type=str, default='scannet', 
                        help='Dataset to use [\'nyuv2, matterport, scannet, etc\']')
    parser.add_argument('--test_split', nargs='?', type=str, default='', help='The split of dataset in testing')

    parser.add_argument('--loss', nargs='?', type=str, default='l1', 
                        help='Loss type: cosine, l1')
    parser.add_argument('--model_num', nargs='?', type=str, default='1', 
                        help='Checkpoint index [\'1,2,3, etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=240, 
                        help='Height of the input image, 256(mt), 240(nyu)')
    parser.add_argument('--img_cols', nargs='?', type=int, default=320, 
                        help='Width of the input image, 320(yinda and nyu)')
    
    parser.add_argument('--testset', dest='imgset', action='store_true', 
                        help='Test on set from dataloader, decided by --dataset | True by default')
    parser.add_argument('--no_testset', dest='imgset', action='store_false', 
                        help='Test on single image | True by default')
    parser.set_defaults(imgset=True)
    parser.add_argument('--testset_out_path', nargs='?', type=str, default='./result/sc_small', 
                        help='Path of the output normal')
    
    parser.add_argument('--img_path', nargs='?', type=str, default='../Depth2Normal/Dataset/normal/', 
                        help='Path of the input image')
    parser.add_argument('--depth_path', nargs='?', type=str, default='../Depth2Normal/Dataset/normal/', 
                        help='Path of the input image, mt_data_clean!!!!!!!!!')
    parser.add_argument('--ir_path', nargs='?', type=str, default='../Depth2Normal/Dataset/ir_mask/', 
                        help='Path of the input image, mt_data_clean!!!!!!!!!')
    parser.add_argument('--out_path', nargs='?', type=str, default='../Depth2Normal/Dataset/normal/', 
                        help='Path of the output normal')

    parser.add_argument('--img_norm', dest='img_norm', action='store_true', 
                        help='Enable input image scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest='img_norm', action='store_false', 
                        help='Disable input image scales normalization [0, 1] | True by default')
    parser.set_defaults(img_norm=True)

    parser.add_argument('--img_rotate', dest='img_rot', action='store_true', 
                        help='Enable input image transpose | False by default')
    parser.add_argument('--no-img_rotate', dest='img_rot', action='store_false', 
                        help='Disable input image transpose | False by default')
    parser.set_defaults(img_rot=True)
    
    args = parser.parse_args()
    test(args)
