##########################
# Test normal estimation
# RGBD input
# coupled with train_RGBD_ms.py
# Jin Zeng, 20181031
#########################

import sys, os
import torch
import argparse
import timeit
import numpy as np
import scipy.misc as misc
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
from utils import norm_imsave, change_channel
from models.eval import eval_normal_pixel, eval_print
from loader.loader_utils import png_reader_32bit, png_reader_uint8


def test(args):
    # Setup Model
    # Setup the fusion model (RGB+Depth)
    model_name_F = args.arch_F
    model_F = get_model(model_name_F, True)  # concat and output
    model_F = torch.nn.DataParallel(model_F, device_ids=range(torch.cuda.device_count()))
    # Setup the map model
    if args.arch_map == 'map_conv':
        model_name_map = args.arch_map
        model_map = get_model(model_name_map, True)  # concat and output
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
        v_loader = data_loader(data_path, split=args.test_split, img_size=(args.img_rows, args.img_cols),
                               img_norm=args.img_norm)
        evalloader = data.DataLoader(v_loader, batch_size=1)
        print("Finish Loader Setup")

        model_F.cuda()
        model_F.eval()
        if args.arch_map == 'map_conv':
            model_map.cuda()
            model_map.eval()

        sum_mean, sum_median, sum_small, sum_mid, sum_large, sum_num = [], [], [], [], [], []
        evalcount = 0
        with torch.no_grad():
            for i_val, (images_val, labels_val, masks_val, valids_val, depthes_val, meshdepthes_val) in tqdm(
                    enumerate(evalloader)):

                images_val = Variable(images_val.contiguous().cuda())
                labels_val = Variable(labels_val.contiguous().cuda())
                masks_val = Variable(masks_val.contiguous().cuda())
                valids_val = Variable(valids_val.contiguous().cuda())
                depthes_val = Variable(depthes_val.contiguous().cuda())

                if args.arch_map == 'map_conv':
                    outputs_valid = model_map(torch.cat((depthes_val, valids_val[:, np.newaxis, :, :]), dim=1))
                    outputs, outputs1, outputs2, outputs3, output_d = model_F(images_val, depthes_val,
                                                                              outputs_valid.squeeze(1))
                else:
                    outputs, outputs1, outputs2, outputs3, output_d = model_F(images_val, depthes_val, valids_val)

                outputs_n, pixelnum, mean_i, median_i, small_i, mid_i, large_i = eval_normal_pixel(outputs, labels_val,
                                                                                                   masks_val)
                outputs_norm = np.squeeze(outputs_n.data.cpu().numpy(), axis=0)
                labels_val_norm = np.squeeze(labels_val.data.cpu().numpy(), axis=0)
                images_val = np.squeeze(images_val.data.cpu().numpy(), axis=0)
                images_val = images_val + 0.5
                images_val = images_val.transpose(1, 2, 0)
                depthes_val = np.squeeze(depthes_val.data.cpu().numpy(), axis=0)
                depthes_val = np.transpose(depthes_val, [1, 2, 0])
                depthes_val = np.repeat(depthes_val, 3, axis=2)

                outputs_norm = change_channel(outputs_norm)
                labels_val_norm = (labels_val_norm + 1) / 2
                labels_val_norm = change_channel(labels_val_norm)

                # if (i_val+1)%10 == 0:
                misc.imsave(pjoin(args.testset_out_path, "{}_MS_hyb.png".format(i_val + 1)), outputs_norm)
                misc.imsave(pjoin(args.testset_out_path, "{}_gt.png".format(i_val + 1)), labels_val_norm)
                misc.imsave(pjoin(args.testset_out_path, "{}_in.jpg".format(i_val + 1)), images_val)
                misc.imsave(pjoin(args.testset_out_path, "{}_depth.png".format(i_val + 1)), depthes_val)

                # accumulate the metrics in matrix
                if ((np.isnan(mean_i)) | (np.isinf(mean_i)) == False):
                    sum_mean.append(mean_i)
                    sum_median.append(median_i)
                    sum_small.append(small_i)
                    sum_mid.append(mid_i)
                    sum_large.append(large_i)
                    sum_num.append(pixelnum)
                    evalcount += 1
                    if (i_val + 1) % 10 == 0:
                        print("Iteration %d Evaluation Loss: mean %.4f, median %.4f, 11.25 %.4f, 22.5 %.4f, 30 %.4f" % (
                            i_val + 1,
                            mean_i, median_i, small_i, mid_i, large_i))

                        # Summarize the result
            eval_print(sum_mean, sum_median, sum_small, sum_mid, sum_large, sum_num, item='Pixel-Level')

            avg_mean = sum(sum_mean) / evalcount
            sum_mean.append(avg_mean)
            avg_median = sum(sum_median) / evalcount
            sum_median.append(avg_median)
            avg_small = sum(sum_small) / evalcount
            sum_small.append(avg_small)
            avg_mid = sum(sum_mid) / evalcount
            sum_mid.append(avg_mid)
            avg_large = sum(sum_large) / evalcount
            sum_large.append(avg_large)
            print(
                    "evalnum is %d, Evaluation Image-Level Mean Loss: mean %.4f, median %.4f, 11.25 %.4f, 22.5 %.4f, 30 %.4f" % (
                evalcount,
                avg_mean, avg_median, avg_small, avg_mid, avg_large))

            sum_matrix = np.transpose([sum_mean, sum_median, sum_small, sum_mid, sum_large])
            if args.model_full_name != '':
                sum_file = args.model_full_name[:-4] + '.csv'

            np.savetxt(pjoin(args.model_savepath, sum_file), sum_matrix, fmt='%.6f', delimiter=',')
            print("Saving to %s" % (sum_file))
            # end of dataset test
    else:
        if os.path.isdir(args.out_path) == False:
            os.mkdir(args.out_path)
        print("Read Input Image from : {}".format(args.img_path))
        for i in os.listdir(args.img_path):
            if not i.endswith('.jpg'):
                continue

            print i
            input_f = args.img_path + i
            depth_f = args.depth_path + i[:-4] + '.png'
            output_f = args.out_path + i[:-4] + '_rgbd.png'
            img = misc.imread(input_f)

            orig_size = img.shape[:-1]
            if args.img_rot:
                img = np.transpose(img, (1, 0, 2))
                img = np.flipud(img)
                img = misc.imresize(img, (args.img_cols, args.img_rows))  # Need resize the image to model inputsize
            else:
                img = misc.imresize(img, (args.img_rows, args.img_cols))  # Need resize the image to model inputsize

            img = img.astype(np.float)
            if args.img_norm:
                img = (img - 128) / 255
            # NHWC -> NCHW
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, 0)
            img = torch.from_numpy(img).float()

            if args.img_rot:
                depth = png_reader_32bit(depth_f, (args.img_rows, args.img_cols))
                depth = np.transpose(depth, (1, 0))
                depth = np.flipud(depth)
                # valid = png_reader_uint8(mask_f, (args.img_rows,args.img_cols))
                # valid = np.transpose(valid, (1,0))
                # valid = np.flipud(valid)
            else:
                depth = png_reader_32bit(depth_f, (args.img_rows, args.img_cols))
                # valid = png_reader_uint8(mask_f, (args.img_rows,args.img_cols))

            depth = depth.astype(float)
            # Please change to the scale so that scaled_depth=1 corresponding to real 10m depth
            # matterpot depth=depth/40000  scannet depth=depth/10000
            depth = depth / (args.d_scale)
            if depth.ndim == 3:  # to dim 2
                depth = depth[:, :, 0]
                # if valid.ndim == 3: #to dim 2
            #     valid = valid[:,:,0]

            # valid = 1-depth
            # valid[valid>1] = 1
            valid = (depth > 0.0001).astype(float)
            # valid = depth.astype(float)
            depth = depth[np.newaxis, :, :]
            depth = np.expand_dims(depth, 0)
            valid = np.expand_dims(valid, 0)
            depth = torch.from_numpy(depth).float()
            valid = torch.from_numpy(valid).float()

            if torch.cuda.is_available():
                model_F.cuda()
                model_F.eval()
                if args.arch_map == 'map_conv':
                    model_map.cuda()
                    model_map.eval()
                images = Variable(img.contiguous().cuda())
                depth = Variable(depth.contiguous().cuda())
                valid = Variable(valid.contiguous().cuda())
            else:
                images = Variable(img)
                depth = Variable(depth)
                valid = Variable(valid)

            with torch.no_grad():
                if args.arch_map == 'map_conv':
                    outputs_valid = model_map(torch.cat((depth, valid[:, np.newaxis, :, :]), dim=1))
                    outputs, outputs1, outputs2, outputs3, output_d = model_F(images, depth,
                                                                              outputs_valid.squeeze(1))
                else:
                    outputs, outputs1, outputs2, outputs3, output_d = model_F(images, depth, outputs_valid)

            outputs_norm = norm_imsave(outputs)
            outputs_norm = np.squeeze(outputs_norm.data.cpu().numpy(), axis=0)
            # outputs_norm = misc.imresize(outputs_norm, orig_size)
            outputs_norm = change_channel(outputs_norm)
            misc.imsave(output_f, outputs_norm)
        print("Complete")
        # end of test on no dataset images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--arch_RGB', nargs='?', type=str, default='vgg_16_in',
                        help='Architecture for RGB to use [\'vgg_16,vgg_16_in etc\']')
    parser.add_argument('--arch_D', nargs='?', type=str, default='unet_3_mask_in',
                        help='Architecture for Depth to use [\'unet_3, unet_3_mask, unet_3_mask_in etc\']')
    parser.add_argument('--arch_F', nargs='?', type=str, default='fconv_ms',
                        help='Architecture for Fusion to use [\'fconv,fconv_in, fconv_ms etc\']')
    parser.add_argument('--arch_map', nargs='?', type=str, default='map_conv',
                        help='Architecture for confidence map to use [\'mask, map_conv etc\']')
    parser.add_argument('--model_savepath', nargs='?', type=str, default='./checkpoint/FCONV_MS',
                        help='Path for model saving [\'checkpoint etc\']')
    parser.add_argument('--model_full_name', nargs='?', type=str, default='',
                        help='The full name of the model to be tested.')

    parser.add_argument('--dataset', nargs='?', type=str, default='matterport',
                        help='Dataset to use [\'nyuv2, matterport, scannet, etc\']')
    parser.add_argument('--test_split', nargs='?', type=str, default='', help='The split of dataset in testing')

    parser.add_argument('--loss', nargs='?', type=str, default='l1',
                        help='Loss type: cosine, l1')
    parser.add_argument('--model_num', nargs='?', type=str, default='2',
                        help='Checkpoint index [\'1,2,3, etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256,
                        help='Height of the input image, 256(mt), 240(nyu)')
    parser.add_argument('--img_cols', nargs='?', type=int, default=320,
                        help='Width of the input image, 320(yinda and nyu)')

    parser.add_argument('--testset', dest='imgset', action='store_true',
                        help='Test on set from dataloader, decided by --dataset | True by default')
    parser.add_argument('--no_testset', dest='imgset', action='store_false',
                        help='Test on single image | True by default')
    parser.set_defaults(imgset=True)
    parser.add_argument('--testset_out_path', nargs='?', type=str, default='./result/mt_clean_small',
                        help='Path of the output normal')

    parser.add_argument('--img_path', nargs='?', type=str, default='../Depth2Normal/Dataset/normal/',
                        help='Path of the input image')
    parser.add_argument('--depth_path', nargs='?', type=str, default='../Depth2Normal/Dataset/normal/',
                        help='Path of the input image, mt_data_clean!!!!!!!!!')
    parser.add_argument('--ir_path', nargs='?', type=str, default='../Depth2Normal/Dataset/ir_mask/',
                        help='Path of the input image, mt_data_clean!!!!!!!!!')
    parser.add_argument('--out_path', nargs='?', type=str, default='../Depth2Normal/Dataset/normal/',
                        help='Path of the output normal')
    parser.add_argument('--d_scale', nargs='?', type=int, default=40000,
                        help='Depth scale for depth input. Set the scale to make the 1 in scaled depth equal to 10m.\
                         Only valid testing using image folder')

    parser.add_argument('--img_norm', dest='img_norm', action='store_true',
                        help='Enable input image scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest='img_norm', action='store_false',
                        help='Disable input image scales normalization [0, 1] | True by default')
    parser.set_defaults(img_norm=True)

    parser.add_argument('--img_rotate', dest='img_rot', action='store_true',
                        help='Enable input image transpose | False by default')
    parser.add_argument('--no-img_rotate', dest='img_rot', action='store_false',
                        help='Disable input image transpose | False by default')

    parser.set_defaults(img_rot=False)

    args = parser.parse_args()
    test(args)
