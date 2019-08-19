##########################
# Test normal estimation
# Pure RGB input without Depth
# coupled with train_RGB.py
# Jin Zeng, 20180903
#########################

import sys, os
import torch
import argparse
import time
import numpy as np
import scipy.misc as misc
from os.path import join as pjoin

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from models import get_model, get_lossfun
from loader import get_data_path, get_loader
from pre_trained import get_premodel
from utils import norm_imsave, get_dataList, change_channel
from models.eval import eval_normal

# For mat file handling
import scipy.io as sio


def test(args):
    # Setup Model
    model_name = args.arch_RGB
    model = get_model(model_name, True)  # vgg_16
    testset_out_default_path = "./result/"
    testset_out_path = args.result_path
    if args.imgset:
        test_info = args.test_dataset + '_' + args.test_split
    else:
        if args.img_datalist == '':
            # Single image
            test_info = 'single'
        else:
            # Image list
            test_info = 'batch'

    if args.model_full_name != '':
        # Use the full name of model to load
        print("Load training model: " + args.model_full_name)
        checkpoint = torch.load(pjoin(args.model_savepath, args.model_full_name))
        if testset_out_path == '':
            testset_out_path = "{}{}_{}".format(testset_out_default_path, args.model_full_name, test_info)
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model.load_state_dict(checkpoint['model_RGB_state'])
    else:
        # Pretrain model
        print("Load pretrained model: {}".format(args.state_name))
        state = get_premodel(model, args.state_name)
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model.load_state_dict(state)
        if testset_out_path == '':
            testset_out_path = "{}pretrain_{}".format(testset_out_default_path, test_info)

    # Setup image
    # Create output folder if needed
    # if os.path.isdir(testset_out_path) == False:
    #     os.mkdir(testset_out_path)
    if args.imgset:
        print("Test on dataset: {}".format(args.test_dataset))

        # Set up dataloader
        data_loader = get_loader(args.test_dataset)
        data_path = get_data_path(args.test_dataset)
        v_loader = data_loader(data_path, split=args.test_split, img_size=(args.img_rows, args.img_cols),
                               img_norm=args.img_norm)
        evalloader = data.DataLoader(v_loader, batch_size=1)
        print("Finish Loader Setup")

        model.cuda()
        model.eval()
        sum_mean, sum_median, sum_small, sum_mid, sum_large = 0, 0, 0, 0, 0
        evalcount = 0
        if args.numerical_result_path != '':
            f = open(args.numerical_result_path, 'w')

        with torch.no_grad():
            # for i_val, (images_val, labels_val, masks_val, valids_val) in tqdm(enumerate(evalloader)):
            for i_val, (images_val, labels_val, masks_val, valids_val, depthes_val, meshdepthes_val) in tqdm(
                    enumerate(evalloader)):
                images_val = Variable(images_val.contiguous().cuda())
                labels_val = Variable(labels_val.contiguous().cuda())
                masks_val = Variable(masks_val.contiguous().cuda())
                if args.arch_RGB == 'ms':
                    outputs, outputs2, outputs3, outputs4, outputs5 = model(images_val)
                else:
                    outputs = model(images_val)  # 1*ch*h*w

                outputs_n, mean_i, median_i, small_i, mid_i, large_i = eval_normal(outputs, labels_val, masks_val)
                outputs_norm = np.squeeze(outputs_n.data.cpu().numpy(), axis=0)
                labels_val_norm = np.squeeze(labels_val.data.cpu().numpy(), axis=0)
                images_val = np.squeeze(images_val.data.cpu().numpy(), axis=0)
                images_val = images_val + 0.5
                images_val = images_val.transpose(1, 2, 0)

                outputs_norm = change_channel(outputs_norm)
                # outputs_norm= temp_change_mlt_chanel(outputs_norm)

                labels_val_norm = (labels_val_norm + 1) / 2  # scale to 0 1
                # Change channel to have a better appearance for paper.
                labels_val_norm = change_channel(labels_val_norm)
                misc.imsave(pjoin(testset_out_path, "{}_out.png".format(i_val + 1)), outputs_norm)
                misc.imsave(pjoin(testset_out_path, "{}_gt.png".format(i_val + 1)), labels_val_norm)
                misc.imsave(pjoin(testset_out_path, "{}_in.jpg".format(i_val + 1)), images_val)

                if ((np.isnan(mean_i)) | (np.isinf(mean_i))):
                    print('Error!')
                    sum_mean += 0
                else:
                    sum_mean += mean_i
                    sum_median += median_i
                    sum_small += small_i
                    sum_mid += mid_i
                    sum_large += large_i
                    evalcount += 1
                    if (i_val + 1) % 1 == 0:
                        print("Iteration %d Evaluation Loss: mean %.4f, median %.4f, 11.25 %.4f, 22.5 %.4f, 30 %.4f" % (
                            i_val + 1,
                            mean_i, median_i, small_i, mid_i, large_i))
                    if (args.numerical_result_path != ''):
                        f.write(
                            "Iteration %d Evaluation Loss: mean %.4f, median %.4f, 11.25 %.4f, 22.5 %.4f, 30 %.4f\n" % (
                                i_val + 1,
                                mean_i, median_i, small_i, mid_i, large_i))
            sum_mean = sum_mean / evalcount
            sum_median = sum_median / evalcount
            sum_small = sum_small / evalcount
            sum_mid = sum_mid / evalcount
            sum_large = sum_large / evalcount
            if args.numerical_result_path != '':
                f.write(
                    "evalnum is %d, Evaluation Mean Loss: mean %.4f, median %.4f, 11.25 %.4f, 22.5 %.4f, 30 %.4f" % (
                        evalcount, sum_mean, sum_median, sum_small, sum_mid, sum_large))
                f.close()
            print("evalnum is %d, Evaluation Mean Loss: mean %.4f, median %.4f, 11.25 %.4f, 22.5 %.4f, 30 %.4f" % (
                evalcount, sum_mean, sum_median, sum_small, sum_mid, sum_large))
            # end of dataset test
    else:
        if args.img_datalist == "":
            # For single image, without GT
            print("Read Input Image from : {}".format(args.img_path))
            img = misc.imread(args.img_path)
            if args.img_rot:
                img = np.transpose(img, (1, 0, 2))
                img = np.flipud(img)
            orig_size = img.shape[:-1]
            img = misc.imresize(img, (args.img_rows, args.img_cols))  # Need resize the image to model inputsize

            img = img.astype(np.float)
            if args.img_norm:
                img = (img - 128) / 255
            # NHWC -> NCHW
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, 0)
            img = torch.from_numpy(img).float()

            if torch.cuda.is_available():
                model.cuda(0)
                images = Variable(img.contiguous().cuda(0))
            else:
                images = Variable(img)

            with torch.no_grad():
                outputs = model(images)

            outputs_norm = norm_imsave(outputs)
            outputs_norm = np.squeeze(outputs_norm.data.cpu().numpy(), axis=0)

            # Change channels
            outputs_norm = change_channel(outputs_norm)

            misc.imsave(args.out_path, outputs_norm)
            print("Complete")
            # end of test on single image
        else:
            # For image list without GT
            data_list = get_dataList(args.img_datalist)
            for img_path in data_list:
                print("Read Input Image from : {}".format(img_path))
                img = misc.imread(pjoin(args.img_dataroot, img_path))
                height, width, channels = img.shape
                output_filename = img_path.split('/')[-1]
                if args.img_rot:
                    img = np.transpose(img, (1, 0, 2))
                    img = np.flipud(img)
                orig_size = img.shape[:-1]
                img = misc.imresize(img, (args.img_rows, args.img_cols))  # Need resize the image to model inputsize

                img = img.astype(np.float)
                if args.img_norm:
                    img = (img - 128) / 255
                # NHWC -> NCHW
                img = img.transpose(2, 0, 1)
                img = np.expand_dims(img, 0)
                img = torch.from_numpy(img).float()

                if torch.cuda.is_available():
                    model.cuda(0)
                    images = Variable(img.contiguous().cuda(0))
                else:
                    images = Variable(img)

                with torch.no_grad():
                    outputs = model(images)

                outputs_norm = norm_imsave(outputs)
                outputs_norm = np.squeeze(outputs_norm.data.cpu().numpy(), axis=0)

                # Change channels
                outputs_norm = change_channel(outputs_norm)

                # Resize to the original size, if needed
                # outputs_norm = misc.imresize(outputs_norm, (height, width))

                # Save the result
                misc.imsave(pjoin(args.out_path, output_filename), outputs_norm)

                # Save to mat file, if needed
                # outputs_mat = outputs.tolist()
                # mat_filename = output_filename.replace(output_filename.split('.')[-1], 'mat')
                # sio.savemat(pjoin(testset_out_path, mat_filename), {'normal': outputs_mat});
            print("Complete")
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--arch_RGB', nargs='?', type=str, default='vgg_16',
                        help='Architecture for RGB to use [\'vgg_16, ms etc\']')
    parser.add_argument('--pretrained', dest='pretrain', action='store_true',
                        help='Use pretrained model from Yinda, state_name should be vgg_16 or vgg_16_mp | True by default')
    parser.add_argument('--no_pretrained', dest='pretrain', action='store_false',
                        help='Use our own model | True by default')
    parser.set_defaults(pretrain=False)

    parser.add_argument('--state_name', nargs='?', type=str, default='vgg_16_mp',
                        help='Path to the saved state dict [\'vgg_16, vgg_16_mp\']')
    parser.add_argument('--numerical_result_path', nargs='?', type=str, default='',
                        help='the filename path to save the numerical result')
    parser.add_argument('--model_savepath', nargs='?', type=str, default='./checkpoint',
                        help='Path for model saving [\'checkpoint etc\']')

    parser.add_argument('--model_full_name', nargs='?', type=str, default='',
                        help='The full name of the model to be tested.')

    parser.add_argument('--train_dataset', nargs='?', type=str, default='matterport',
                        help='Dataset to use in training [\'nyuv2, matterport, scannet, etc\']. Only for model loading')
    parser.add_argument('--test_dataset', nargs='?', type=str, default='matterport',
                        help='Dataset to use in testing [\'nyuv2, matterport, scannet, etc\']')

    parser.add_argument('--test_split', nargs='?', type=str, default='', help='The split of dataset in testing')

    # parser.add_argument('--train_split', nargs='?', type=str, default='',
    #                     help='The split of dataset in training model.Only for model loading')
    # parser.add_argument('--dataset_size', nargs='?', type=int, default=50,
    #                     help='The percentage of used data in training dataset.')

    parser.add_argument('--loss', nargs='?', type=str, default='cosine',
                        help='Loss type: cosine, l1')
    parser.add_argument('--model_num', nargs='?', type=str, default='1',
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

    parser.add_argument('--result_path', nargs='?', type=str, default='',
                        help='Path of the output normal')

    # For image datalist
    parser.add_argument('--img_datalist', nargs='?', type=str, default='',
                        help='Test datalist of input images.')
    parser.add_argument('--img_dataroot', nargs='?', type=str, default='',
                        help='Data root of input images.')
    # For single image
    parser.add_argument('--img_path', nargs='?', type=str, default='./data/xx_1.jpg',
                        help='Path of the input image')
    parser.add_argument('--out_path', nargs='?', type=str, default='./result/',
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
    parser.set_defaults(img_rot=False)

    args = parser.parse_args()
    test(args)
