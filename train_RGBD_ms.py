##########################################
# Train with RGB+D
# Note: 1. feature fusion at multiscale levels, different loss function
#       2. confidence map refined
#       3. Joint model under arch_F structure
#       4. load pretrain model from RGB_IN_l1 for encoder
# Last update on Oct.23, 2018, Jin Zeng
##########################################

import sys, os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import sklearn.preprocessing as sk
from os.path import join as pjoin
from tensorboardX import SummaryWriter

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from models import get_model, get_lossfun
from loader import get_data_path, get_loader
from pre_trained import get_premodel
from models.loss import cross_cosine
from utils import norm_tf, get_fconv_premodel


def train(args):
    writer = SummaryWriter(comment=args.writer)

    # data loader setting, train and evaluatfconvion
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, split='train', img_size=(args.img_rows, args.img_cols), img_norm=args.img_norm)
    v_loader = data_loader(data_path, split='test', img_size=(args.img_rows, args.img_cols), img_norm=args.img_norm)

    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    evalloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=args.num_workers)
    print("Finish Loader Setup")

    # Setup Model and load pretrained model
    model_name_F = args.arch_F
    model_F = get_model(model_name_F, True)  # concat and output
    model_F = torch.nn.DataParallel(model_F, device_ids=range(torch.cuda.device_count()))

    if args.resume:  # use previously trained model
        # Two types of resume training
        checkpoint = torch.load(args.resume_model_path)
        if checkpoint.has_key('model_RGB_state'):
            # Resume with a RGB model.
            # Load the RGB model and change it as encoder in the fusion network
            # Only the first three conv block will be used.
            state = get_fconv_premodel(model_F, checkpoint['model_RGB_state'])
        else:
            # Resume a RGBD model
            state = checkpoint["model_F_state"]
        model_F.load_state_dict(state)

    model_F.cuda()
    print("Finish model setup")

    # Setup model for confidence map
    if args.arch_map == 'map_conv':
        model_name_map = args.arch_map
        model_map = get_model(model_name_map, True)  # concat and output
        model_map = torch.nn.DataParallel(model_map, device_ids=range(torch.cuda.device_count()))
        if args.resume and checkpoint.has_key("model_map_state"):
            model_map.load_state_dict(checkpoint["model_map_state"])
        model_map.cuda()
        print("Finish model_map setup")

    # optimizers and lr-decay setting
    if args.resume:
        optimizer_F = torch.optim.RMSprop(model_F.parameters(), lr=0.1 * args.l_rate)
        scheduler_F = torch.optim.lr_scheduler.MultiStepLR(optimizer_F, milestones=[2, 4, 6, 9, 12], gamma=0.5)
        if args.arch_map == 'map_conv':
            optimizer_map = torch.optim.RMSprop(model_map.parameters(), lr=0.1 * args.l_rate)
            # scheduler_map = torch.optim.lr_scheduler.MultiStepLR(optimizer_map, milestones=[1, 3, 5, 7, 9, 11, 13], gamma=0.5)#second trial
            scheduler_map = torch.optim.lr_scheduler.MultiStepLR(optimizer_map, milestones=[2, 4, 6, 9, 12], gamma=0.5)
    else:
        optimizer_F = torch.optim.RMSprop(model_F.parameters(), lr=args.l_rate)
        scheduler_F = torch.optim.lr_scheduler.MultiStepLR(optimizer_F, milestones=[1, 3, 5, 8, 11, 15], gamma=0.5)

    best_loss = 1
    n_iter_t, n_iter_v = 0, 0
    if args.dataset == 'matterport':
        total_iter_t = 105432 / args.batch_size
    elif args.dataset == 'scannet':
        total_iter_t = 59743 / args.batch_size
    else:
        total_iter_t = 0
    if not os.path.exists(args.model_savepath):
        os.makedirs(args.model_savepath)
    # forward and backward
    for epoch in range(args.n_epoch):

        scheduler_F.step()
        model_F.train()
        if args.arch_map == 'map_conv':
            scheduler_map.step()
            model_map.train()
        for i, (images, labels, masks, valids, depthes, meshdepthes) in enumerate(trainloader):
            n_iter_t += 1

            images = Variable(images.contiguous().cuda())
            labels = Variable(labels.contiguous().cuda())
            masks = Variable(masks.contiguous().cuda())
            valids = Variable(valids.contiguous().cuda())
            depthes = Variable(depthes.contiguous().cuda())

            optimizer_F.zero_grad()
            if args.arch_map == 'map_conv':
                optimizer_map.zero_grad()

            if args.arch_map == 'map_conv':
                outputs_valid = model_map(torch.cat((depthes, valids[:, np.newaxis, :, :]), dim=1))  # 1c output
                outputs, outputs1, outputs2, outputs3, output_d = model_F(images, depthes, outputs_valid.squeeze(1))
            else:
                outputs, outputs1, outputs2, outputs3, output_d = model_F(images, depthes, valids)
            # outputs, outputs1, outputs2, outputs3, output_d = model_F(images, depthes, valids)
            loss, df = get_lossfun(args.loss, outputs, labels, masks)
            if args.hybrid_loss:
                loss1, df1 = get_lossfun(args.loss, outputs1, labels, masks)
                loss2, df2 = get_lossfun('cosine', outputs2, labels, masks)
                loss3, df3 = get_lossfun('cosine', outputs3, labels, masks)

            outputs.backward(gradient=df, retain_graph=True)
            if args.hybrid_loss:
                outputs1.backward(gradient=0.5 * df1, retain_graph=True)
                outputs2.backward(gradient=0.25 * df2, retain_graph=True)
                outputs3.backward(gradient=0.125 * df3)

            optimizer_F.step()
            if args.arch_map == 'map_conv':
                optimizer_map.step()

            if (i + 1) % 100 == 0:
                if args.hybrid_loss:
                    print("Epoch [%d/%d] Iter [%d/%d] Loss: %.4f,%.4f,%.4f,%.4f " % (
                        epoch + 1, args.n_epoch, i, total_iter_t, loss.data, loss1.data, loss2.data, loss3.data))
                else:
                    print("Epoch [%d/%d] Iter [%d/%d] Loss: %.4f" % (
                        epoch + 1, args.n_epoch, i, total_iter_t, loss.data))

            if (i + 1) % 250 == 0:
                writer.add_scalar('loss/trainloss', loss.data.item(), n_iter_t)
                writer.add_images('Image', images + 0.5, n_iter_t)
                writer.add_images('Label', 0.5 * (labels.permute(0, 3, 1, 2) + 1), n_iter_t)
                writer.add_images('Depth',
                                  np.repeat((depthes - torch.min(depthes)) / (torch.max(depthes) - torch.min(depthes)),
                                            3, axis=1), n_iter_t)
                # writer.add_image('Mesh_Depth', np.repeat((meshdepthes-torch.min(depthes))/(torch.max(depthes)-torch.min(depthes)), 3, axis = 1), n_iter_t)
                outputs_n = norm_tf(outputs)
                if (args.hybrid_loss):
                    outputs_n1 = norm_tf(outputs1)
                    outputs_n2 = norm_tf(outputs2)
                    outputs_n3 = norm_tf(outputs3)
                output_nd = norm_tf(output_d)
                writer.add_images('Output', outputs_n, n_iter_t)
                if (args.hybrid_loss):
                    writer.add_images('Output1', outputs_n1, n_iter_t)
                    writer.add_images('Output2', outputs_n2, n_iter_t)
                    writer.add_images('Output3', outputs_n3, n_iter_t)
                writer.add_images('Output_depth', output_nd, n_iter_t)
                if args.arch_map == 'map_conv':
                    outputs_valid_1 = (outputs_valid - torch.min(outputs_valid)) / (
                            torch.max(outputs_valid) - torch.min(outputs_valid))
                    writer.add_images('Output_Mask', outputs_valid_1.repeat([1, 3, 1, 1]), n_iter_t)

        model_F.eval()
        if args.arch_map == 'map_conv':
            model_map.eval()
        mean_loss, sum_loss = 0, 0
        evalcount = 0
        with torch.no_grad():
            for i_val, (images_val, labels_val, masks_val, valids_val, depthes_val, meshdepthes_val) in tqdm(
                    enumerate(evalloader)):
                n_iter_v += 1
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

                # outputs, outputs1, outputs2, outputs3,output_d = model_F(images_val, depthes_val, valids_val)
                loss, df = get_lossfun(args.loss, outputs, labels_val, masks_val, False)  # valid_val not used infact
                if ((np.isnan(loss)) | (np.isinf(loss))):
                    sum_loss += 0
                else:
                    sum_loss += loss
                    evalcount += 1

                if (i_val + 1) % 250 == 0:
                    # print("Epoch [%d/%d] Evaluation Loss: %.4f" % (epoch+1, args.n_epoch, loss))
                    writer.add_scalar('loss/evalloss', loss, n_iter_v)
                    writer.add_images('Eval Image', images_val + 0.5, n_iter_t)
                    writer.add_images('Eval Label', 0.5 * (labels_val.permute(0, 3, 1, 2) + 1), n_iter_t)
                    writer.add_images('Eval Depth', np.repeat(
                        (depthes_val - torch.min(depthes_val)) / (torch.max(depthes_val) - torch.min(depthes_val)), 3,
                        axis=1), n_iter_t)
                    # writer.add_image('Eval Mesh Depth', np.repeat((meshdepthes_val-torch.min(depthes_val))/(torch.max(meshdepthes_val)-torch.min(depthes_val)), 3, axis = 1), n_iter_t)
                    outputs_n = norm_tf(outputs)
                    output_nd = norm_tf(output_d)
                    writer.add_images('Eval Output', outputs_n, n_iter_t)
                    writer.add_images('Eval Output_depth', output_nd, n_iter_t)
                    if args.arch_map == 'map_conv':
                        outputs_valid_1 = (outputs_valid - torch.min(outputs_valid)) / (
                                torch.max(outputs_valid) - torch.min(outputs_valid))
                        writer.add_images('Eval Output_Mask', outputs_valid_1.repeat([1, 3, 1, 1]), n_iter_t)

            # mean_loss = sum_loss/(evalloader.__len__())
            mean_loss = sum_loss / evalcount
            print("Epoch [%d/%d] Evaluation Mean Loss: %.4f" % (epoch + 1, args.n_epoch, mean_loss))
            writer.add_scalar('loss/evalloss_mean', mean_loss, epoch)

        if mean_loss < best_loss:  # if (epoch+1)%20 == 0:
            best_loss = mean_loss
            if args.hybrid_loss:
                other_info = 'hybrid'
            else:
                other_info = 'nohybrid'
            if args.resume:
                other_info += '_resume'
            if args.arch_map == 'map_conv':
                state = {'epoch': epoch + 1,
                         'model_F_state': model_F.state_dict(),
                         'optimizer_F_state': optimizer_F.state_dict(),
                         'model_map_state': model_map.state_dict(),
                         'optimizer_map_state': optimizer_map.state_dict(), }
                torch.save(state, pjoin(args.model_savepath,
                                        "{}_{}_{}_{}_{}_best.pkl".format(args.arch_F, args.dataset, args.loss,
                                                                         args.model_num, other_info)))
                # For retrain purpose
                # torch.save(state, pjoin(args.model_savepath,
                #                         "{}_{}_{}_{}_{}_{}_best.pkl".format(args.arch_F, args.dataset, args.loss,
                #                                                             'e' + str(epoch), args.model_num,
                #                                                             other_info)))
            else:
                state = {'epoch': epoch + 1,
                         'model_F_state': model_F.state_dict(),
                         'optimizer_F_state': optimizer_F.state_dict(), }
                torch.save(state, pjoin(args.model_savepath,
                                        "{}_{}_{}_{}_{}_best.pkl".format(args.arch_F, args.dataset, args.loss,
                                                                         args.model_num, other_info)))

        # state = {'epoch': epoch+1,
        #             'model_F_state': model_F.state_dict(),
        #             'optimizer_F_state' : optimizer_F.state_dict(),}
        # torch.save(state, pjoin(args.model_savepath, "{}_{}_{}_{}_best.pkl".format(args.arch_F, args.dataset, args.loss, args.model_num)))

    print('Finish training for dataset %s trial %s' % (args.dataset, args.model_num))
    writer.export_scalars_to_json("./{}_{}_{}_{}.json".format(args.arch_F, args.dataset, args.loss, args.model_num))
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch_RGB', nargs='?', type=str, default='vgg_16_in',
                        help='Architecture for RGB to use [\'vgg_16, vgg_16_in etc\']')
    parser.add_argument('--arch_D', nargs='?', type=str, default='unet_3_mask_in',
                        help='Architecture for Depth to use [\'unet_3, unet_3_mask, unet_3_mask_in etc\']')
    parser.add_argument('--arch_F', nargs='?', type=str, default='fconv_ms',
                        help='Architecture for Fusion to use [\'fconv, fconv_in, fconv_ms etc\']')
    parser.add_argument('--arch_map', nargs='?', type=str, default='map_conv',
                        help='Architecture for confidence map to use [\'mask, map_conv etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='matterport',
                        help='Dataset to use [\'nyuv2, matterport, scannet, etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256,
                        help='Height of the input image, 256(yinda), 240(nyu)')
    parser.add_argument('--img_cols', nargs='?', type=int, default=320,
                        help='Width of the input image, 320(yinda and nyu)')

    parser.add_argument('--img_norm', dest='img_norm', action='store_true',
                        help='Enable input image scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest='img_norm', action='store_false',
                        help='Disable input image scales normalization [0, 1] | True by default')
    parser.set_defaults(img_norm=True)

    parser.add_argument('--n_epoch', nargs='?', type=int, default=10,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3,
                        help='Learning Rate')

    parser.add_argument('--tfboard', dest='tfboard', action='store_true',
                        help='Enable visualization(s) on tfboard | False by default')
    parser.add_argument('--no-tfboard', dest='tfboard', action='store_false',
                        help='Disable visualization(s) on tfboard | False by default')
    parser.set_defaults(tfboard=False)

    parser.add_argument('--state_name', nargs='?', type=str, default='vgg_16_mp',
                        help='Path to the saved state dict, vgg_16, vgg_16_mp, vgg_16_mp_in')

    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='previous saved model to restart from |False by default')
    parser.add_argument('--noresume', dest='resume', action='store_false',
                        help='donot use previous saved model to restart from | False by default')

    parser.add_argument('--resume_model_path', nargs='?', type=str, default='',
                        help='model path for the resume model')

    parser.set_defaults(resume=False)

    parser.add_argument('--model_savepath', nargs='?', type=str, default='./checkpoint/FCONV_MS',
                        help='Path for model saving [\'checkpoint etc\']')
    parser.add_argument('--model_num', nargs='?', type=str, default='1',
                        help='Checkpoint index [\'1,2,3, etc\']')

    parser.add_argument('--loss', nargs='?', type=str, default='l1',
                        help='Loss type: cosine, sine, l1')
    parser.add_argument('--hybrid_loss', dest='hybrid_loss', action='store_true',
                        help='Whether use hybrid loss| False by default')
    parser.add_argument('--no_hybrid_loss', dest='hybrid_loss', action='store_false',
                        help='Whether use hybrid loss| False by default')

    parser.set_defaults(hybrid_loss=False)

    parser.add_argument('--writer', nargs='?', type=str, default='fms',
                        help='writer comment: fms')
    parser.add_argument('--num_workers', nargs='?', type=int, default=4, help='Number of workers for data loading')
    args = parser.parse_args()
    train(args)
