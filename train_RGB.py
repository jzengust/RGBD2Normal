##########################################
# Train with RGB exclusively
# RGBD and Depth exclusively are also supported
# Last update on Oct.18, 2018, Jin Zeng
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
from utils import norm_tf, load_resume_state_dict


# from sync_batchnorm import DataParallelWithCallback  

def train(args):
    writer = SummaryWriter(comment=args.writer)

    # data loader setting, train and evaluation
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, split='train', img_size=(args.img_rows, args.img_cols), img_norm=args.img_norm)
    v_loader = data_loader(data_path, split='test', img_size=(args.img_rows, args.img_cols), img_norm=args.img_norm)

    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    evalloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=args.num_workers)
    print("Finish Loader Setup")

    # Setup Model and load pretrained model
    model_name = args.arch_RGB
    # print(model_name)
    model = get_model(model_name, True)  # vgg_16
    if args.pretrain:  # True by default
        if args.input == 'rgb':  # only for rgb we have pretrain option
            state = get_premodel(model, args.state_name)
            model.load_state_dict(state)
            model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        elif args.input == 'd':  # for d, load from result from...
            print("Load training model: {}_{}_{}_{}_best.pkl".format(args.arch_RGB, args.dataset, args.loss, 1))
            checkpoint = torch.load(pjoin(args.model_savepath_pretrain,
                                          "{}_{}_{}_{}_best.pkl".format(args.arch_RGB, args.dataset, args.loss, 1)))
            # model.load_state_dict(load_resume_state_dict(model, checkpoint['model_D_state']))   
            model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
            model.load_state_dict(checkpoint['model_D_state'])
    else:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # model_RGB = DataParallelWithCallback(model_RGB, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    print("Finish model setup with model %s and state_dict %s" % (args.arch_RGB, args.state_name))

    # optimizers and lr-decay setting
    if args.pretrain:  # True by default
        optimizer_RGB = torch.optim.RMSprop(model.parameters(), lr=0.25 * args.l_rate)
        scheduler_RGB = torch.optim.lr_scheduler.MultiStepLR(optimizer_RGB, milestones=[1, 2, 4, 8], gamma=0.5)
    else:
        optimizer_RGB = torch.optim.RMSprop(model.parameters(), lr=args.l_rate)
        scheduler_RGB = torch.optim.lr_scheduler.MultiStepLR(optimizer_RGB, milestones=[1, 3, 5, 8, 11, 15], gamma=0.5)

    # forward and backward
    best_loss = 3
    n_iter_t, n_iter_v = 0, 0
    if args.dataset == 'matterport':
        total_iter_t = 105432 / args.batch_size
    elif args.dataset == 'scannet':
        total_iter_t = 59743 / args.batch_size
    else:
        total_iter_t = 0

    if not os.path.exists(args.model_savepath):
        os.makedirs(args.model_savepath)

    for epoch in range(args.n_epoch):

        scheduler_RGB.step()
        model.train()

        for i, (images, labels, masks, valids, depthes, meshdepthes) in enumerate(trainloader):
            n_iter_t += 1

            images = Variable(images.contiguous().cuda())
            labels = Variable(labels.contiguous().cuda())
            masks = Variable(masks.contiguous().cuda())

            optimizer_RGB.zero_grad()
            if args.input == 'rgb':
                outputs = model(images)
            else:
                depthes = Variable(depthes.contiguous().cuda())
                if args.input == 'rgbd':
                    rgbd_input = torch.cat((images, depthes), dim=1)
                    outputs = model(rgbd_input)
                elif args.input == 'd':
                    outputs = model(depthes)

            loss, df = get_lossfun(args.loss, outputs, labels, masks)
            if args.l1regular:
                loss_rgl, df_rgl = get_lossfun('l1gra', outputs, labels, masks)
            elif args.gradloss:
                loss_grad, df_grad = get_lossfun('gradmap', outputs, labels, masks)

            if args.l1regular:
                outputs.backward(gradient=df, retain_graph=True)
                outputs.backward(gradient=0.1 * df_rgl)
            elif args.gradloss:
                outputs.backward(gradient=df, retain_graph=True)
                outputs.backward(gradient=0.5 * df_grad)
            else:
                outputs.backward(gradient=df)

            optimizer_RGB.step()

            if (i + 1) % 100 == 0:
                if args.l1regular:
                    print("Epoch [%d/%d] Iter [%d/%d] Loss and RGL: %.4f, %.4f" % (
                        epoch + 1, args.n_epoch, i, total_iter_t, loss.data, loss_rgl.data))
                elif args.gradloss:
                    print("Epoch [%d/%d] Iter [%d/%d] Loss and GradLoss: %.4f, %.4f" % (
                        epoch + 1, args.n_epoch, i, total_iter_t, loss.data, loss_grad.data))
                else:
                    print("Epoch [%d/%d] Iter [%d/%d] Loss: %.4f" % (
                        epoch + 1, args.n_epoch, i, total_iter_t, loss.data))

            if (i + 1) % 250 == 0:
                writer.add_scalar('loss/trainloss', loss.data.item(), n_iter_t)
                if args.l1regular:
                    writer.add_scalar('loss/trainloss_rgl', loss_rgl.data.item(), n_iter_t)
                elif args.gradloss:
                    writer.add_scalar('loss/trainloss_grad', loss_grad.data.item(), n_iter_t)

                writer.add_images('Image', images + 0.5, n_iter_t)

                if args.input != 'rgb':
                    writer.add_images('Depth', np.repeat(
                        (depthes - torch.min(depthes)) / (torch.max(depthes) - torch.min(depthes)), 3, axis=1),
                                      n_iter_t)
                writer.add_images('Label', 0.5 * (labels.permute(0, 3, 1, 2) + 1), n_iter_t)
                outputs_n = norm_tf(outputs)
                writer.add_images('Output', outputs_n, n_iter_t)

        model.eval()
        mean_loss, sum_loss, sum_rgl, sum_grad = 0, 0, 0, 0
        evalcount = 0
        with torch.no_grad():
            for i_val, (images_val, labels_val, masks_val, valids_val, depthes_val, meshdepthes_val) in tqdm(
                    enumerate(evalloader)):
                n_iter_v += 1
                images_val = Variable(images_val.contiguous().cuda())
                labels_val = Variable(labels_val.contiguous().cuda())
                masks_val = Variable(masks_val.contiguous().cuda())

                if args.input == 'rgb':
                    outputs = model(images_val)
                else:
                    depthes_val = Variable(depthes_val.contiguous().cuda())
                    if args.input == 'rgbd':
                        rgbd_input = torch.cat((images_val, depthes_val), dim=1)
                        outputs = model(rgbd_input)
                    elif args.input == 'd':
                        outputs = model(depthes_val)

                loss, df = get_lossfun(args.loss, outputs, labels_val, masks_val, False)  # valid_val not used infact
                if args.l1regular:
                    loss_rgl, df_rgl = get_lossfun('l1gra', outputs, labels_val, masks_val, False)
                elif args.gradloss:
                    loss_grad, df_grad = get_lossfun('gradmap', outputs, labels_val, masks_val, False)

                if ((np.isnan(loss)) | (np.isinf(loss))):
                    sum_loss += 0
                else:
                    sum_loss += loss
                    evalcount += 1
                    if args.l1regular:
                        sum_rgl += loss_rgl
                    elif args.gradloss:
                        sum_grad += loss_grad

                if (i_val + 1) % 250 == 0:
                    # print("Epoch [%d/%d] Evaluation Loss: %.4f" % (epoch+1, args.n_epoch, loss))
                    writer.add_scalar('loss/evalloss', loss, n_iter_v)

                    writer.add_images('Eval Image', images_val + 0.5, n_iter_t)
                    if args.input != 'rgb':
                        writer.add_image('Depth', np.repeat(
                            (depthes_val - torch.min(depthes_val)) / (torch.max(depthes_val) - torch.min(depthes_val)),
                            3, axis=1), n_iter_t)
                    writer.add_images('Eval Label', 0.5 * (labels_val.permute(0, 3, 1, 2) + 1), n_iter_t)
                    outputs_n = norm_tf(outputs)
                    writer.add_images('Eval Output', outputs_n, n_iter_t)

            mean_loss = sum_loss / evalcount
            print("Epoch [%d/%d] Evaluation Mean Loss: %.4f" % (epoch + 1, args.n_epoch, mean_loss))
            writer.add_scalar('loss/evalloss_mean', mean_loss, epoch)
            writer.add_scalar('loss/evalloss_rgl_mean', sum_rgl / evalcount, epoch)
            writer.add_scalar('loss/evalloss_grad_mean', sum_grad / evalcount, epoch)

        if mean_loss < best_loss:  # if (epoch+1)%20 == 0:
            best_loss = mean_loss
            state = {'epoch': epoch + 1,
                     'model_RGB_state': model.state_dict(),
                     'optimizer_RGB_state': optimizer_RGB.state_dict(), }
            if args.pretrain:
                if args.l1regular:
                    torch.save(state, pjoin(args.model_savepath,
                                            "{}_{}_{}_{}_rgls_best.pkl".format(args.arch_RGB, args.dataset, args.loss,
                                                                               args.model_num)))
                elif args.gradloss:
                    torch.save(state, pjoin(args.model_savepath,
                                            "{}_{}_{}_{}_grad_best.pkl".format(args.arch_RGB, args.dataset, args.loss,
                                                                               args.model_num)))
                else:
                    torch.save(state, pjoin(args.model_savepath,
                                            "{}_{}_{}_{}_resume_RGB_best.pkl".format(args.arch_RGB, args.dataset,
                                                                                     args.loss, args.model_num)))
            else:
                torch.save(state, pjoin(args.model_savepath,
                                        "{}_{}_{}_{}_resume_RGB_best.pkl".format(args.arch_RGB, args.dataset, args.loss,
                                                                                 args.model_num)))

    print('Finish training for dataset %s trial %s' % (args.dataset, args.model_num))
    # state = {'epoch': epoch+1,
    #                  'model_RGB_state': model_RGB.state_dict(),
    #                  'optimizer_RGB_state' : optimizer_RGB.state_dict(),}
    # if args.pretrain:
    #     torch.save(state, pjoin(args.model_savepath, "{}_{}_{}_{}_RGB_final.pkl".format(args.arch_RGB, args.dataset, args.loss, args.model_num)))
    # elif args.l1regular:
    #     torch.save(state, pjoin(args.model_savepath, "{}_{}_{}_{}_rgls_final.pkl".format(args.arch_RGB, args.dataset, args.loss, args.model_num)))
    # else:
    #     torch.save(state, pjoin(args.model_savepath, "{}_{}_{}_{}_nopretrain_final.pkl".format(args.arch_RGB, args.dataset, args.loss, args.model_num)))                    

    writer.export_scalars_to_json("./{}_{}_{}_{}.json".format(args.arch_RGB, args.dataset, args.loss, args.model_num))
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch_RGB', nargs='?', type=str, default='vgg_16_in',
                        help='Architecture for RGB to use [\'vgg_16, ms, vgg_16_in, vgg_16_in_rgbd, unet_3_in etc\']')
    parser.add_argument('--arch_D', nargs='?', type=str, default='unet_3',
                        help='Architecture for Depth to use [\'unet_3, etc\']')
    parser.add_argument('--arch_F', nargs='?', type=str, default='fconv',
                        help='Architecture for Fusion to use [\'fconv etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='scannet',
                        help='Dataset to use [\'nyuv2, matterport, scannet, etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=240,
                        help='Height of the input image, 256(yinda), 240(nyu)')
    parser.add_argument('--img_cols', nargs='?', type=int, default=320,
                        help='Width of the input image, 320(yinda and nyu)')

    parser.add_argument('--img_norm', dest='img_norm', action='store_true',
                        help='Enable input image scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest='img_norm', action='store_false',
                        help='Disable input image scales normalization [0, 1] | True by default')
    parser.set_defaults(img_norm=True)

    parser.add_argument('--n_epoch', nargs='?', type=int, default=10,
                        help='# of the epochs, max 20')
    parser.add_argument('--batch_size', nargs='?', type=int, default=2,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3,
                        help='Learning Rate')

    parser.add_argument('--tfboard', dest='tfboard', action='store_true',
                        help='Enable visualization(s) on tfboard | False by default')
    parser.add_argument('--no-tfboard', dest='tfboard', action='store_false',
                        help='Disable visualization(s) on tfboard | False by default')
    parser.set_defaults(tfboard=False)

    parser.add_argument('--state_name', nargs='?', type=str, default='vgg_16',
                        help='Path to the saved state dict, vgg_16, vgg_16_mp, vgg_16_mp_in')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--model_savepath', nargs='?', type=str, default='./checkpoint/instance_norm',
                        help='Path for model saving [\'checkpoint etc\']')
    parser.add_argument('--model_num', nargs='?', type=str, default='1',
                        help='Checkpoint index [\'1,2,3, etc\']')

    parser.add_argument('--loss', nargs='?', type=str, default='l1',
                        help='Loss type: cosine, sine, l1')
    parser.add_argument('--pretrained', dest='pretrain', action='store_true',
                        help='Load state_dict from pretrained model | True by default')
    parser.add_argument('--nopretrained', dest='pretrain', action='store_false',
                        help='DONOT load state_dict from pretrained model | True by default')
    parser.set_defaults(pretrain=False)
    parser.add_argument('--l1regular', dest='l1regular', action='store_true',
                        help='Use l1 norm of gradient as regularization loss | False by default')
    parser.add_argument('--nol1regular', dest='l1regular', action='store_false',
                        help='DONOT Use l1 norm of gradient as regularization loss | False by default')
    parser.set_defaults(l1regular=False)

    parser.add_argument('--gradloss', dest='gradloss', action='store_true',
                        help='Extra gradient supervision | False by default')
    parser.add_argument('--nogradloss', dest='gradloss', action='store_false',
                        help='DONOT Use Extra gradient supervision | False by default')
    parser.set_defaults(gradloss=False)

    parser.add_argument('--input', nargs='?', type=str, default='rgbd',
                        help='input type: rgb, rgbd, d')
    parser.add_argument('--model_savepath_pretrain', nargs='?', type=str, default='./checkpoint',
                        help='Path for loading pretrain model[\'checkpoint/instance_norm etc\']')
    parser.add_argument('--writer', nargs='?', type=str, default='fms',
                        help='writer comment: fms')
    parser.add_argument('--num_workers', nargs='?', type=int, default=1, help='Number of workers for data loading')
    args = parser.parse_args()
    train(args)
