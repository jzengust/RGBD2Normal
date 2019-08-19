############################
# Evaluate estimated normal
# criterion include:
# mean, median, 11.25, 22.5, 30
# Jin Zeng, 20180821
############################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.misc as m

def eval_normal(input, label, mask):
    # bs = 1 for testing
    # input: bs*ch*h*w
    # label: bs*h*w*ch
    # mask: bs*h*w
    bz, ch, h, w = input.size()
    
    # normalization
    input = input.permute(0,2,3,1).contiguous().view(-1,ch)
    input_v = F.normalize(input,p=2)    
    label_v = label.contiguous().view(-1,ch)
    # label_v = F.normalize(label_v,p=2) 
    # input_v[torch.isnan(input_v)] = 0

    mask_t = mask.view(-1,1)
    mask_t = torch.squeeze(mask_t)

    loss = F.cosine_similarity(input_v, label_v)#compute inner product     
    loss[torch.ge(loss,1)] = 1
    loss[torch.le(loss,-1)] = -1  
    loss_angle = (180/np.pi)*torch.acos(loss)
    loss_angle = loss_angle[torch.nonzero(mask_t)] 

    mean = torch.mean(loss_angle)
    median = torch.median(loss_angle)
    val_num = loss_angle.size(0)
    small = torch.sum(torch.lt(loss_angle, 11.25)).to(torch.float)/val_num
    mid = torch.sum(torch.lt(loss_angle, 22.5)).to(torch.float)/val_num
    large = torch.sum(torch.lt(loss_angle, 30)).to(torch.float)/val_num

    outputs_n = 0.5*(input_v+1)                
    outputs_n = outputs_n.view(-1, h, w, ch)# bs*h*w*3

    return outputs_n, mean.data.item(), median.data.item(), small.data.item(), mid.data.item(), large.data.item()

def eval_normal_pixel(input, label, mask):
    # bs = 1 for testing
    # input: bs*ch*h*w
    # label: bs*h*w*ch
    # mask: bs*h*w
    bz, ch, h, w = input.size()
    
    # normalization
    input = input.permute(0,2,3,1).contiguous().view(-1,ch)
    input_v = F.normalize(input,p=2)    
    label_v = label.contiguous().view(-1,ch)
    # label_v = F.normalize(label_v,p=2) 
    # input_v[torch.isnan(input_v)] = 0

    mask_t = mask.view(-1,1)
    mask_t = torch.squeeze(mask_t)

    loss = F.cosine_similarity(input_v, label_v)#compute inner product     
    loss[torch.ge(loss,1)] = 1
    loss[torch.le(loss,-1)] = -1  
    loss_angle = (180/np.pi)*torch.acos(loss)
    loss_angle = loss_angle[torch.nonzero(mask_t)]

    val_num = loss_angle.size(0) 

    if val_num>0:
        mean = torch.mean(loss_angle).data.item()
        median = torch.median(loss_angle).data.item()    
        small = (torch.sum(torch.lt(loss_angle, 11.25)).to(torch.float)/val_num).data.item()
        mid = (torch.sum(torch.lt(loss_angle, 22.5)).to(torch.float)/val_num).data.item()
        large = (torch.sum(torch.lt(loss_angle, 30)).to(torch.float)/val_num).data.item()
    else:
        mean=0
        median=0
        small=0
        mid=0
        large=0

    outputs_n = 0.5*(input_v+1)                
    outputs_n = outputs_n.view(-1, h, w, ch)# bs*h*w*3

    return outputs_n, val_num, mean, median, small, mid, large

def eval_normal_detail(input, label, mask):
    # bs = 1 for testing
    # input: bs*ch*h*w
    # label: bs*h*w*ch
    # mask: bs*h*w
    bz, ch, h, w = input.size()
    
    # normalization
    input = input.permute(0,2,3,1).contiguous().view(-1,ch)
    input_v = F.normalize(input,p=2)    
    label_v = label.contiguous().view(-1,ch)
    # label_v = F.normalize(label_v,p=2) 
    # input_v[torch.isnan(input_v)] = 0

    mask_t = mask.view(-1,1)
    mask_t = torch.squeeze(mask_t)

    loss = F.cosine_similarity(input_v, label_v)#compute inner product     
    loss[torch.ge(loss,1)] = 1
    loss[torch.le(loss,-1)] = -1  
    loss_angle = (180/np.pi)*torch.acos(loss)
    loss_angle = loss_angle[torch.nonzero(mask_t)] 

    return loss_angle

def eval_print(sum_mean, sum_median, sum_small, sum_mid, sum_large, sum_num, item='Pixel-Level'):
    allnum = sum(sum_num)
    if allnum == 0:
        print("Empty in %s pixels" % (item))
    else:
        pixel_mean = np.sum(np.array(sum_mean)*np.array(sum_num))/allnum   
        pixel_median = np.sum(np.array(sum_median)*np.array(sum_num))/allnum    
        pixel_small = np.sum(np.array(sum_small)*np.array(sum_num))/allnum   
        pixel_mid = np.sum(np.array(sum_mid)*np.array(sum_num))/allnum   
        pixel_large = np.sum(np.array(sum_large)*np.array(sum_num))/allnum   

        print("Evaluation %s Mean Loss: mean %.4f, median %.4f, 11.25 %.4f, 22.5 %.4f, 30 %.4f" % (item, 
        pixel_mean, pixel_median, pixel_small, pixel_mid, pixel_large))                         

def eval_mask_resize(segment_val, img_rows, img_cols):
    segment_val = np.squeeze(segment_val.data.cpu().numpy(), axis=0)#uint8 array
    segment_val = m.imresize(segment_val, (img_rows, img_cols))#only works for 8 bit image
    segment_val = segment_val>0
    segment_val = torch.from_numpy(segment_val.astype(np.uint8))
    segment_val = Variable(segment_val.contiguous().cuda())

    return segment_val