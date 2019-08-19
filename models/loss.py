# ################################
# Loss functions
# cosine, l1
# Jin Zeng, 20181003
#################################

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def cross_cosine(input, label, mask, train=True):
    # input: bs*ch*h*w
    # label: bs*h*w*ch
    # mask: bs*h*w
    bz, ch, h, w = input.size()
    
    # normalization
    input = input.permute(0,2,3,1).contiguous().view(-1,ch)
    input_v = F.normalize(input,p=2)
    label_v = label.contiguous().view(-1,ch)
    target = torch.ones([input.size(0),], dtype=torch.float).cuda()
    mask_t = mask.contiguous().view(-1,1)
    mask_t = torch.squeeze(mask_t)
    target[torch.eq(mask_t,0)] = -1
    
    if(train == True): # use mask from surface normal
        loss = F.cosine_embedding_loss(input_v, label_v, target, margin=1)
        df = torch.autograd.grad(loss,input_v,only_inputs=True)
        df = df[0]
        df = torch.autograd.grad(input_v,input,grad_outputs=df,only_inputs=True)
        df = df[0]
        mask = mask.contiguous().view(-1,1).expand_as(df)
        df[torch.eq(mask,0)] = 0
        df = df.view(-1, h, w, ch)
        df = df.permute(0,3,1,2).contiguous()
    else:  # use mask from depth valid
        # mask = mask.view(-1,1).expand_as(input_v)
        # input_v[torch.eq(mask,0)] = 0
        # label_v[torch.eq(mask,0)] = 0
        # loss = F.cosine_embedding_loss(input_v, label_v, target, margin=1, size_average=False)
        # loss = loss/sum(mask_t)
        # loss = loss.data.item()
        input_v[torch.isnan(input_v)] = 0
        loss = F.cosine_similarity(input_v, label_v)#compute inner product 
        loss[torch.ge(loss,1)] = 1
        loss[torch.le(loss,-1)] = -1
        loss = torch.acos(loss)   
        loss[torch.eq(mask_t,0)] = 0 #rm the masked pixels
        loss = (torch.mean(loss))*(mask_t.numel()/(np.sum(mask_t.data.cpu().numpy())))
        loss = loss.data.item()
        df = None        
    
    return loss, df

def l1norm(input, label, mask, train=True):
    # input: bs*ch*h*w
    # label: bs*h*w*ch
    # mask: bs*h*w
    bz, ch, h, w = input.size()
    
    # normalization
    input = input.permute(0,2,3,1).contiguous().view(-1,ch)
    input_v = F.normalize(input,p=2)
    label_v = label.contiguous().view(-1,ch)
    target = torch.ones([input.size(0),], dtype=torch.float).cuda()
    mask_t = mask.contiguous().view(-1,1)
    mask_t = torch.squeeze(mask_t)
    target[torch.eq(mask_t,0)] = -1
    
    if(train == True): # use mask from surface normal
        loss = F.l1_loss(input_v, label_v, reduce=False)#compute inner product
        loss[torch.eq(mask_t,0)] = 0 #rm the masked pixels 
        loss = torch.mean(loss)
        df = torch.autograd.grad(loss,input_v,only_inputs=True)
        df = df[0]
        df = torch.autograd.grad(input_v,input,grad_outputs=df,only_inputs=True)
        df = df[0]
        mask = mask.contiguous().view(-1,1).expand_as(df)
        df[torch.eq(mask,0)] = 0
        df = df.view(-1, h, w, ch)
        df = df.permute(0,3,1,2).contiguous()
    else:  # use mask from depth valid
        # input_v[torch.isnan(input_v)] = 0
        # loss = F.cosine_similarity(input_v, label_v)#compute inner product
        # loss[torch.ge(loss,1)] = 1
        # loss[torch.le(loss,-1)] = -1
        # loss = torch.acos(loss)
        # loss[torch.eq(mask_t,0)] = 0 #rm the masked pixels
        # loss = (torch.mean(loss))*(mask_t.numel()/(np.sum(mask_t.data.cpu().numpy())))
        # loss = loss.data.item()
        # df = None
        loss = F.l1_loss(input_v, label_v, reduce=False)
        loss[torch.eq(mask_t, 0)] = 0
        loss = torch.mean(loss)
        df = None
    
    return loss, df

def l1_normgrad(input, label, mask, train=True):
    # input: bs*ch*h*w
    # label: bs*h*w*ch
    # mask: bs*h*w
    bz, ch, h, w = input.size()
    
    # compute diff
    # input = input.permute(0,2,3,1).contiguous().view(-1,ch)
    # input_v = F.normalize(input,p=2)
    label = label.permute(0,3,1,2).contiguous()
    label_sm = F.upsample(label,size=[h/8,w/8], mode='bilinear')
    label_lg = F.upsample(label_sm,size=[h,w], mode='bilinear')
    label_diff = label - label_lg

    mask_t = mask.unsqueeze(0).repeat(3,1,1,1).permute(1,0,2,3)
    mask_v = mask_t.contiguous().view(-1,1)
    mask_v = torch.squeeze(mask_v)
    
    if(train == True): # use mask from surface normal
        loss = F.l1_loss(input, label_diff, reduce=False)#compute inner product
        loss[torch.eq(mask_t, 0)] = 0 #rm the masked pixels 
        loss = torch.mean(loss)
        df = torch.autograd.grad(50*loss,input,only_inputs=True)
        df = df[0]
        df[torch.eq(mask_t,0)] = 0
    else:  # use mask from depth valid
        loss = F.l1_loss(input, label_diff, reduce=False)#compute inner product
        loss[torch.eq(mask_t, 0)] = 0 #rm the masked pixels 
        loss = (torch.mean(loss))*(mask_v.numel()/(sum(mask_v)))
        loss = loss.data.item()
        df = None    
    
    return loss, df

def energy(input, label, mask, train=True):
    # input: bs*ch*h*w
    # label: bs*h*w*ch
    # mask: bs*h*w
    # bz, ch, h, w = input.size()
    
    if(train == True):
        loss = 1-5*torch.sqrt(torch.mean(torch.pow(input, 2))) # max energy = min (1-energy)   
        threshold = torch.zeros(loss.size()).cuda()
        loss = torch.max(loss,threshold)
        df = torch.autograd.grad(loss,input,only_inputs=True)
        df = df[0]
    else:  # use mask from depth valid
        loss = torch.sqrt(torch.mean(torch.pow(input, 2))) # compute energy, no inverse    
        loss = loss.data.item()
        df = None    
    
    return loss, df

def l1_sm(input, label, mask, train=True):
    # input: bs*ch*h/8*w/8
    # label: bs*h*w*ch
    # mask: bs*h/8*w/8
    bz, ch, h, w = input.size()
    
    # compute downsampled version of label
    label = label.permute(0,3,1,2).contiguous()
    label_sm = F.upsample(label,size=[h,w], mode='bilinear')

    mask_t = mask.unsqueeze(0).repeat(3,1,1,1).permute(1,0,2,3)
    
    if(train == True): # use mask from surface normal
        loss = F.l1_loss(input, label_sm, reduce=False)#compute inner product
        loss[torch.eq(mask_t, 0)] = 0 #rm the masked pixels 
        loss = torch.mean(loss)
        df = torch.autograd.grad(loss,input,only_inputs=True)
        df = df[0]
        df[torch.eq(mask_t,0)] = 0
    else:  # use mask from depth valid
        loss = F.l1_loss(input, label_sm, reduce=False)#compute inner product
        loss[torch.eq(mask_t, 0)] = 0 #rm the masked pixels 
        loss = (torch.mean(loss))*(mask.numel()/(sum(mask)))
        loss = loss.data.item()
        df = None    
    
    return loss, df


def l1granorm(input, label, mask, train=True):
    # input: bs*ch*h*w
    # label: bs*h*w*ch
    # mask: bs*h*w
    bz, ch, h, w = input.size()
    
    # normalization
    input = input.permute(0,2,3,1).contiguous().view(-1,ch)
    input_v = F.normalize(input,p=2)
    input_v = input_v.view(-1, h, w, ch)
    input_v = input_v.permute(0,3,1,2)

    input_v_r = input_v.narrow(1,0,1)
    input_v_g = input_v.narrow(1,1,1)
    input_v_b = input_v.narrow(1,2,1)

    a=np.array([[0,0,0],[0,1,-1],[0,0,0]])
    conv1=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight=nn.Parameter(torch.from_numpy(a).float().cuda().unsqueeze(0).unsqueeze(0))    
    b=np.array([[0, 0, 0],[0,1,0],[0,-1,0]])
    conv2=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight=nn.Parameter(torch.from_numpy(b).float().cuda().unsqueeze(0).unsqueeze(0))

    G_r_x=conv1(input_v_r) 
    G_r_y=conv2(input_v_r)
    G_g_x=conv1(input_v_g)     
    G_g_y=conv2(input_v_g)
    G_b_x=conv1(input_v_b)     
    G_b_y=conv2(input_v_b)
    eps = 0.0000001
    
    if(train == True): 
        loss_r = torch.mean(torch.pow(torch.abs(G_r_x)+eps, 0.5))+torch.mean(torch.pow(torch.abs(G_r_y)+eps, 0.5))
        loss_g = torch.mean(torch.pow(torch.abs(G_g_x)+eps, 0.5))+torch.mean(torch.pow(torch.abs(G_g_y)+eps, 0.5))
        loss_b = torch.mean(torch.pow(torch.abs(G_b_x)+eps, 0.5))+torch.mean(torch.pow(torch.abs(G_b_y)+eps, 0.5))
        loss = (loss_r+loss_g+loss_b)/(2*3)
        df = torch.autograd.grad(loss,input,only_inputs=True)       
        df = df[0]
        df = df.view(-1, h, w, ch)
        df = df.permute(0,3,1,2).contiguous()
    else:  
        loss_r = torch.mean(torch.pow(torch.abs(G_r_x)+eps, 0.5))+torch.mean(torch.pow(torch.abs(G_r_y)+eps, 0.5))
        loss_g = torch.mean(torch.pow(torch.abs(G_g_x)+eps, 0.5))+torch.mean(torch.pow(torch.abs(G_g_y)+eps, 0.5))
        loss_b = torch.mean(torch.pow(torch.abs(G_b_x)+eps, 0.5))+torch.mean(torch.pow(torch.abs(G_b_y)+eps, 0.5))
        loss = (loss_r+loss_g+loss_b)/(2*3)
        loss = loss.data.item()
        df = None        
    
    return loss, df

def gradmap(input, label, mask, train=True):
    # supervise 1st order gradient
    # input: bs*ch*h*w
    # label: bs*h*w*ch
    # mask: bs*h*w
    bz, ch, h, w = input.size()
    fblur=np.array([[1,1,1],[1,1,1],[1,1,1]])
    conv0=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv0.weight=nn.Parameter(torch.from_numpy(fblur).float().cuda().unsqueeze(0).unsqueeze(0))  
    
    mask_t = mask.contiguous().view(-1,1)
    mask_t = torch.squeeze(mask_t)
    mask_b = mask.unsqueeze(0).permute(1,0,2,3)# add one channel
    mask_b = conv0(mask_b) #dilate
    mask_b = conv0(mask_b) 
    mask_binary = torch.eq(mask_b,0)
    
    # normalization
    input_v = input.permute(0,2,3,1).contiguous().view(-1,ch)
    input_v = F.normalize(input_v,p=2)
    input_v = input_v.view(-1, h, w, ch)
    input_v = input_v.permute(0,3,1,2)

    input_v_r = input_v.narrow(1,0,1)
    input_v_g = input_v.narrow(1,1,1)
    input_v_b = input_v.narrow(1,2,1)

    a=np.array([[0,0,0],[0,1,-1],[0,0,0]])
    conv1=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight=nn.Parameter(torch.from_numpy(a).float().cuda().unsqueeze(0).unsqueeze(0))    
    b=np.array([[0, 0, 0],[0,1,0],[0,-1,0]])
    conv2=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight=nn.Parameter(torch.from_numpy(b).float().cuda().unsqueeze(0).unsqueeze(0))

    G_input = torch.zeros([bz,6,h,w], dtype=torch.float).cuda()
    G_input[:,0:1,:,:] = conv1(input_v_r) 
    G_input[:,1:2,:,:] = conv2(input_v_r)
    G_input[:,2:3,:,:] = conv1(input_v_g)     
    G_input[:,3:4,:,:] = conv2(input_v_g)
    G_input[:,4:5,:,:] = conv1(input_v_b)     
    G_input[:,5:6,:,:] = conv2(input_v_b)

    label_v = label.permute(0,3,1,2)
    label_v_r = label_v.narrow(1,0,1)
    label_v_g = label_v.narrow(1,1,1)
    label_v_b = label_v.narrow(1,2,1)
    G_label = torch.zeros([bz,6,h,w], dtype=torch.float).cuda()
    G_label[:,0:1,:,:]=conv1(label_v_r) 
    G_label[:,1:2,:,:]=conv2(label_v_r)
    G_label[:,2:3,:,:]=conv1(label_v_g)     
    G_label[:,3:4,:,:]=conv2(label_v_g)
    G_label[:,4:5,:,:]=conv1(label_v_b)     
    G_label[:,5:6,:,:]=conv2(label_v_b)
    
    if(train == True): 
        loss = F.l1_loss(G_input, G_label, reduce=False)#compute inner product
        loss = torch.sum(loss,dim=1,keepdim=True)
        loss[mask_binary] = 0 #rm the masked pixels 
        loss = torch.mean(loss)
        
        df = torch.autograd.grad(loss,input,only_inputs=True)   
        df = df[0]
        df[mask_binary.repeat(1,3,1,1)] = 0    
    else:  
        loss = F.l1_loss(G_input, G_label, reduce=False)#compute inner product
        loss = torch.sum(loss,dim=1,keepdim=True)
        loss[mask_binary] = 0 #rm the masked pixels 
        loss = torch.mean(loss)
        loss = loss.data.item()
        df = None        
    
    return loss, df

def sin_cosine(input, label, mask, train=True):
    # input: bs*ch*h*w
    # label: bs*h*w*ch
    # mask: bs*h*w
    bz, ch, h, w = input.size()
    
    # normalization
    input = input.permute(0,2,3,1).contiguous().view(-1,ch)
    input_v = F.normalize(input,p=2)
    label_v = label.contiguous().view(-1,ch)
    mask_t = mask.contiguous().view(-1,1)
    mask_t = torch.squeeze(mask_t)
    ones_v = torch.ones([input.size(0),], dtype=torch.float).cuda(0)
    target = torch.zeros([input.size(0),], dtype=torch.float).cuda(0)
    target[torch.eq(mask_t,0)] = 1
    comp_zeros = torch.zeros([input.size(0),], dtype=torch.float).cuda(0)
    
    if(train == True): # use mask from surface normal
        loss = F.cosine_similarity(input_v, label_v)#compute inner product
        loss = ones_v - torch.pow(loss,2)
        loss = torch.sqrt(loss)
        loss = torch.max(loss-target, comp_zeros)
        # loss[torch.eq(mask_t,0)] = 0 #rm the masked pixels         
        loss = torch.mean(loss)
        df = torch.autograd.grad(loss,input_v,only_inputs=True)
        df = df[0]
        df = torch.autograd.grad(input_v,input,grad_outputs=df,only_inputs=True)
        df = df[0]
        mask = mask.contiguous().view(-1,1).expand_as(df)
        df[torch.eq(mask,0)] = 0
        df = df.view(-1, h, w, ch)
        df = df.permute(0,3,1,2).contiguous()
    else:  # use mask from depth valid
        loss = F.cosine_similarity(input_v, label_v)#compute inner product 
        loss = torch.acos(loss)   
        loss[torch.eq(mask_t,0)] = 0 #rm the masked pixels
        loss = sum(loss)/sum(mask_t)
        loss = loss.data.item()
        df = None       
    
    return loss, df
