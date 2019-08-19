import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
from loader_utils import png_reader_32bit, png_reader_uint8

from tqdm import tqdm
from torch.utils import data

def get_data_path(name):
    """Extract path to data from config file.

    Args:
        name (str): The name of the dataset.

    Returns:
        (str): The path to the root directory containing the dataset.
    """
    js = open('config.json').read()
    data = json.loads(js)
    return os.path.expanduser(data[name]['data_path'])

class mtLoader(data.Dataset):
    """Data loader for the MatterPort3D dataset.

    """
    def __init__(self, root, split='train', img_size=(256,320), img_norm=True):
        self.root = os.path.expanduser(root)
        self.split = split
        self.img_norm = img_norm
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) \
                                               else (img_size, img_size)


        # for split in ['train', 'test', 'testsmall','small_100']:
        for split in ['train', 'test', 'testsmall']:
            path = pjoin('./datalist', 'mp_' + split + '_list.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name_base = self.files[self.split][index]
        im_path = pjoin(self.root,  im_name_base)

        im_name = im_name_base.replace('_i', '_d')
        im_name = im_name.replace('undistorted_color_dmages', 'undistorted_depth_images')        
        im_name = im_name.replace('.jpg', '.png')
        depth_path = pjoin(self.root, im_name)

        im_name = im_name_base.replace('_i', '_d')
        im_name = im_name.replace('undistorted_color_dmages', 'render_normal')            
        lb_path_nx = pjoin(self.root,  im_name.replace('.jpg', '_mesh_nx.png'))
        lb_path_ny = pjoin(self.root,  im_name.replace('.jpg', '_mesh_ny.png')) 
        lb_path_nz = pjoin(self.root,  im_name.replace('.jpg', '_mesh_nz.png')) 

        im_name = im_name_base.replace('_i', '_d')
        im_name = im_name.replace('undistorted_color_dmages', 'render_depth')   
        meshdepth_path = pjoin(self.root,  im_name.replace('.jpg', '_mesh_depth.png'))             

        im = png_reader_uint8(im_path, self.img_size)#uint8
        rawdepth = png_reader_32bit(depth_path, self.img_size)   #32bit uint
        lbx = png_reader_32bit(lb_path_nx, self.img_size)
        lby = png_reader_32bit(lb_path_ny, self.img_size)
        lbz = png_reader_32bit(lb_path_nz, self.img_size)
        meshdepth = png_reader_32bit(meshdepth_path, self.img_size)  

        im = im.astype(float) 
        rawdepth = rawdepth.astype(float)     
        lbx = lbx.astype(float)
        lby = lby.astype(float)
        lbz = lbz.astype(float)  
        meshdepth = meshdepth.astype(float)      

        if self.img_norm:
            # Resize scales images from -0.5 ~ 0.5
            im = (im-128) / 255
            # Resize scales labels from -1 ~ 1
            lbx = lbx/65535
            lby = lby/65535
            lbz = lbz/65535
            # Resize scales masks from 0 ~ 1
            mask = np.power(lbx,2) + np.power(lby,2) + np.power(lbz,2)
            mask = (mask>0.001).astype(float)
            #file holes            
            lbx[mask == 0] = 0.5
            lby[mask == 0] = 0.5
            lbz[mask == 0] = 0.5
            lb = np.concatenate((lbx[:,:,np.newaxis], 1-lbz[:,:,np.newaxis], lby[:,:,np.newaxis]), axis = 2)
            lb = 2*lb-1 
            # Resize scales valid, devide by mean value           
            rawdepth = rawdepth/40000
            meshdepth = meshdepth/40000
            # Get valid from rawdepth
            valid = (rawdepth>0.0001).astype(float)

        # NHWC -> NCHW
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float()

        lb = torch.from_numpy(lb).float()
        mask = torch.from_numpy(mask).float()
        valid = torch.from_numpy(valid).float()

        rawdepth = rawdepth[np.newaxis,:,:]
        rawdepth = torch.from_numpy(rawdepth).float()

        meshdepth = meshdepth[np.newaxis,:,:]
        meshdepth = torch.from_numpy(meshdepth).float()

        # input: im, 3*h*w
        # gt: lb, h*w*3
        # mask: gt!=0,h*w
        # valid: rawdepth!=0, h*w
        # rawdepth: depth with hole, 1*h*w
        # meshdepth: depth with hole, 1*h*w
        return im, lb, mask, valid, rawdepth, meshdepth

# Leave code for debugging purposes
if __name__ == '__main__':
    # Config your local data path
    local_path = ''
    bs = 5 
    dst = mtLoader(root=local_path)
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        imgs, labels, masks, valids, depths, meshdepths = data

        imgs = imgs.numpy()
        imgs = np.transpose(imgs, [0,2,3,1])
        imgs = imgs+0.5

        labels = labels.numpy()
        labels = 0.5*(labels+1)

        masks = masks.numpy()
        masks = np.repeat(masks[:, :, :, np.newaxis], 3, axis = 3)

        valids = valids.numpy()
        valids = np.repeat(valids[:, :, :, np.newaxis], 3, axis = 3)

        depths = depths.numpy()
        depths = np.transpose(depths, [0,2,3,1])
        depths = np.repeat(depths, 3, axis = 3)

        meshdepths = meshdepths.numpy()
        meshdepths = np.transpose(meshdepths, [0,2,3,1])
        meshdepths = np.repeat(meshdepths, 3, axis = 3)

        f, axarr = plt.subplots(bs,6)
        for j in range(bs):
            # print(im_name[j])
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(labels[j])
            axarr[j][2].imshow(masks[j])
            axarr[j][3].imshow(valids[j])
            axarr[j][4].imshow(depths[j])            
            axarr[j][5].imshow(meshdepths[j])
            
        plt.show()
        a = raw_input()
        if a == 'ex':
            break
        else:
            plt.close()

