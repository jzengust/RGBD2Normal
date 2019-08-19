##################################
# Only used in local machine
# Jin Zeng, 20181026
##################################

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


class scLoader(data.Dataset):
    """Data loader for the scannet dataset.

    """

    def __init__(self, root, split, img_size=(240, 320), img_norm=True, mode=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.img_norm = img_norm
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) \
            else (img_size, img_size)
        self.mode = mode

        for split in ['train', 'test', 'testsmall']:
            path = pjoin('./datalist', 'sc_' + split + '_list.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name_base = self.files[self.split][index]

        # raw_depth
        raw_depth_path = pjoin(self.root, str(im_name_base))
        raw_depth = png_reader_32bit(raw_depth_path, self.img_size)
        raw_depth = raw_depth.astype(float)
        raw_depth = raw_depth / 10000

        # raw_depth_mask
        raw_depth_mask = (raw_depth > 0.0001).astype(float)
        raw_depth = raw_depth[np.newaxis, :, :]
        raw_depth = torch.from_numpy(raw_depth).float()
        raw_depth_mask = torch.from_numpy(raw_depth_mask).float()

        # segmentation label
        seg_path = raw_depth_path.replace('/depth/', '/label/')
        seg_img = png_reader_32bit(seg_path, self.img_size)
        seg_img = torch.from_numpy(seg_img)

        # image
        rgb_path = raw_depth_path.replace('/depth/', '/color/')
        rgb_path = rgb_path.replace('.png', '.jpg')
        image = png_reader_uint8(rgb_path, self.img_size)
        image = image.astype(float)
        # image    = image / 255
        image = (image - 128) / 255
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()

        index = im_name_base[19:]
        index = index.zfill(13)
        scene_name = im_name_base[:12]

        # render_depth
        render_depth_name = index.replace('.png', '_mesh_depth.png')
        render_depth_path = pjoin(self.root, scene_name, 'render_depth', render_depth_name)
        render_depth = png_reader_32bit(render_depth_path, self.img_size)
        render_depth = render_depth.astype(float)
        render_depth = render_depth / 40000
        render_depth = render_depth[np.newaxis, :, :]
        render_depth = torch.from_numpy(render_depth).float()

        # normal
        normal_x_path = render_depth_path.replace('_depth.png', '_nx.png')
        normal_y_path = render_depth_path.replace('_depth.png', '_ny.png')
        normal_z_path = render_depth_path.replace('_depth.png', '_nz.png')
        normal_x_path = normal_x_path.replace('/render_depth/', '/render_normal/')
        normal_y_path = normal_y_path.replace('/render_depth/', '/render_normal/')
        normal_z_path = normal_z_path.replace('/render_depth/', '/render_normal/')

        normal_x = png_reader_32bit(normal_x_path, self.img_size)
        normal_y = png_reader_32bit(normal_y_path, self.img_size)
        normal_z = png_reader_32bit(normal_z_path, self.img_size)

        normal_x = normal_x.astype(float)
        normal_y = normal_y.astype(float)
        normal_z = normal_z.astype(float)
        normal_x = normal_x / 65535
        normal_y = normal_y / 65535
        normal_z = normal_z / 65535

        # normal mask
        normal_mask = np.power(normal_x, 2) + np.power(normal_y, 2) + np.power(normal_z, 2)
        normal_mask = (normal_mask > 0.001).astype(float)

        normal_x[normal_mask == 0] = 0.5
        normal_y[normal_mask == 0] = 0.5
        normal_z[normal_mask == 0] = 0.5

        normal = np.concatenate(
            (normal_x[:, :, np.newaxis], 1 - normal_z[:, :, np.newaxis], normal_y[:, :, np.newaxis]), axis=2)
        normal = 2 * normal - 1
        normal = torch.from_numpy(normal).float()

        # image          : RGB -0.5-0.5, 3*h*w
        # raw_depth      : /10000, 1*h*w
        # raw_depth_mask : 0 or 1, h*w
        # render_depth   : /40000, 1*h*w
        # normal         : /65535, h*w*3
        # normal_mask    : 0 or 1, h*w
        # seg_img        : uint16, h*w same as normal_mask

        if(self.mode=='seg'):
            # For segmentation mask
            return image, normal, normal_mask, raw_depth_mask, raw_depth, seg_img
        else:
            # Ordinary RGBD2normal
            return image, normal, normal_mask, raw_depth_mask, raw_depth, render_depth




if __name__ == '__main__':
    # Config your local data path
    local_path = ''
    bs = 4
    dst = scLoader(root=local_path, split='testsmall')
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        imgs, labels, masks, valids, depths, meshdepths = data
        # imgs, labels, masks, valids, depths= data

        imgs = imgs.numpy()
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        imgs = imgs + 0.5

        labels = labels.numpy()
        labels = 0.5 * (labels + 1)

        masks = masks.numpy()
        masks = np.repeat(masks[:, :, :, np.newaxis], 3, axis=3)

        valids = valids.numpy()
        valids = np.repeat(valids[:, :, :, np.newaxis], 3, axis=3)

        depths = depths.numpy()
        depths = np.transpose(depths, [0, 2, 3, 1])
        depths = np.repeat(depths, 3, axis=3)

        meshdepths = meshdepths.numpy()
        meshdepths = np.transpose(meshdepths, [0, 2, 3, 1])
        meshdepths = np.repeat(meshdepths, 3, axis=3)

        f, axarr = plt.subplots(bs, 6)
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
