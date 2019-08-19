# Implemented in Pytorch by Jin Zeng, 20181008

import torch.nn as nn
import torch
from models_utils import *

class vgg_16_in(nn.Module):
    
    def __init__(self, input_channel, output_channel, track_running_static=True):
        super(vgg_16_in, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.track = track_running_static
        filters = [64, 128, 256, 512, 512]

        # encoder
        self.conv1 = create_conv_2_in(self.input_channel, filters[0],track=self.track)
        self.conv2 = create_conv_2_in(filters[0], filters[1],track=self.track)
        self.conv3 = create_conv_3_in(filters[1], filters[2],track=self.track)
        self.conv4 = create_conv_3_in(filters[2], filters[3],track=self.track)
        self.conv5 = create_conv_3_in(filters[3], filters[4],track=self.track)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # decoder
        self.deconv5 = create_deconv_3_in(filters[4], filters[3],track=self.track)
        self.deconv4 = create_deconv_3_in(filters[3]+filters[3], filters[2],track=self.track)
        self.deconv3 = create_deconv_3_in(filters[2]+filters[2], filters[1],track=self.track)
        self.deconv2 = create_deconv_2_in(filters[1]+filters[1], filters[0],track=self.track)
        self.deconv1 = create_addon(filters[0]+filters[0], filters[0], self.output_channel)

        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)


    def forward(self, input):

        features1 = self.conv1(input)#64
        features1_p, indices1_p = self.pool1(features1)
        features2 = self.conv2(features1_p)#128
        features2_p, indices2_p = self.pool2(features2)
        features3 = self.conv3(features2_p)#256
        features3_p, indices3_p = self.pool3(features3)
        features4 = self.conv4(features3_p)#512
        features4_p, indices4_p = self.pool4(features4)
        features5 = self.conv5(features4_p)#512

        defeature5 = self.deconv5(features5)#512
        defeature4 = torch.cat((self.unpool4(defeature5,indices4_p), features4), 1)#1024
        defeature3t = self.deconv4(defeature4)#256
        defeature3 = torch.cat((self.unpool3(defeature3t,indices3_p), features3), 1)#512
        defeature2t = self.deconv3(defeature3)#128, 1/4
        defeature2 = torch.cat((self.unpool2(defeature2t,indices2_p), features2), 1)#256
        defeature1t = self.deconv2(defeature2)#64, 1/2
        defeature1 = torch.cat((self.unpool1(defeature1t,indices1_p), features1), 1)#128

        output = self.deconv1(defeature1)#3,1/1

        return output
        



