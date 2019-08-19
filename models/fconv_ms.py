# fusion of RGB and depth at multiple scales in the decoder phase
# Note: 1. Use IN instead of BN
#       2. fuse at 4 levels
#       3. output at multi level
# Implemented in Pytorch by Jin Zeng, 20181022

import torch.nn as nn
import torch
from models_utils import *

class fconv_ms(nn.Module):
    
    def __init__(self, input_channel1, input_channel2, output_channel, track_running_static=True):
        super(fconv_ms, self).__init__()
        self.input_channel1 = input_channel1
        self.input_channel2 = input_channel2
        self.output_channel = output_channel
        self.track = track_running_static
        filters_rgb = [64, 128, 256, 256, 256]
        filters_d = [64, 128, 256, 256]
        filters_fconv = [32, 16, 8]

        # encoder for RGB
        self.conv1 = create_conv_2_in(self.input_channel1, filters_rgb[0],track=self.track)# 3-64
        self.conv2 = create_conv_2_in(filters_rgb[0], filters_rgb[1],track=self.track)#64-128
        self.conv3 = create_conv_3_in(filters_rgb[1], filters_rgb[2],track=self.track)#128-256
        self.conv4 = create_conv_3_in(filters_rgb[2], filters_rgb[3],track=self.track)#256-256
        self.conv5 = create_conv_3_in(filters_rgb[3], filters_rgb[4],track=self.track)#256-256

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # encoder for depth
        self.conv1_d = create_conv_2_in(self.input_channel2, filters_d[0],track=self.track)
        self.conv2_d = create_conv_2_in(filters_d[0], filters_d[1],track=self.track)
        self.conv3_d = create_conv_2_in(filters_d[1], filters_d[2],track=self.track)
        self.conv4_d = create_conv_2_in(filters_d[2], filters_d[3],track=self.track)

        self.pool1_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pool2_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pool3_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)

        # decoder for depth
        self.deconv4_d = create_deconv_2_in(filters_d[3], filters_d[2],track=self.track)
        self.deconv3_d = create_deconv_2_in(filters_d[2]+filters_d[2], filters_d[1],track=self.track)
        self.deconv2_d = create_deconv_2_in(filters_d[1]+filters_d[1], filters_d[0],track=self.track)
        self.deconv1_d = create_addon(filters_d[0]+filters_d[0], filters_d[0], self.output_channel)

        self.unpool1_d = nn.Upsample(scale_factor=2, mode='bilinear')
        self.unpool2_d = nn.Upsample(scale_factor=2, mode='bilinear')
        self.unpool3_d = nn.Upsample(scale_factor=2, mode='bilinear')

        # decoder for RGB
        self.deconv5 = create_deconv_3_in(filters_rgb[4], filters_rgb[3],track=self.track)
        self.deconv4 = create_deconv_3_in(filters_rgb[3]+filters_rgb[3], filters_rgb[2],track=self.track)
        self.deconv3 = create_deconv_3_in(filters_rgb[2]+filters_rgb[2], filters_rgb[1],track=self.track)
        self.deconv2 = create_deconv_2_in(filters_rgb[1]+filters_rgb[1], filters_rgb[0],track=self.track)
        self.deconv1 = create_addon(filters_rgb[0]+filters_rgb[0], filters_rgb[0], self.output_channel)

        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # Fusion convs
        self.deconv3_f = create_deconv_2_in(filters_rgb[2]+filters_rgb[2], filters_rgb[2],track=self.track)
        self.deconv2_f = create_deconv_2_in(filters_rgb[1]+filters_rgb[1], filters_rgb[1],track=self.track)
        self.deconv1_f = create_deconv_2_in(filters_rgb[0]+filters_rgb[0], filters_rgb[0],track=self.track)

        # encoder in the final stage
        self.conv1_f1 = create_conv_2_in(self.input_channel1+self.input_channel1, filters_fconv[0],track=self.track) # 6->32->32
        self.conv2_f1 = create_conv_2_in(filters_fconv[0], filters_fconv[1],track=self.track) # 32->16->16
        # decoder in the final stage
        self.deconv1_f1 = create_addon(filters_fconv[1], filters_fconv[2], self.output_channel) # 16->8->3

        # multiscale loss
        self.deconv3_ms = nn.ConvTranspose2d(filters_rgb[2], 3, 3, 1, 1)
        self.deconv2_ms = nn.ConvTranspose2d(filters_rgb[1], 3, 3, 1, 1)
        self.deconv1_ms = nn.ConvTranspose2d(filters_rgb[0], 3, 3, 1, 1)

    def forward(self, input1, input2, mask):

        filters_d = [64, 128, 256, 256]
        # encoder RGB
        features1 = self.conv1(input1)#64
        features1_p, indices1_p = self.pool1(features1)
        features2 = self.conv2(features1_p)#128
        features2_p, indices2_p = self.pool2(features2)
        features3 = self.conv3(features2_p)#256
        features3_p, indices3_p = self.pool3(features3)
        features4 = self.conv4(features3_p)#256
        features4_p, indices4_p = self.pool4(features4)
        features5 = self.conv5(features4_p)#256

        # encoder depth
        features1_d = self.conv1_d(input2)#64
        features1_p_d= self.pool1_d(features1_d)
        features2_d = self.conv2_d(features1_p_d)#128
        features2_p_d = self.pool2_d(features2_d)
        features3_d = self.conv3_d(features2_p_d)#256
        features3_p_d = self.pool3_d(features3_d)
        features4_d = self.conv4_d(features3_p_d)#256

        # encoder mask, follow depth, generate 4 difference sizes
        mask = mask.unsqueeze(0).permute(1,0,2,3)
        mask1_p = self.pool1_d(mask)
        mask2_p = self.pool2_d(mask1_p)
        mask3_p = self.pool3_d(mask2_p)

        # decoder depth, as the giver
        defeature3t_d = self.deconv4_d(features4_d)#256, 1/8
        defeature3_d = torch.cat((self.unpool3_d(defeature3t_d), features3_d), 1)#512
        defeature2t_d = self.deconv3_d(defeature3_d)#128, 1/4
        defeature2_d = torch.cat((self.unpool2_d(defeature2t_d), features2_d), 1)#256
        defeature1t_d = self.deconv2_d(defeature2_d)#64, 1/2
        defeature1_d = torch.cat((self.unpool1_d(defeature1t_d), features1_d), 1)#128
        output_d = self.deconv1_d(defeature1_d)#3, 1/1

        # masking features from depth
        mask = mask.repeat(1,3,1,1)
        mask1_p = mask1_p.repeat(1,filters_d[0],1,1)
        mask2_p = mask2_p.repeat(1,filters_d[1],1,1)
        mask3_p = mask3_p.repeat(1,filters_d[2],1,1)
        output_d = output_d*mask#3, 1/1
        defeature1t_d = defeature1t_d*mask1_p#64, 1/2
        defeature2t_d = defeature2t_d*mask2_p#128, 1/4
        defeature3t_d = defeature3t_d*mask3_p#256, 1/8

        # decorder for rgb
        defeature5 = self.deconv5(features5)#256
        defeature4 = torch.cat((self.unpool4(defeature5,indices4_p), features4), 1)#512
        defeature3t = self.deconv4(defeature4)#256, 1/8
        defeature3t_f = self.deconv3_f(torch.cat((defeature3t, defeature3t_d),1))#512->256
        defeature3 = torch.cat((self.unpool3(defeature3t_f,indices3_p), features3), 1)#512

        defeature2t = self.deconv3(defeature3)#128, 1/4
        defeature2t_f = self.deconv2_f(torch.cat((defeature2t, defeature2t_d),1))#256->128
        defeature2 = torch.cat((self.unpool2(defeature2t_f,indices2_p), features2), 1)#256

        defeature1t = self.deconv2(defeature2)#64, 1/2
        defeature1t_f = self.deconv1_f(torch.cat((defeature1t, defeature1t_d),1))#128->64        
        defeature1 = torch.cat((self.unpool1(defeature1t_f,indices1_p), features1), 1)#128        

        output = self.deconv1(defeature1)#3,1/1
        feature0 = torch.cat((output, output_d), 1)
        features1 = self.conv1_f1(feature0)
        features2 = self.conv2_f1(features1)
        output_f = self.deconv1_f1(features2)

        # ms
        output_f3 = self.deconv3_ms(defeature3t_f)
        output_f2 = self.deconv2_ms(defeature2t_f)
        output_f1 = self.deconv1_ms(defeature1t_f)

        return output_f, output_f1, output_f2, output_f3,output_d
        



