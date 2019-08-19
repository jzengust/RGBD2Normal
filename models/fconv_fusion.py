# fusion of RGB and depth branches
# Implemented in Pytorch by Jin Zeng, 20180724

import torch.nn as nn
import torch
from models_utils import create_conv_2, create_addon


class fconv(nn.Module):

    def __init__(self, input_channel1, input_channel2, output_channel, track_running_static=True):
        super(fconv, self).__init__()
        self.input_channel1 = input_channel1
        self.input_channel2 = input_channel2
        self.output_channel = output_channel
        self.track = track_running_static
        filters = [32, 16, 8]

        # encoder
        self.conv1 = create_conv_2(self.input_channel1 + self.input_channel2, filters[0], track=self.track)  # 6->32->32
        self.conv2 = create_conv_2(filters[0], filters[1], track=self.track)  # 32->16->16

        # decoder
        self.deconv1 = create_addon(filters[1], filters[2], self.output_channel)  # 16->8->3

    def forward(self, input1, input2):
        feature0 = torch.cat((input1, input2), 1)
        features1 = self.conv1(feature0)
        features2 = self.conv2(features1)

        output = self.deconv1(features2)

        return output
