# mask to confidence map for normal refinement
# IN
# Implemented in Pytorch by Jin Zeng, 20181019

import torch.nn as nn
import torch
from models_utils import *

class map_conv(nn.Module):
    
    def __init__(self, input_channel, output_channel, track_running_static=True):
        super(map_conv, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.track = track_running_static
        filters = [64, 64, 64, 64]

        self.conv1 = nn.Sequential(nn.Conv2d(self.input_channel, filters[0], 3, 1, 1),
                                  nn.InstanceNorm2d(filters[0], affine=True),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(filters[0], filters[1], 3, 1, 1),
                                  nn.InstanceNorm2d(filters[1], affine=True),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(filters[1], filters[2], 1),
                                #   nn.InstanceNorm2d(filters[1], affine=True),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(filters[2], filters[3], 1),
                                #   nn.InstanceNorm2d(filters[1], affine=True),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(filters[3], self.output_channel, 1),)

    def forward(self, input):

        output = self.conv1(input)

        return output
        



