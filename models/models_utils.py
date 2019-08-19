# Define basic modules
# Implemented by Jin Zeng, 20181008

import torch
import torch.nn as nn


# from sync_batchnorm import SynchronizedBatchNorm2d

class create_conv_2(nn.Module):

    def __init__(self, c1, c2, track=True):
        super(create_conv_2, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(c1, c2, 3, 1, 1),
                                  # SynchronizedBatchNorm2d(c2),
                                  nn.BatchNorm2d(c2),
                                  nn.ReLU(),
                                  nn.Conv2d(c2, c2, 3, 1, 1),
                                  # SynchronizedBatchNorm2d(c2),
                                  nn.BatchNorm2d(c2),
                                  nn.ReLU(), )

    def forward(self, input):
        output = self.conv(input)
        return output


#
class create_conv_3(nn.Module):

    def __init__(self, c1, c2, track=True):
        super(create_conv_3, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(c1, c2, 3, 1, 1),
                                  # SynchronizedBatchNorm2d(c2),
                                  nn.BatchNorm2d(c2),
                                  nn.ReLU(),
                                  nn.Conv2d(c2, c2, 3, 1, 1),
                                  # SynchronizedBatchNorm2d(c2),
                                  nn.BatchNorm2d(c2),
                                  nn.ReLU(),
                                  nn.Conv2d(c2, c2, 3, 1, 1),
                                  # SynchronizedBatchNorm2d(c2),
                                  nn.BatchNorm2d(c2),
                                  nn.ReLU(), )

    def forward(self, input):
        output = self.conv(input)
        return output


class create_deconv_3(nn.Module):

    def __init__(self, c1, c2, track=True):
        super(create_deconv_3, self).__init__()
        self.conv = nn.Sequential(nn.ConvTranspose2d(c1, c2, 3, 1, 1),
                                  # SynchronizedBatchNorm2d(c2),
                                  nn.BatchNorm2d(c2),
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(c2, c2, 3, 1, 1),
                                  # SynchronizedBatchNorm2d(c2),
                                  nn.BatchNorm2d(c2),
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(c2, c2, 3, 1, 1),
                                  # SynchronizedBatchNorm2d(c2),
                                  nn.BatchNorm2d(c2),
                                  nn.ReLU(), )

    def forward(self, input):
        output = self.conv(input)
        return output


class create_deconv_2(nn.Module):

    def __init__(self, c1, c2, track=True):
        super(create_deconv_2, self).__init__()
        self.conv = nn.Sequential(nn.ConvTranspose2d(c1, c2, 3, 1, 1),
                                  # SynchronizedBatchNorm2d(c2),
                                  nn.BatchNorm2d(c2),
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(c2, c2, 3, 1, 1),
                                  # SynchronizedBatchNorm2d(c2),
                                  nn.BatchNorm2d(c2),
                                  nn.ReLU(), )

    def forward(self, input):
        output = self.conv(input)
        return output


class create_addon(nn.Module):

    def __init__(self, c1, c2, c3):
        super(create_addon, self).__init__()
        self.conv = nn.Sequential(nn.ConvTranspose2d(c1, c2, 3, 1, 1),
                                  #   nn.BatchNorm2d(c2),
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(c2, c3, 3, 1, 1), )

    def forward(self, input):
        output = self.conv(input)
        return output


class create_conv_2_in(nn.Module):

    def __init__(self, c1, c2, track=True):
        super(create_conv_2_in, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(c1, c2, 3, 1, 1),
                                  nn.InstanceNorm2d(c2, affine=True),
                                  nn.ReLU(),
                                  nn.Conv2d(c2, c2, 3, 1, 1),
                                  nn.InstanceNorm2d(c2, affine=True),
                                  nn.ReLU(), )

    def forward(self, input):
        output = self.conv(input)
        return output


class create_conv_3_in(nn.Module):

    def __init__(self, c1, c2, track=True):
        super(create_conv_3_in, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(c1, c2, 3, 1, 1),
                                  nn.InstanceNorm2d(c2, affine=True),
                                  nn.ReLU(),
                                  nn.Conv2d(c2, c2, 3, 1, 1),
                                  nn.InstanceNorm2d(c2, affine=True),
                                  nn.ReLU(),
                                  nn.Conv2d(c2, c2, 3, 1, 1),
                                  nn.InstanceNorm2d(c2, affine=True),
                                  nn.ReLU(), )

    def forward(self, input):
        output = self.conv(input)
        return output


class create_deconv_3_in(nn.Module):

    def __init__(self, c1, c2, track=True):
        super(create_deconv_3_in, self).__init__()
        self.conv = nn.Sequential(nn.ConvTranspose2d(c1, c2, 3, 1, 1),
                                  nn.InstanceNorm2d(c2, affine=True),
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(c2, c2, 3, 1, 1),
                                  nn.InstanceNorm2d(c2, affine=True),
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(c2, c2, 3, 1, 1),
                                  nn.InstanceNorm2d(c2, affine=True),
                                  nn.ReLU(), )

    def forward(self, input):
        output = self.conv(input)
        return output


class create_deconv_2_in(nn.Module):

    def __init__(self, c1, c2, track=True):
        super(create_deconv_2_in, self).__init__()
        self.conv = nn.Sequential(nn.ConvTranspose2d(c1, c2, 3, 1, 1),
                                  nn.InstanceNorm2d(c2, affine=True),
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(c2, c2, 3, 1, 1),
                                  nn.InstanceNorm2d(c2, affine=True),
                                  nn.ReLU(), )

    def forward(self, input):
        output = self.conv(input)
        return output
