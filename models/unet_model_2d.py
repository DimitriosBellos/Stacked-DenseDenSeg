#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F

# python 3 confusing imports :(
from .unet_parts_2d import *


class UNet2D(nn.Module):
    def __init__(self, n_channels, n_classes, ignore_class=-100, n_layers=4, fls=32, MxPool=2, MaxPool=True, Kernel_size=3, Batchnorm=True, Stride=1, StridePool=2, Padding=1, Bias=False, trilinear=False):
        super(UNet2D, self).__init__()
        self.n_layers=n_layers
        self.inc = inconv(n_channels, fls, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.down1 = down(fls, 2*fls, MxPool, MaxPool, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.up4 = up(2*fls, MxPool, Batchnorm, Kernel_size, Stride, Padding, Bias, trilinear)
        if n_layers>=3:
            self.down2 = down(2*fls, 4*fls, MxPool, MaxPool, Batchnorm, Kernel_size, Stride, Padding, Bias)
            self.up3 = up(4*fls, MxPool, Batchnorm, Kernel_size, Stride, Padding, Bias, trilinear)
            if n_layers>=4:
                self.down3 = down(4*fls, 8*fls, MxPool, MaxPool, Batchnorm, Kernel_size, Stride, Padding, Bias)
                self.up2 = up(8*fls, MxPool, Batchnorm, Kernel_size, Stride, Padding, Bias, trilinear)
                if n_layers>=5:
                    self.down4 = down(8*fls, 16*fls, MxPool, MaxPool, Batchnorm, Kernel_size, Stride, Padding, Bias)
                    self.up1 = up(16*fls, MxPool, Batchnorm, Kernel_size, Stride, Padding, Bias, trilinear)
        self.outc = outconv(fls, n_classes, Kernel_size, Stride, Padding, Bias)
        self.ignore_class = ignore_class
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        n_layers=self.n_layers
        x1 = self.inc(x)
        x2 = self.down1(x1)
        if n_layers<3:
            x = self.up4(x2, x1)
        elif n_layers<4:
            x3 = self.down2(x2)
            x = self.up3(x3, x2)
            x = self.up4(x, x1)
        elif n_layers<5:
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x = self.up2(x4, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        else:
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x4, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        x = self.outc(x)
        if self.ignore_class > -1:
            s = list(x.shape)
            s[1] = 1
            x = torch.cat((x[:, :(self.ignore_class), :], torch.zeros(s), x[:, (self.ignore_class):, :]), dim=1)
        return x
