# -*- coding: utf-8 -*-
#!/usr/bin/python
# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, Batchnorm=True, Kernel_size=3, Stride=1, Padding=1, Bias=True):
        super(double_conv, self).__init__()
        if Batchnorm:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, in_ch, Kernel_size, stride=Stride, padding=Padding, bias=Bias),
                nn.BatchNorm3d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_ch, out_ch, Kernel_size, stride=Stride, padding=Padding, bias=Bias),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, in_ch, Kernel_size, stride=Stride, padding=Padding, bias=Bias),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_ch, out_ch, Kernel_size, stride=Stride, padding=Padding, bias=Bias),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv_in(nn.Module):
    def __init__(self, in_ch, out_ch, Batchnorm=True, Kernel_size=3, Stride=1, Padding=1, Bias=True):
        super(double_conv_in, self).__init__()
        if Batchnorm:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, int(out_ch/2), Kernel_size, stride=Stride, padding=Padding, bias=Bias),
                nn.BatchNorm3d(int(out_ch/2)),
                nn.ReLU(inplace=True),
                nn.Conv3d(int(out_ch/2), out_ch, Kernel_size, stride=Stride, padding=Padding, bias=Bias),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, int(out_ch/2), Kernel_size, stride=Stride, padding=Padding, bias=Bias),
                nn.ReLU(inplace=True),
                nn.Conv3d(int(out_ch/2), out_ch, Kernel_size, stride=Stride, padding=Padding, bias=Bias),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, Batchnorm=True, Kernel_size=3, Stride=1, Padding=1, Bias=True):
        super(inconv, self).__init__()
        self.conv = double_conv_in(in_ch, out_ch, Batchnorm, Kernel_size, Stride, Padding, Bias)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, MxPool=2, MaxPool=True, Batchnorm=True, Kernel_size=3, Stride=1, Padding=1, Bias=True):
        super(down, self).__init__()
        if MaxPool:
            self.mpconv = nn.Sequential(
                nn.MaxPool3d(MxPool),
                double_conv(in_ch, out_ch, Batchnorm, Kernel_size, Stride, Padding, Bias)
            )
        else:
            self.mpconv = nn.Sequential(
                nn.Conv3d(in_ch, in_ch, MxPool, stride=MxPool),
                double_conv(in_ch, out_ch, Batchnorm, Kernel_size, Stride, Padding, Bias)
            )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, in_ch2, out_ch, factor=2, Batchnorm=True, Kernel_size= 3, Stride=1, Padding=1, Bias=True, trilinear=False):
        super(up, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        self.factor=factor
        if trilinear:
            self.up = nn.Upsample(scale_factor=factor, mode='trilinear')
        else:
            self.up = nn.ConvTranspose3d(in_ch, in_ch, factor, stride=factor)

        self.conv = double_conv(in_ch2, out_ch, Batchnorm, Kernel_size, Stride, Padding, Bias)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        factor = self.factor
        diffZ = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        diffX = x1.size()[4] - x2.size()[4]
        x2 = F.pad(x2, (diffZ // factor, int(diffZ / factor),
                        diffY // factor, int(diffY / factor),
                        diffX // factor, int(diffX / factor)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, Kernel_size= 1, Stride=1, Padding=1, Bias=True):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, Kernel_size, stride=Stride, padding=Padding, bias=Bias)
        #self.softmax = nn.Softmax(1)
        '''
        for a in range(0,int(self.conv.weight.data.shape[0])):
            for b in range(0,int(self.conv.weight.data.shape[1])):
                for c in range(0,int(self.conv.weight.data.shape[2])):
                    for d in range(0,int(self.conv.weight.data.shape[3])):
                        for e in range(0,int(self.conv.weight.data.shape[4])):
                            self.conv.weight.data[a][b][c][d][e]=torch.normal(torch.arange(0,1),torch.arange(0.001,0.0011,0.001)).numpy()[0].astype(float)
        for f in range(0,len(self.conv.bias.data)):
            self.conv.bias.data[f]=0
        '''

    def forward(self, x):
        x = self.conv(x)
        return x
