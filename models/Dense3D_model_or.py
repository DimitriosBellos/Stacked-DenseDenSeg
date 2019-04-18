#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F

# python 3 confusing imports :(
from .Dense3D_parts import *


class DenseNet(nn.Module):
    def __init__(self, n_channels, n_classes, ignore_class=-100, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=16, Kernel_size=3, Batchnorm=True, Stride=1, Padding=1):
        super(DenseNet, self).__init__()
        Bias = not Batchnorm
        k1 = 4
        #self.downini = nn.Conv3d(n_channels, n_channels, kernel_size=2, stride=2, padding=0, bias=Bias)
        #self.downini2 = nn.Conv3d(n_channels, n_channels, kernel_size=4, stride=4, padding=0, bias=Bias)
        self.inc = inconv(n_channels, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.down = conv_layer2(k0, k0, Batchnorm, 2, 2, 0, Bias)
        modules1 = 6
        self.db1 = DenseBlock(k0, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias, modules1)
        self.in_ch = int(k0 + modules1 * Growth_rate)
        a = k1 # n_classes #int(self.in_ch/8)
        self.up1 = nn.ConvTranspose3d(self.in_ch, n_classes, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.tr1 = TransitionBlock(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        modules2 = 12
        self.db2 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias, modules2)
        self.in_ch = int(self.in_ch + modules2 * Growth_rate)
        b = k1 # n_classes #int(self.in_ch/8)
        self.up2 = nn.ConvTranspose3d(self.in_ch, n_classes, kernel_size=6, stride=4, padding=1, bias=Bias)
        self.tr2 = TransitionBlock(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        modules3 = 24
        self.db3 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias, modules3)
        self.in_ch = int(self.in_ch + modules3 * Growth_rate)
        c = k1 # n_classes #int(self.in_ch/8)
        self.up3 = nn.ConvTranspose3d(self.in_ch, n_classes, kernel_size=10, stride=8, padding=1, bias=Bias)
        self.tr3 = TransitionBlock(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        modules4 = 16
        self.db4 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias, modules4)
        self.in_ch = int(self.in_ch + modules4 * Growth_rate)
        d = k1 # n_classes #int(self.in_ch/8)
        self.up4 = nn.ConvTranspose3d(self.in_ch, n_classes, kernel_size=18, stride=16, padding=1, bias=Bias)
        self.out = conv_layer2(k0 + a+b+c+d, n_classes, Batchnorm, 1, 1, 0, Bias)
        self.ignore_class = ignore_class
        # self.sfmx = nn.Softmax(1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x1 = self.inc(x)
        x = self.down(x1)
        x2 = self.db1(x)
        x = self.tr1(x2)
        x3 = self.db2(x)
        x = self.tr2(x3)
        x4 = self.db3(x)
        x = self.tr3(x4)
        x5 = self.db4(x)
        x2 = self.up1(x2)
        x3 = self.up2(x3)
        x4 = self.up3(x4)
        x5 = self.up4(x5)
        x = torch.cat([x5, x4, x3, x2, x1], dim=1)
        x = self.out(x)
        if self.ignore_class > -1:
            s = list(x.shape)
            s[1] = 1
            x = torch.cat((x[:, :(self.ignore_class), :], torch.zeros(s), x[:, (self.ignore_class):, :]), dim=1)
        return x
