#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F

# python 3 confusing imports :(
from .Dense3D_parts import *
from .Dense2D_parts import *


class DenseModel(nn.Module):
    def __init__(self, n_channels, n_classes, ignore_class=-100, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=16, Kernel_size=3, Batchnorm=True, Stride=1, Padding=1, Bias=False):
        super(DenseModel, self).__init__()
        self.inc = inconv(n_channels, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.down = conv_layer2(k0, k0, Batchnorm, 2, 2, 0, Bias)
        self.db1 = DenseBlock(k0, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(k0 + 4 * Growth_rate)
        self.up1 = nn.ConvTranspose3d(self.in_ch, n_classes, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.tr1 = TransitionBlock(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db2 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up2 = nn.ConvTranspose3d(self.in_ch, n_classes, kernel_size=6, stride=4, padding=1, bias=Bias)
        self.tr2 = TransitionBlock(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db3 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up3 = nn.ConvTranspose3d(self.in_ch, n_classes, kernel_size=10, stride=8, padding=1, bias=Bias)
        self.tr3 = TransitionBlock(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db4 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up4 = nn.ConvTranspose3d(self.in_ch, n_classes, kernel_size=18, stride=16, padding=1, bias=Bias)
        self.out = conv_layer2(k0 + 4 * n_classes, n_classes, Batchnorm, 1, 1, 0, Bias)
        self.ignore_class = ignore_class
        # self.sfmx = nn.Softmax(1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        #x0 = x
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
        return x


class DenseModel5L(nn.Module):
    def __init__(self, n_channels, n_classes, ignore_class=-100, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=16, Kernel_size=3, Batchnorm=True, Stride=1, Padding=1, Bias=False):
        super(DenseModel5L, self).__init__()
        self.inc = inconv(n_channels, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.down = conv_layer2(k0, k0, Batchnorm, 2, 2, 0, Bias)
        self.db1 = DenseBlock(k0, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(k0 + 4 * Growth_rate)
        self.up1 = nn.ConvTranspose3d(self.in_ch, n_classes, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.tr1 = TransitionBlock(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db2 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up2 = nn.ConvTranspose3d(self.in_ch, n_classes, kernel_size=6, stride=4, padding=1, bias=Bias)
        self.tr2 = TransitionBlock(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db3 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up3 = nn.ConvTranspose3d(self.in_ch, n_classes, kernel_size=10, stride=8, padding=1, bias=Bias)
        self.tr3 = TransitionBlock(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db4 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up4 = nn.ConvTranspose3d(self.in_ch, n_classes, kernel_size=18, stride=16, padding=1, bias=Bias)
        self.tr4 = TransitionBlock(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db5 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up5 = nn.ConvTranspose3d(self.in_ch, n_classes, kernel_size=34, stride=32, padding=1, bias=Bias)
        self.out = conv_layer2(k0 + 5 * n_classes, n_classes, Batchnorm, 1, 1, 0, Bias)
        self.ignore_class = ignore_class
        # self.sfmx = nn.Softmax(1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #x0 = x
        x1 = self.inc(x)
        x = self.down(x1)
        x2 = self.db1(x)
        x = self.tr1(x2)
        x3 = self.db2(x)
        x = self.tr2(x3)
        x4 = self.db3(x)
        x = self.tr3(x4)
        x5 = self.db4(x)
        x = self.tr4(x5)
        x6 = self.db5(x)
        x2 = self.up1(x2)
        x3 = self.up2(x3)
        x4 = self.up3(x4)
        x5 = self.up4(x5)
        x6 = self.up5(x6)
        x = torch.cat([x6, x5, x4, x3, x2, x1], dim=1)
        x = self.out(x)
        if self.ignore_class > -1:
            s = list(x.shape)
        return x


class DenseModelDenoise(nn.Module):
    def __init__(self, n_channels, n_classes, ignore_class=-100, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=16, Kernel_size=3, Batchnorm=True, Stride=1, Padding=1, Bias=False):
        super(DenseModelDenoise, self).__init__()
        #self.inc = inconv(n_channels, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.down = conv_layer2(k0, k0, Batchnorm, 2, 2, 0, Bias)
        self.db1 = DenseBlock(k0, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(k0 + 4 * Growth_rate)
        self.up1 = nn.ConvTranspose3d(self.in_ch, 4, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.tr1 = TransitionBlock(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db2 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up2 = nn.ConvTranspose3d(self.in_ch, 4, kernel_size=6, stride=4, padding=1, bias=Bias)
        self.tr2 = TransitionBlock(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db3 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up3 = nn.ConvTranspose3d(self.in_ch, 4, kernel_size=10, stride=8, padding=1, bias=Bias)
        self.tr3 = TransitionBlock(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db4 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up4 = nn.ConvTranspose3d(self.in_ch, 4, kernel_size=18, stride=16, padding=1, bias=Bias)
        self.out = conv_layer2(k0 + 4 * 4, n_classes, Batchnorm, 1, 1, 0, Bias)
        self.ignore_class = ignore_class
        # self.sfmx = nn.Softmax(1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x1):
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
        x_1 = self.out(x)
        if self.ignore_class > -1:
            s = list(x.shape)
        return x, x_1


class DenseModelStack(nn.Module):
    def __init__(self, n_channels, n_classes, ignore_class=-100, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=16, Kernel_size=3, Batchnorm=True, Stride=1, Padding=1, Bias=False):
        super(DenseModelStack, self).__init__()
        self.inc = inconv(n_channels, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.D1 = DenseModelDenoise(n_channels, 1, ignore_class, k0, Theta, Dropout, Growth_rate, Kernel_size, Batchnorm, Stride, Padding, Bias)
        self.c1a = conv_layer2(k0 + 4 * 4, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.c1b = conv_layer2(k0, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.c2b = conv_layer2(1, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.D2 = DenseModel(n_channels, n_classes, ignore_class, k0, Theta, Dropout, Growth_rate, Kernel_size, Batchnorm, Stride, Padding, Bias)

    def forward(self, x):
        x1a = self.inc(x)
        x, x_1 = self.D1(x1a)
        x = self.c1a(x)
        x = self.c1b(x)
        x1b = self.c2b(x_1)
        x = x1a + x1b + x 
        x_2 = self.D2(x)
        return x_2, x_1


class DenseUSegDenoise(nn.Module):
    def __init__(self, n_channels, n_classes, ignore_class=-100, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=16, Kernel_size=3, Batchnorm=True, Stride=1, Padding=1, Bias=False):
        super(DenseUSegDenoise, self).__init__()
        self.down = conv_layer2(k0, k0, Batchnorm, 2, 2, 0, Bias)
        self.db1 = DenseBlock(k0, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(k0 + 4 * Growth_rate)
        self.a = self.in_ch
        self.up1 = nn.ConvTranspose3d(int(self.in_ch), k0, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.conv1 = convUp(int(k0*2), Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.tr1 = TransitionBlock(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db2 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.b = self.in_ch
        self.up2 = nn.ConvTranspose3d(int(self.in_ch), self.a, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.conv2 = convUp(int(self.a*2), Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.tr2 = TransitionBlock(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db3 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.c = self.in_ch
        self.up3 = nn.ConvTranspose3d(int(self.in_ch), self.b, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.conv3 = convUp(int(self.b*2), Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.tr3 = TransitionBlock(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db4 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up4 = nn.ConvTranspose3d(int(self.in_ch), self.c, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.conv4 = convUp(int(self.c*2), Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.out = conv_layer2(k0, n_classes, Batchnorm, 1, 1, 0, Bias)
        self.ignore_class = ignore_class
        # self.sfmx = nn.Softmax(1)

    def forward(self, x1):
        x = self.down(x1)
        x2 = self.db1(x)
        x = self.tr1(x2)
        x3 = self.db2(x)
        x = self.tr2(x3)
        x4 = self.db3(x)
        x = self.tr3(x4)
        x = self.db4(x)
        x = self.up4(x)
        x = torch.cat([x4, x], dim=1)
        x = self.conv4(x)
        x = self.up3(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv3(x)
        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv2(x)
        x = self.up1(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv1(x)
        x_1 = self.out(x)
        if self.ignore_class > -1:
            s = list(x.shape)
            s[1] = 1
            x = torch.cat((x[:, :(self.ignore_class), :], torch.zeros(s), x[:, (self.ignore_class):, :]), dim=1)
        return x, x_1


class DenseUSeg(nn.Module):
    def __init__(self, n_channels, n_classes, ignore_class=-100, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=16, Kernel_size=3, Batchnorm=True, Stride=1, Padding=1, Bias=False):
        super(DenseUSeg, self).__init__()
        self.inc = inconv(n_channels, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.DenseU = DenseUSegStrip(n_channels, n_classes, ignore_class, k0, Theta, 0, Growth_rate, Kernel_size, Batchnorm, Stride, Padding, Bias)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x = self.DenseU(x1)
        return x


class DenseUSegStrip(nn.Module):
    def __init__(self, n_channels, n_classes, ignore_class=-100, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=16, Kernel_size=3, Batchnorm=True, Stride=1, Padding=1, Bias=False):
        super(DenseUSegStrip, self).__init__()
        self.down = conv_layer2(k0, k0, Batchnorm, 2, 2, 0, Bias)
        self.db1 = DenseBlock(k0, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(k0 + 4 * Growth_rate)
        self.a = self.in_ch
        self.up1 = nn.ConvTranspose3d(int(self.in_ch), k0, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.conv1 = convUp(int(k0*2), Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.tr1 = TransitionBlock(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db2 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.b = self.in_ch
        self.up2 = nn.ConvTranspose3d(int(self.in_ch), self.a, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.conv2 = convUp(int(self.a*2), Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.tr2 = TransitionBlock(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db3 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.c = self.in_ch
        self.up3 = nn.ConvTranspose3d(int(self.in_ch), self.b, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.conv3 = convUp(int(self.b*2), Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.tr3 = TransitionBlock(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db4 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up4 = nn.ConvTranspose3d(int(self.in_ch), self.c, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.conv4 = convUp(int(self.c*2), Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.out = conv_layer2(k0, n_classes, Batchnorm, 1, 1, 0, Bias)
        self.ignore_class = ignore_class
        # self.sfmx = nn.Softmax(1)

    def forward(self, x1):
        x = self.down(x1)
        x2 = self.db1(x)
        x = self.tr1(x2)
        x3 = self.db2(x)
        x = self.tr2(x3)
        x4 = self.db3(x)
        x = self.tr3(x4)
        x = self.db4(x)
        x = self.up4(x)
        x = torch.cat([x4, x], dim=1)
        x = self.conv4(x)
        x = self.up3(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv3(x)
        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv2(x)
        x = self.up1(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv1(x)
        x = self.out(x)
        if self.ignore_class > -1:
            s = list(x.shape)
            s[1] = 1
            x = torch.cat((x[:, :(self.ignore_class), :], torch.zeros(s), x[:, (self.ignore_class):, :]), dim=1)
        return x


class DenseUSegStack(nn.Module):
    def __init__(self, n_channels, n_classes, ignore_class=-100, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=16, Kernel_size=3, Batchnorm=True, Stride=1, Padding=1, Bias=False):
        super(DenseUSegStack, self).__init__()
        self.inc = inconv(n_channels, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.D1 = DenseUSegDenoise(n_channels, 1, ignore_class, k0, Theta, Dropout, Growth_rate, Kernel_size, Batchnorm, Stride, Padding, Bias)
        #self.c = inconv(1, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.c1 = conv_layer(k0, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.c2 = conv_layer(1, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.D2 = DenseUSegStrip(n_channels, n_classes, ignore_class, k0, Theta, 0, Growth_rate, Kernel_size, Batchnorm, Stride, Padding, Bias)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1a = self.inc(x)
        x, x_1 = self.D1(x1a)
        x = self.c1(x)
        x1b = self.c2(x_1)
        x = x1a + x1b + x
        #x = self.c(x_1)
        x_2 = self.D2(x)
        return x_2, x_1


class DenseModelHalfScope(nn.Module):
    def __init__(self, n_channels, n_classes, ignore_class=-100, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=16, Kernel_size=3, Batchnorm=True, Stride=1, Padding=1, Bias=False):
        super(DenseModelHalfScope, self).__init__()
        self.inc = inconv(n_channels, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.down = conv_layer2(k0, k0, Batchnorm, 2, 2, 0, Bias)
        self.db1 = DenseBlock(k0, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(k0 + 4 * Growth_rate)
        self.up1 = nn.Conv3d(self.in_ch, n_classes, Kernel_size, stride=Stride, padding=Padding, bias=Bias)
        self.tr1 = TransitionBlock(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db2 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up2 = nn.ConvTranspose3d(self.in_ch, n_classes, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.tr2 = TransitionBlock(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db3 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up3 = nn.ConvTranspose3d(self.in_ch, n_classes, kernel_size=6, stride=4, padding=1, bias=Bias)
        self.tr3 = TransitionBlock(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db4 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up4 = nn.ConvTranspose3d(self.in_ch, n_classes, kernel_size=10, stride=8, padding=1, bias=Bias)
        self.out = conv_layer2(k0 + 4 * n_classes, n_classes, Batchnorm, 1, 1, 0, Bias)
        self.ignore_class = ignore_class
        # self.sfmx = nn.Softmax(1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        #x0 = x
        x = self.inc(x)
        x1 = self.down(x)
        x2 = self.db1(x1)
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
        return x


class DenseModel2_5(nn.Module):
    def __init__(self, n_channels, n_classes, ignore_class=-100, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=16, Kernel_size=3, Batchnorm=True, Stride=1, Padding=1, Bias=False):
        super(DenseModel2_5, self).__init__()
        self.inc = inconv(n_channels, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.down = conv_layer2(k0, k0, Batchnorm, (5, 2, 2), (3, 2, 2), 0, Bias)
        self.db1 = DenseBlock(k0, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(k0 + 4 * Growth_rate)
        self.up1 = nn.ConvTranspose3d(self.in_ch, n_classes, kernel_size=(5,4,4), stride=(4,2,2), padding=1, bias=Bias)
        self.tr1 = TransitionBlock2(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db2 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up2 = nn.ConvTranspose3d(self.in_ch, n_classes, kernel_size=(5,6,6), stride=4, padding=1, bias=Bias)
        self.tr2 = TransitionBlock2(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db3 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up3 = nn.ConvTranspose3d(self.in_ch, n_classes, kernel_size=(5,10,10), stride=(4,8,8), padding=1, bias=Bias)
        self.tr3 = TransitionBlock2(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db4 = DenseBlock(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up4 = nn.ConvTranspose3d(self.in_ch, n_classes, kernel_size=(5,18,18), stride=(4,16,16), padding=1, bias=Bias)
        self.out = conv_layer2(k0 + 4 * n_classes, n_classes, Batchnorm, 1, 1, 0, Bias)
        self.ignore_class = ignore_class
        # self.sfmx = nn.Softmax(1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        #x0 = x
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
        return x


class Dense2DModel(nn.Module):
    def __init__(self, n_channels, n_classes, ignore_class=-100, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=16, Kernel_size=3, Batchnorm=True, Stride=1, Padding=1, Bias=False):
        super(Dense2DModel, self).__init__()
        self.inc = inconv_2D(n_channels, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.down = conv_layer2_2D(k0, k0, Batchnorm, 2, 2, 0, Bias)
        self.db1 = DenseBlock_2D(k0, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(k0 + 4 * Growth_rate)
        self.up1 = nn.ConvTranspose2d(self.in_ch, n_classes, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.tr1 = TransitionBlock_2D(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db2 = DenseBlock_2D(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up2 = nn.ConvTranspose2d(self.in_ch, n_classes, kernel_size=6, stride=4, padding=1, bias=Bias)
        self.tr2 = TransitionBlock_2D(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db3 = DenseBlock_2D(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up3 = nn.ConvTranspose2d(self.in_ch, n_classes, kernel_size=10, stride=8, padding=1, bias=Bias)
        self.tr3 = TransitionBlock_2D(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db4 = DenseBlock_2D(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up4 = nn.ConvTranspose2d(self.in_ch, n_classes, kernel_size=18, stride=16, padding=1, bias=Bias)
        self.out = conv_layer2_2D(k0 + 4 * n_classes, n_classes, Batchnorm, 1, 1, 0, Bias)
        self.ignore_class = ignore_class
        # self.sfmx = nn.Softmax(1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #x0 = x
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
        return x


class Dense2DModel5L(nn.Module):
    def __init__(self, n_channels, n_classes, ignore_class=-100, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=16, Kernel_size=3, Batchnorm=True, Stride=1, Padding=1, Bias=False):
        super(Dense2DModel5L, self).__init__()
        self.inc = inconv_2D(n_channels, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.down = conv_layer2_2D(k0, k0, Batchnorm, 2, 2, 0, Bias)
        self.db1 = DenseBlock_2D(k0, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(k0 + 4 * Growth_rate)
        self.up1 = nn.ConvTranspose2d(self.in_ch, n_classes, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.tr1 = TransitionBlock_2D(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db2 = DenseBlock_2D(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up2 = nn.ConvTranspose2d(self.in_ch, n_classes, kernel_size=6, stride=4, padding=1, bias=Bias)
        self.tr2 = TransitionBlock_2D(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db3 = DenseBlock_2D(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up3 = nn.ConvTranspose2d(self.in_ch, n_classes, kernel_size=10, stride=8, padding=1, bias=Bias)
        self.tr3 = TransitionBlock_2D(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db4 = DenseBlock_2D(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up4 = nn.ConvTranspose2d(self.in_ch, n_classes, kernel_size=18, stride=16, padding=1, bias=Bias)
        self.tr4 = TransitionBlock_2D(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db5 = DenseBlock_2D(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up5 = nn.ConvTranspose2d(self.in_ch, n_classes, kernel_size=34, stride=32, padding=1, bias=Bias)
        self.out = conv_layer2_2D(k0 + 5 * n_classes, n_classes, Batchnorm, 1, 1, 0, Bias)
        self.ignore_class = ignore_class
        # self.sfmx = nn.Softmax(1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #x0 = x
        x1 = self.inc(x)
        x = self.down(x1)
        x2 = self.db1(x)
        x = self.tr1(x2)
        x3 = self.db2(x)
        x = self.tr2(x3)
        x4 = self.db3(x)
        x = self.tr3(x4)
        x5 = self.db4(x)
        x = self.tr4(x5)
        x6 = self.db5(x)
        x2 = self.up1(x2)
        x3 = self.up2(x3)
        x4 = self.up3(x4)
        x5 = self.up4(x5)
        x6 = self.up5(x6)
        x = torch.cat([x6, x5, x4, x3, x2, x1], dim=1)
        x = self.out(x)
        if self.ignore_class > -1:
            s = list(x.shape)
        return x


class Dense2DModelDenoise(nn.Module):
    def __init__(self, n_channels, n_classes, ignore_class=-100, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=16, Kernel_size=3, Batchnorm=True, Stride=1, Padding=1, Bias=False):
        super(Dense2DModelDenoise, self).__init__()
        #self.inc = inconv_2D(n_channels, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.down = conv_layer2_2D(k0, k0, Batchnorm, 2, 2, 0, Bias)
        self.db1 = DenseBlock_2D(k0, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(k0 + 4 * Growth_rate)
        self.up1 = nn.ConvTranspose2d(self.in_ch, 4, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.tr1 = TransitionBlock_2D(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db2 = DenseBlock_2D(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up2 = nn.ConvTranspose2d(self.in_ch, 4, kernel_size=6, stride=4, padding=1, bias=Bias)
        self.tr2 = TransitionBlock_2D(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db3 = DenseBlock_2D(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up3 = nn.ConvTranspose2d(self.in_ch, 4, kernel_size=10, stride=8, padding=1, bias=Bias)
        self.tr3 = TransitionBlock_2D(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db4 = DenseBlock_2D(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch = int(self.in_ch + 4 * Growth_rate)
        self.up4 = nn.ConvTranspose2d(self.in_ch, 4, kernel_size=18, stride=16, padding=1, bias=Bias)
        self.out = conv_layer2_2D(k0 + 4 * 4, n_classes, Batchnorm, 1, 1, 0, Bias)
        self.ignore_class = ignore_class
        # self.sfmx = nn.Softmax(1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x1):
        #x0 = x
        #x1 = self.inc(x)
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
        x_1 = self.out(x)
        if self.ignore_class > -1:
            s = list(x.shape)
        return x, x_1


class DenseUSeg2D(nn.Module):
    def __init__(self, n_channels, n_classes, ignore_class=-100, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=16, Kernel_size=3, Batchnorm=True, Stride=1, Padding=1, Bias=False):
        super(DenseUSeg2D, self).__init__()
        #self.inc = inconv_2D(n_channels, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.down = conv_layer2_2D(k0, k0, Batchnorm, 2, 2, 0, Bias)
        self.db1 = DenseBlock_2D(k0, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias, 6)
        self.in_ch = int(k0 + 6 * Growth_rate)
        self.a = self.in_ch
        self.up1 = nn.ConvTranspose2d(int(self.in_ch), k0, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.conv1 = convUp_2D(int(k0*2), Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.tr1 = TransitionBlock_2D(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db2 = DenseBlock_2D(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias, 12)
        self.in_ch = int(self.in_ch + 12 * Growth_rate)
        self.b = self.in_ch
        self.up2 = nn.ConvTranspose2d(int(self.in_ch), self.a, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.conv2 = convUp_2D(int(self.a*2), Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.tr2 = TransitionBlock_2D(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db3 = DenseBlock_2D(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias, 24)
        self.in_ch = int(self.in_ch + 24 * Growth_rate)
        self.c = self.in_ch
        self.up3 = nn.ConvTranspose2d(int(self.in_ch), self.b, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.conv3 = convUp_2D(int(self.b*2), Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.tr3 = TransitionBlock_2D(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db4 = DenseBlock_2D(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias, 16)
        self.in_ch = int(self.in_ch + 16 * Growth_rate)
        self.up4 = nn.ConvTranspose2d(int(self.in_ch), self.c, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.conv4 = convUp_2D(int(self.c*2), Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.out = conv_layer2_2D(k0, n_classes, Batchnorm, 1, 1, 0, Bias)
        self.ignore_class = ignore_class
        # self.sfmx = nn.Softmax(1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x1):
        #x1 = self.inc(x)
        x = self.down(x1)
        x2 = self.db1(x)
        x = self.tr1(x2)
        x3 = self.db2(x)
        x = self.tr2(x3)
        x4 = self.db3(x)
        x = self.tr3(x4)
        x = self.db4(x)
        x = self.up4(x)
        x = torch.cat([x4, x], dim=1)
        x = self.conv4(x)
        x = self.up3(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv3(x)
        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv2(x)
        x = self.up1(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv1(x)
        x = self.out(x)
        if self.ignore_class > -1:
            s = list(x.shape)
            s[1] = 1
            x = torch.cat((x[:, :(self.ignore_class), :], torch.zeros(s), x[:, (self.ignore_class):, :]), dim=1)
        return x


class DenseModelStack_2D(nn.Module):
    def __init__(self, n_channels, n_classes, ignore_class=-100, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=16, Kernel_size=3, Batchnorm=True, Stride=1, Padding=1, Bias=False):
        super(DenseModelStack_2D, self).__init__()
        self.inc = inconv_2D(n_channels, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.D1 = Dense2DModelDenoise(n_channels, 1, ignore_class, k0, Theta, Dropout, Growth_rate, Kernel_size, Batchnorm, Stride, Padding, Bias)
        self.c1a = conv_layer2_2D(k0 + 4 * 4, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.c1b = conv_layer2_2D(k0, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.c2b = conv_layer2_2D(1, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.D2 = Dense2DModel(n_channels, n_classes, ignore_class, k0, Theta, Dropout, Growth_rate, Kernel_size, Batchnorm, Stride, Padding, Bias)

    def forward(self, x):
        x1a = self.inc(x)
        x, x_1 = self.D1(x1a)
        x = self.c1a(x)
        x = self.c1b(x)
        x1b = self.c2b(x_1)
        x = x1a + x1b + x
        x_2 = self.D2(x)
        return x_2, x_1


class DenseUSegDenoise2D(nn.Module):
    def __init__(self, n_channels, n_classes, ignore_class=-100, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=16, Kernel_size=3, Batchnorm=True, Stride=1, Padding=1, Bias=False):
        super(DenseUSegDenoise2D, self).__init__()
        #self.inc = inconv_2D(n_channels, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.down = conv_layer2_2D(k0, k0, Batchnorm, 2, 2, 0, Bias)
        self.db1 = DenseBlock_2D(k0, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias, 3)
        self.in_ch = int(k0 + 3 * Growth_rate)
        self.a = self.in_ch
        self.up1 = nn.ConvTranspose2d(int(self.in_ch), k0, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.conv1 = convUp_2D(int(k0*2), Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.tr1 = TransitionBlock_2D(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db2 = DenseBlock_2D(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias, 6)
        self.in_ch = int(self.in_ch + 6 * Growth_rate)
        self.b = self.in_ch
        self.up2 = nn.ConvTranspose2d(int(self.in_ch), self.a, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.conv2 = convUp_2D(int(self.a*2), Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.tr2 = TransitionBlock_2D(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db3 = DenseBlock_2D(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias, 12)
        self.in_ch = int(self.in_ch + 12 * Growth_rate)
        self.c = self.in_ch
        self.up3 = nn.ConvTranspose2d(int(self.in_ch), self.b, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.conv3 = convUp_2D(int(self.b*2), Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.tr3 = TransitionBlock_2D(self.in_ch, Theta, Batchnorm, Bias)
        self.in_ch = int(self.in_ch * Theta)
        self.db4 = DenseBlock_2D(self.in_ch, Growth_rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias, 8)
        self.in_ch = int(self.in_ch + 8 * Growth_rate)
        self.up4 = nn.ConvTranspose2d(int(self.in_ch), self.c, kernel_size=4, stride=2, padding=1, bias=Bias)
        self.conv4 = convUp_2D(int(self.c*2), Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.out = conv_layer2_2D(k0, n_classes, Batchnorm, 1, 1, 0, Bias)
        self.ignore_class = ignore_class
        # self.sfmx = nn.Softmax(1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x1):
        #x1 = self.inc(x)
        x = self.down(x1)
        x2 = self.db1(x)
        x = self.tr1(x2)
        x3 = self.db2(x)
        x = self.tr2(x3)
        x4 = self.db3(x)
        x = self.tr3(x4)
        x = self.db4(x)
        x = self.up4(x)
        x = torch.cat([x4, x], dim=1)
        x = self.conv4(x)
        x = self.up3(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv3(x)
        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv2(x)
        x = self.up1(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv1(x)
        x_1 = self.out(x)
        if self.ignore_class > -1:
            s = list(x.shape)
            s[1] = 1
            x = torch.cat((x[:, :(self.ignore_class), :], torch.zeros(s), x[:, (self.ignore_class):, :]), dim=1)
        return x, x_1


class DenseUSegStack_2D(nn.Module):
    def __init__(self, n_channels, n_classes, ignore_class=-100, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=16, Kernel_size=3, Batchnorm=True, Stride=1, Padding=1, Bias=False):
        super(DenseUSegStack_2D, self).__init__()
        self.inc = inconv_2D(n_channels, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.D1 = DenseUSegDenoise2D(n_channels, 1, ignore_class, k0, Theta, Dropout, Growth_rate, Kernel_size, Batchnorm, Stride, Padding, Bias)
        self.c1a = conv_layer2_2D(k0, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.c1b = conv_layer2_2D(k0, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.c2b = conv_layer2_2D(1, k0, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.D2 = DenseUSeg2D(n_channels, n_classes, ignore_class, k0, Theta, Dropout, Growth_rate, Kernel_size, Batchnorm, Stride, Padding, Bias)

    def forward(self, x):
        x1a = self.inc(x)
        x, x_1 = self.D1(x1a)
        x = self.c1a(x)
        x = self.c1b(x)
        x1b = self.c2b(x_1)
        x = x1a + x1b + x
        x_2 = self.D2(x)
        return x_2, x_1