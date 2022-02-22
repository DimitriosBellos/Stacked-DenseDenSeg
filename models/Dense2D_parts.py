import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

class conv_layer_2D(nn.Module):
    '''(conv => BN => ReLU) * 2  TODO Residual and/or Atrous'''
    def __init__(self, in_ch, out_ch, Batchnorm=True, Kernel_size=3, Stride=1, Padding=1, Bias=True):
        super(conv_layer_2D, self).__init__()
        if Batchnorm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, Kernel_size, stride=Stride, padding=Padding, bias=Bias),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, Kernel_size, stride=Stride, padding=Padding, bias=Bias),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_layer2_2D(nn.Module):
    '''(conv => BN => ReLU) * 2  TODO Residual and/or Atrous'''
    def __init__(self, in_ch, out_ch, Batchnorm=True, Kernel_size=3, Stride=1, Padding=1, Bias=True):
        super(conv_layer2_2D, self).__init__()
        if Batchnorm:
            self.conv2 = nn.Sequential(
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, Kernel_size, stride=Stride, padding=Padding, bias=Bias)
            )
        else:
            self.conv2 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, Kernel_size, stride=Stride, padding=Padding, bias=Bias)
            )

    def forward(self, x):
        x = self.conv2(x)
        return x

class inconv_2D(nn.Module):
    def __init__(self, in_ch, out_ch, Batchnorm=True, Kernel_size=3, Stride=1, Padding=1, Bias=True):
        super(inconv_2D, self).__init__()
        self.conv = nn.Sequential(
            conv_layer_2D(in_ch, out_ch, Batchnorm, Kernel_size, Stride, Padding, Bias),
            conv_layer_2D(out_ch, out_ch, Batchnorm, Kernel_size, Stride, Padding, Bias),
            nn.Conv2d(out_ch, out_ch, Kernel_size, stride=Stride, padding=Padding, bias=Bias)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class DenseConv_2D(nn.Module):
    def __init__(self, in_ch, Groth_Rate=16, Dropout=0.2, Batchnorm=True, Kernel_size=3, Stride=1, Padding=1, Bias=True):
        super(DenseConv_2D, self).__init__()
        self.dconv = nn.Sequential(
            conv_layer2_2D(in_ch, 128, Batchnorm, 1, 1, 0, Bias),
            conv_layer2_2D(128, Groth_Rate, Batchnorm, Kernel_size, Stride, Padding, Bias),
            nn.Dropout2d(p=Dropout, inplace=True)
        )

    def forward(self, x):
        x = self.dconv(x)
        return x


class DenseBlock_2D(nn.Module):
    def __init__(self, in_ch, Groth_Rate=16, Dropout=0.2, Batchnorm=True, Kernel_size=3, Stride=1, Padding=1, Bias=True, Num_layers=4):
        super(DenseBlock_2D, self).__init__()
        for i in range(Num_layers):
            self.add_module(('layer_%d' % i), DenseConv_2D(in_ch, Groth_Rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias))
            in_ch += Groth_Rate

    def forward(self, res):
        for idx, c in enumerate(self.children()):
            if idx < self._modules.__len__():
                res = torch.cat((res, c(res)), dim=1)
        return res


class TransitionBlock_2D(nn.Module):
    def __init__(self, in_ch, theta, Batchnorm=True, Bias=True):
        super(TransitionBlock_2D, self).__init__()
        self.tbconv = nn.Sequential(
            conv_layer2_2D(in_ch, int(in_ch*theta), Batchnorm, 1, 1, 0, Bias),
            conv_layer2_2D(int(in_ch*theta), int(in_ch*theta), Batchnorm, 2, 2, 0, Bias)
        )

    def forward(self, x):
        x = self.tbconv(x)
        return x


class convUp_2D(nn.Module):
    def __init__(self, in_ch, Batchnorm=True, Kernel_size=3, Stride=1, Padding=1, Bias=True):
        super(convUp_2D, self).__init__()
        self.conv = nn.Sequential(
            conv_layer2_2D(in_ch, int(in_ch/2), Batchnorm, Kernel_size, Stride, Padding, Bias),
            conv_layer2_2D(int(in_ch/2), int(in_ch/2), Batchnorm, Kernel_size, Stride, Padding, Bias)
        )

    def forward(self, x):
        x = self.conv(x)
        return x