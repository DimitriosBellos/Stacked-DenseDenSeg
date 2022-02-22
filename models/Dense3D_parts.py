import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

class conv_layer(nn.Module):
    '''(conv => BN => ReLU) * 2  TODO Residual and/or Atrous'''
    def __init__(self, in_ch, out_ch, Batchnorm=True, Kernel_size=3, Stride=1, Padding=1, Bias=True):
        super(conv_layer, self).__init__()
        if Batchnorm:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, Kernel_size, stride=Stride, padding=Padding, bias=Bias),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, Kernel_size, stride=Stride, padding=Padding, bias=Bias),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_layer2(nn.Module):
    '''(conv => BN => ReLU) * 2  TODO Residual and/or Atrous'''
    def __init__(self, in_ch, out_ch, Batchnorm=True, Kernel_size=3, Stride=1, Padding=1, Bias=True):
        super(conv_layer2, self).__init__()
        if Batchnorm:
            self.conv2 = nn.Sequential(
                nn.BatchNorm3d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_ch, out_ch, Kernel_size, stride=Stride, padding=Padding, bias=Bias)
            )
        else:
            self.conv2 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv3d(in_ch, out_ch, Kernel_size, stride=Stride, padding=Padding, bias=Bias)
            )

    def forward(self, x):
        x = self.conv2(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, Batchnorm=True, Kernel_size=3, Stride=1, Padding=1, Bias=True):
        super(inconv, self).__init__()
        self.conv = nn.Sequential(
            conv_layer(in_ch, out_ch, Batchnorm, Kernel_size, Stride, Padding, Bias),
            conv_layer(out_ch, out_ch, Batchnorm, Kernel_size, Stride, Padding, Bias),
            nn.Conv3d(out_ch, out_ch, Kernel_size, stride=Stride, padding=Padding, bias=Bias)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class DenseConv(nn.Module):
    def __init__(self, in_ch, Groth_Rate=16, Dropout=0.2, Batchnorm=True, Kernel_size=3, Stride=1, Padding=1, Bias=True):
        super(DenseConv, self).__init__()
        self.dconv = nn.Sequential(
            conv_layer2(in_ch, 256, Batchnorm, 1, 1, 0, Bias),
            conv_layer2(256, Groth_Rate, Batchnorm, Kernel_size, Stride, Padding, Bias),
            nn.Dropout3d(p=Dropout, inplace=True)
        )

    def forward(self, x):
        x = self.dconv(x)
        return x


class convUp(nn.Module):
    def __init__(self, in_ch, Batchnorm=True, Kernel_size=3, Stride=1, Padding=1, Bias=True):
        super(convUp, self).__init__()
        self.conv = nn.Sequential(
            conv_layer2(in_ch, int(in_ch/2), Batchnorm, Kernel_size, Stride, Padding, Bias),
            conv_layer2(int(in_ch/2), int(in_ch/2), Batchnorm, Kernel_size, Stride, Padding, Bias)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_ch, Groth_Rate=16, Dropout=0.2, Batchnorm=True, Kernel_size=3, Stride=1, Padding=1, Bias=True):
        super(DenseBlock, self).__init__()
        self.in_ch=in_ch
        self.dbconv1 = DenseConv(self.in_ch, Groth_Rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch=self.in_ch+Groth_Rate
        self.dbconv2 = DenseConv(self.in_ch, Groth_Rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch=self.in_ch+Groth_Rate
        self.dbconv3 = DenseConv(self.in_ch, Groth_Rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch=self.in_ch+Groth_Rate
        self.dbconv4 = DenseConv(self.in_ch, Groth_Rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch=self.in_ch+Groth_Rate

    def forward(self, x):
        x1 = self.dbconv1(x)
        x1t = torch.cat([x1, x], dim=1)
        x2 = self.dbconv2(x1t)
        x2t = torch.cat([x2, x1, x], dim=1)
        x3 = self.dbconv3(x2t)
        x3t = torch.cat([x3, x2, x1, x], dim=1)
        x4 = self.dbconv4(x3t)
        x = torch.cat([x4, x3, x2, x1, x], dim=1)
        return x


class TransitionBlock(nn.Module):
    def __init__(self, in_ch, theta, Batchnorm=True, Bias=True):
        super(TransitionBlock, self).__init__()
        self.tbconv = nn.Sequential(
            conv_layer2(in_ch, int(in_ch*theta), Batchnorm, 1, 1, 0, Bias),
            conv_layer2(int(in_ch*theta), int(in_ch*theta), Batchnorm, 2, 2, 0, Bias)
        )

    def forward(self, x):
        x = self.tbconv(x)
        return x


class TransitionBlock2(nn.Module):
    def __init__(self, in_ch, theta, Batchnorm=True, Bias=True):
        super(TransitionBlock2, self).__init__()
        self.tbconv = nn.Sequential(
            conv_layer2(in_ch, int(in_ch*theta), Batchnorm, 1, 1, 0, Bias),
            conv_layer2(int(in_ch*theta), int(in_ch*theta), Batchnorm, (1, 2, 2), (1, 2, 2), 0, Bias)
        )

    def forward(self, x):
        x = self.tbconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, Batchnorm=True, Kernel_size=2, Stride=2, Padding=1, Bias=False):
        super(Up, self).__init__()
        if Batchnorm:
            self.up = nn.Sequential(
                nn.BatchNorm3d(in_ch),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(in_ch, out_ch, Kernel_size, stride=Stride, padding=Padding, bias=Bias)
            )
        else:
            self.up = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(in_ch, out_ch, Kernel_size, stride=Stride, padding=Padding, bias=Bias)
            )

    def forward(self, x):
        x = self.up(x)
        return x


class DenseUBlock(nn.Module):
    def __init__(self, in_ch, Groth_Rate=16, Dropout=0.2, theta=0.5, Batchnorm=True, Kernel_size=3, Stride=1, Padding=1, Bias=True):
        super(DenseUBlock, self).__init__()
        self.in_ch=in_ch
        self.dbconv1 = DenseConv(self.in_ch, Groth_Rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch=self.in_ch+Groth_Rate
        self.dbconv2 = DenseConv(self.in_ch, Groth_Rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch=self.in_ch+Groth_Rate
        self.dbconv3 = DenseConv(self.in_ch, Groth_Rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch=self.in_ch+Groth_Rate
        self.dbconv4 = DenseConv(self.in_ch, Groth_Rate, Dropout, Batchnorm, Kernel_size, Stride, Padding, Bias)
        self.in_ch=self.in_ch+Groth_Rate
        self.condense = conv_layer2(in_ch, int(in_ch*theta), Batchnorm, 1, 1, 0, Bias)

    def forward(self, x):
        x1 = self.dbconv1(x)
        x1t = torch.cat([x1, x], dim=1)
        x2 = self.dbconv2(x1t)
        x2t = torch.cat([x2, x1, x], dim=1)
        x3 = self.dbconv3(x2t)
        x3t = torch.cat([x3, x2, x1, x], dim=1)
        x4 = self.dbconv4(x3t)
        x = torch.cat([x4, x3, x2, x1, x], dim=1)
        x = self.condense(x)
        return x


class TransitionUBlock(nn.Module):
    def __init__(self, in_ch, Batchnorm=True, Bias=True):
        super(TransitionUBlock, self).__init__()
        self.trans = conv_layer2(int(in_ch), int(in_ch), Batchnorm, 2, 2, 0, Bias)

    def forward(self, x):
        x = self.trans(x)
        return x
