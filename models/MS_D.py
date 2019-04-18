import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MSD(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, growth_rate, ignore_class=-100, kernel_size=3):
        super(MSD, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.ignore_class = ignore_class
        current_channels = in_channels
        for i in range(num_layers):
            dilation=(i+1)%10+1
            dilated_kernel_size = (kernel_size - 1) * dilation + 1
            padding = dilated_kernel_size // 2
            conv = nn.Sequential(
                nn.Conv2d(current_channels, growth_rate, kernel_size, dilation=dilation, padding=padding),
                nn.ReLU(inplace=True))
            self.add_module(('layer_%d'%i), conv)
            current_channels += growth_rate
        conv = nn.Sequential(
            nn.Conv2d(current_channels, out_channels, kernel_size=1))
            # nn.Softmax(dim=1))
        self.add_module('last', conv)
        n = kernel_size*kernel_size*(in_channels+growth_rate*(num_layers-1))+out_channels
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, res):
        for idx, c in enumerate(self.children()):
            if idx<self.num_layers:
                res = torch.cat((res, c(res)), dim=1)
            else:
                res = c(res)
        if self.ignore_class > -1:
            s = list(res.shape)
            s[1] = 1
            if res.is_cuda:
                res = torch.cat((res[:, :(self.ignore_class), :], torch.zeros(s).cuda(), res[:, (self.ignore_class):, :]), dim=1)
            else:
                res = torch.cat((res[:, :(self.ignore_class), :], torch.zeros(s), res[:, (self.ignore_class):, :]), dim=1)
        return res


class MSDrop(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, growth_rate, ignore_class=-100, kernel_size=3):
        super(MSDrop, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.ignore_class = ignore_class
        current_channels = in_channels
        for i in range(num_layers):
            dilation=(i+1)%10+1
            dilated_kernel_size = (kernel_size - 1) * dilation + 1
            padding = dilated_kernel_size // 2
            conv = nn.Sequential(
                nn.Conv2d(current_channels, growth_rate, kernel_size, dilation=dilation, padding=padding),
                nn.ReLU(inplace=False),
                nn.Dropout2d(p=0.2, inplace=True))
            self.add_module(('layer_%d'%i), conv)
            current_channels += growth_rate
        conv = nn.Sequential(
            nn.Conv2d(current_channels, out_channels, kernel_size=1))
            # nn.Softmax(dim=1))
        self.add_module('last', conv)
        n = kernel_size*kernel_size*(in_channels+growth_rate*(num_layers-1))+out_channels
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, res):
        for idx, c in enumerate(self.children()):
            if idx<self.num_layers:
                res = torch.cat((res, c(res)), dim=1)
            else:
                res = c(res)
        if self.ignore_class > -1:
            s = list(res.shape)
            s[1] = 1
            if res.is_cuda:
                res = torch.cat((res[:, :(self.ignore_class), :], torch.zeros(s).cuda(), res[:, (self.ignore_class):, :]), dim=1)
            else:
                res = torch.cat((res[:, :(self.ignore_class), :], torch.zeros(s), res[:, (self.ignore_class):, :]), dim=1)
        return res


class MSDenoise(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, growth_rate, ignore_class=-100, kernel_size=3):
        super(MSDenoise, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.ignore_class = ignore_class
        current_channels = in_channels
        for i in range(num_layers):
            dilation=(i+1)%10+1
            dilated_kernel_size = (kernel_size - 1) * dilation + 1
            padding = dilated_kernel_size // 2
            conv = nn.Sequential(
                nn.Conv2d(current_channels, growth_rate, kernel_size, dilation=dilation, padding=padding),
                nn.ReLU(inplace=True))
            self.add_module(('layer_%d'%i), conv)
            current_channels += growth_rate
        conv = nn.Sequential(
            nn.Conv2d(current_channels, out_channels, kernel_size=1))
            # nn.Softmax(dim=1))
        self.add_module('last', conv)
        n = kernel_size*kernel_size*(in_channels+growth_rate*(num_layers-1))+out_channels
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, res):
        for idx, c in enumerate(self.children()):
            if idx<self.num_layers:
                res = torch.cat((res, c(res)), dim=1)
            else:
                out = c(res)
        return res, out
