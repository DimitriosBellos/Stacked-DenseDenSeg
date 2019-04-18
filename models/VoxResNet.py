import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxResModule(nn.Module):
    def __init__(self):
        super(VoxResModule, self).__init__()
        self.bnorm1 = nn.BatchNorm3d(64)
        self.conv1 = nn.Conv3d(64, 64, 3, padding=1)
        self.bnorm2 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 64, 3, padding=1)

    def forward(self, x):
        h = F.relu(self.bnorm1(x))
        h = self.conv1(h)
        h = F.relu(self.bnorm2(h))
        h = self.conv2(h)
        return h + x


class VoxResNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=1):
        super(VoxResNet, self).__init__()
        self.conv1a = nn.Conv3d(in_channels, 32, 3, padding=1)
        self.bnorm1a = nn.BatchNorm3d(32)
        self.conv1b = nn.Conv3d(32, 32, 3, padding=1)
        self.bnorm1b = nn.BatchNorm3d(32)
        self.conv1c = nn.Conv3d(32, 64, 3, stride=2, padding=1)
        self.voxres2 = VoxResModule()
        self.voxres3 = VoxResModule()
        self.bnorm3 = nn.BatchNorm3d(64)
        self.conv4 = nn.Conv3d(64, 64, 3, stride=2, padding=1)
        self.voxres5 = VoxResModule()
        self.voxres6 = VoxResModule()
        self.bnorm6 = nn.BatchNorm3d(64)
        self.conv7 = nn.Conv3d(64, 64, 3, stride=2, padding=1)
        self.voxres8 = VoxResModule()
        self.voxres9 = VoxResModule()
        self.c1deconv = nn.ConvTranspose3d(32, 32, 3, padding=1)
        self.c1conv = nn.Conv3d(32, n_classes, 3, padding=1)
        self.c2deconv = nn.ConvTranspose3d(64, 64, 4, stride=2, padding=1)
        self.c2conv = nn.Conv3d(64, n_classes, 3, padding=1)
        self.c3deconv = nn.ConvTranspose3d(64, 64, 6, stride=4, padding=1)
        self.c3conv = nn.Conv3d(64, n_classes, 3, padding=1)
        self.c4deconv = nn.ConvTranspose3d(64, 64, 10, stride=8, padding=1)
        self.c4conv = nn.Conv3d(64, n_classes, 3, padding=1)
        self.out = nn.Conv3d(n_classes, n_classes, 1, padding=0, bias=False)

    def forward(self, x):
        h = self.conv1a(x)
        h = F.relu(self.bnorm1a(h))
        h = self.conv1b(h)
        c1 = F.relu(self.c1deconv(h))
        c1 = self.c1conv(c1)

        h = F.relu(self.bnorm1b(h))
        h = self.conv1c(h)
        h = self.voxres2(h)
        h = self.voxres3(h)
        c2 = F.relu(self.c2deconv(h))
        c2 = self.c2conv(c2)

        h = F.relu(self.bnorm3(h))
        h = self.conv4(h)
        h = self.voxres5(h)
        h = self.voxres6(h)
        c3 = F.relu(self.c3deconv(h))
        c3 = self.c3conv(c3)

        h = F.relu(self.bnorm6(h))
        h = self.conv7(h)
        h = self.voxres8(h)
        h = self.voxres9(h)
        c4 = F.relu(self.c4deconv(h))
        c4 = self.c4conv(c4)

        c = c1 + c2 + c3 + c4
        c = self.out(c)

        return c, c1, c2, c3, c4


class VoxResModule_2D(nn.Module):
    def __init__(self):
        super(VoxResModule_2D, self).__init__()
        self.bnorm1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bnorm2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

    def forward(self, x):
        h = F.relu(self.bnorm1(x))
        h = self.conv1(h)
        h = F.relu(self.bnorm2(h))
        h = self.conv2(h)
        return h + x


class VoxResNet_2D(nn.Module):
    def __init__(self, in_channels=11, n_classes=4):
        super(VoxResNet_2D, self).__init__()
        self.conv1a = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bnorm1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, 3, padding=1)
        self.bnorm1b = nn.BatchNorm2d(32)
        self.conv1c = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.voxres2 = VoxResModule_2D()
        self.voxres3 = VoxResModule_2D()
        self.bnorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.voxres5 = VoxResModule_2D()
        self.voxres6 = VoxResModule_2D()
        self.bnorm6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.voxres8 = VoxResModule_2D()
        self.voxres9 = VoxResModule_2D()
        self.c1deconv = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.c1conv = nn.Conv2d(32, n_classes, 3, padding=1)
        self.c2deconv = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.c2conv = nn.Conv2d(64, n_classes, 3, padding=1)
        self.c3deconv = nn.ConvTranspose2d(64, 64, 6, stride=4, padding=1)
        self.c3conv = nn.Conv2d(64, n_classes, 3, padding=1)
        self.c4deconv = nn.ConvTranspose2d(64, 64, 10, stride=8, padding=1)
        self.c4conv = nn.Conv2d(64, n_classes, 3, padding=1)

    def forward(self, x):
        h = self.conv1a(x)
        h = F.relu(self.bnorm1a(h))
        h = self.conv1b(h)
        c1 = F.relu(self.c1deconv(h))
        c1 = self.c1conv(c1)

        h = F.relu(self.bnorm1b(h))
        h = self.conv1c(h)
        h = self.voxres2(h)
        h = self.voxres3(h)
        c2 = F.relu(self.c2deconv(h))
        c2 = self.c2conv(c2)

        h = F.relu(self.bnorm3(h))
        h = self.conv4(h)
        h = self.voxres5(h)
        h = self.voxres6(h)
        c3 = F.relu(self.c3deconv(h))
        c3 = self.c3conv(c3)

        h = F.relu(self.bnorm6(h))
        h = self.conv7(h)
        h = self.voxres8(h)
        h = self.voxres9(h)
        c4 = F.relu(self.c4deconv(h))
        c4 = self.c4conv(c4)

        c = c1 + c2 + c3 + c4

        return c