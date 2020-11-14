import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import confusion_matrix
import math
import numbers
from multiprocessing import Array, Process


def meas_net(y_pred, y, mask=-100, gpu=True):
    y_pred = F.softmax(y_pred, dim=1)
    if gpu:
        y_pred = (torch.from_numpy(np.argmax(y_pred.cpu().detach().numpy(), axis=1))).cuda()
        if y.dim() > 3 and y.shape[1] == 1:
            y = (torch.from_numpy(np.argmax(y.cpu().detach().numpy(), axis=1))).cuda()
        if mask > 0:
            masked_sum = int((y_pred == mask).sum().cpu())
            sim = int((y_pred == y).sum().cpu())
            acc = sim / float(y.numel() - masked_sum)
        else:
            sim = int((y_pred == y).sum().cpu())
            acc = sim / float(y.numel())
    else:
        y_pred = torch.from_numpy(np.argmax(y_pred.detach().numpy(), axis=1))
        if y.dim() > 3 and y.shape[1] == 1:
            y = torch.from_numpy(np.argmax(y.detach().numpy(), axis=1))
        if mask > 0:
            masked_sum = int((y_pred == mask).sum())
            sim = int((y_pred == y).sum())
            acc = sim / float(y.numel() - masked_sum)
        else:
            sim = int((y_pred == y).sum())
            acc = sim / float(y.numel())
    return acc


def meas_cm(y_pred, y, n_cls, mask=-100, gpu=True):
    y_pred = F.softmax(y_pred, dim=1)
    if gpu:
        y_pred = (torch.from_numpy(np.argmax(y_pred.cpu().detach().numpy(), axis=1))).cuda()
        if y.dim() > 3 and y.shape[1] == 1:
            y = (torch.from_numpy(np.argmax(y.cpu().detach().numpy(), axis=1))).cuda()
        if mask > 0:
            masked_sum = int((y_pred == mask).sum().cpu())
            sim = int((y_pred == y).sum().cpu())
            acc = sim / float(y.numel() - masked_sum)
        else:
            sim = int((y_pred == y).sum().cpu())
            acc = sim / float(y.numel())
    else:
        y_pred = torch.from_numpy(np.argmax(y_pred.detach().numpy(), axis=1))
        if y.dim() > 3 and y.shape[1] == 1:
            y = torch.from_numpy(np.argmax(y.detach().numpy(), axis=1))
        if mask > 0:
            masked_sum = int((y_pred == mask).sum())
            sim = int((y_pred == y).sum())
            acc = sim / float(y.numel() - masked_sum)
        else:
            sim = int((y_pred == y).sum())
            acc = sim / float(y.numel())
    cm = []
    for i in range(0, y_pred.shape[0]):
        if gpu:
            if i == 0:
                cm = confusion_matrix(y[i, :, :].cpu().detach().numpy().flatten(), y_pred[i, :, :].cpu().detach().numpy().flatten(), labels=list(np.arange(n_cls)))
            else:
                cm += confusion_matrix(y[i, :, :].cpu().detach().numpy().flatten(), y_pred[i, :, :].cpu().detach().numpy().flatten(), labels=list(np.arange(n_cls)))
        else:
            if i == 0:
                cm = confusion_matrix(y[i, :, :].detach().numpy().flatten(), y_pred[i, :, :].detach().numpy().flatten(), labels=list(np.arange(n_cls)))
            else:
                cm += confusion_matrix(y[i, :, :].detach().numpy().flatten(), y_pred[i, :, :].detach().numpy().flatten(), labels=list(np.arange(n_cls)))
    # del sim, y_pred, y, sm, options
    return acc, cm


def meas_net_weighted(y_pred, y, n_cls, gpu=True):
    y_pred = F.softmax(y_pred, dim=1)
    if gpu:
        y_pred = (torch.from_numpy(np.argmax(y_pred.cpu().detach().numpy(), axis=1))).cuda()
        if y.dim() > 3 and y.shape[1] == 1:
            y = (torch.from_numpy(np.argmax(y.cpu().detach().numpy(), axis=1))).cuda()
        size = list(y_pred.shape)
        size = size[:1] + [n_cls] + size[1:]
        y_ex = torch.zeros(size).cuda()
        y_pred_ex = torch.zeros(size).cuda()
        # Next Hot One Encryption
        for i in range(0, n_cls):
            y_ex[:, i, :] = y == i
            y_pred_ex[:, i, :] = y_pred == i
        dif = y_pred_ex * y_ex
        while dif.dim() > 2:
            dif = torch.sum(dif, 2)
            y_ex = torch.sum(y_ex, 2)
        dif = torch.sum(dif, 0)
        y_ex = torch.sum(y_ex, 0)
        acc = (dif.double() / (y_ex.double())).cpu().detach().numpy()
        tot = dif.sum().cpu().detach().numpy() / float(y.numel())
        acc = np.hstack((acc, tot.reshape(1, )))
        # acc = acc.reshape(acc.shape[1],)
    else:
        y_pred = torch.from_numpy(np.argmax(y_pred.detach().numpy(), axis=1))
        if y.dim() > 3 and y.shape[1] == 1:
            y = torch.from_numpy(np.argmax(y.detach().numpy(), axis=1))
        size = list(y_pred.shape)
        size = size[:1] + [n_cls] + size[1:]
        y_ex = torch.zeros(size)
        y_pred_ex = torch.zeros(size)
        for i in range(0, n_cls):
            y_ex[:, i, :] = y == i
            y_pred_ex[:, i, :] = y_pred == i
        dif = y_pred_ex * y_ex
        while dif.dim() > 2:
            dif = torch.sum(dif, 2)
            y_ex = torch.sum(y_ex, 2)
        acc = (dif.double() / y_ex.double()).detach().numpy()
        tot = dif.sum().detach().numpy() / float(y.numel())
        acc = np.hstack((acc, tot.reshape(1, )))
        # acc = acc.reshape(acc.shape[1],)
    return acc


def meas_cm_weighted2(y_pred, y, n_cls, gpu=True):
    y_pred = F.softmax(y_pred, dim=1)
    if gpu:
        y_pred = np.argmax(y_pred.cpu().detach().numpy(), axis=1)                            
    else:
        y_pred = torch.from_numpy(np.argmax(y_pred.detach().numpy(), axis=1))
    if y.dim() > 3 and y.shape[1] == 1:
        y = torch.from_numpy(np.argmax(y.detach().numpy(), axis=1))
    else:
        y = y.detach().numpy()
    cm = confusion_matrix(y.flatten(), y_pred.flatten(), labels=list(np.arange(n_cls)))
    return cm


def meas_cm_weighted(yp, y, n_classes, gpu=True, device=0, softmax=True, threshold=True): # list(range(torch.cuda.device_count()))[-1]):
    if gpu:
        y=y.cuda(device=device)
        yp=yp.cuda(device=device)
    if softmax:
        yp = F.softmax(yp, dim=1)
    if threshold: 
        _, yp = yp.max(1)
    x = yp.flatten() * n_classes + y.flatten()
    x = x.unsqueeze(0)
    x2 = torch.LongTensor(n_classes * n_classes, x.shape[1]).zero_()
    if gpu:
        x2 = x2.cuda(device=device)
    x = x.type(torch.cuda.LongTensor)
    x2.scatter_(0, x, 1)
    res = torch.sum(x2, 1).reshape((n_classes, n_classes))
    return res

def meas_cm_weighted_hmm(yp, y, n_classes, gpu=True, device=0, softmax=True): # list(range(torch.cuda.device_count()))[-1], binn=3):
    y=y.cuda(device=device)
    yp=yp.cuda(device=device)
    if softmax:
        yp = F.softmax(yp, dim=1)
    pr = torch.max(yp, dim=1)
    ypp=pr[0].flatten()
    yp=pr[1].flatten()
    y=y.flatten()
    ypp = ypp*n_classes*binn/(n_classes-1) - binn/(n_classes-1)-1
    ypp = torch.ceil(ypp)
    ypp = binn * yp + ypp
    x = y * n_classes * binn + ypp
    x = x.unsqueeze(0)
    x2 = torch.LongTensor(n_classes * n_classes * binn, x.shape[1]).zero_()
    if gpu:
        x2 = x2.cuda(device=device)
    x = x.type(torch.cuda.LongTensor)
    x2 = x2.scatter_(0, x, 1)
    res = torch.sum(x2, 1).reshape(n_classes, n_classes*binn)
    return res

'''
def meas_cm_weighted_fast(y_pred, y, sm, n_cls, gpu=True):

    def par(d, cm):
        for j in d:
            cm[j[0]*j[1]] += 1

    y_pred = sm(y_pred)
    cm = np.zeros((n_cls,n_cls))
    if gpu:
        y_pred = np.argmax(y_pred.cpu().detach().numpy(), axis=1).flatten()
        if y.dim()>3 and y.shape[1]==1:
            y = np.argmax(y.cpu().detach().numpy(), axis=1).flatten()
    else:
        y_pred = np.argmax(y_pred.detach().numpy(), axis=1).flatten()
        if y.dim() > 3 and y.shape[1] == 1:
            y = np.argmax(y.detach().numpy(), axis=1).flatten()
    data=list([(y[i],y_pred[i]) for i in range(0,len(y))])
    cm = Array('i', n_cls*n_cls)
    p = Process(target=par, args=)
    return cm
'''

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        self.kernel_size = kernel_size
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.weight = kernel
        self.groups = channels

        if dim == 1:
            self.conv = nn.Conv1d(self.groups, self.groups, self.kernel_size, stride=1, padding=self.kernel_size // 2, bias=False)
        elif dim == 2:
            self.conv = nn.Conv2d(self.groups, self.groups, self.kernel_size, stride=1, padding=self.kernel_size // 2, bias=False)
        elif dim == 3:
            self.conv = nn.Conv3d(self.groups, self.groups, self.kernel_size, stride=1, padding=self.kernel_size // 2, bias=False)
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )
        self.conv.weight.data = self.weight

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input)


class SSIM(object):
    def __init__(self, window_size=11, data_range=16.693302):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.dimensions = 3
        self.data_range = data_range
        self.C1 = (0.01 ** 2) * (data_range ** 2)
        self.C2 = (0.03 ** 2) * (data_range ** 2)
        self.gaussian = GaussianSmoothing(self.channel, self.window_size, 1.5, dim=self.dimensions)
        self.gaussian.cuda()

    def _ssim(self, img1, img2):
        with torch.no_grad():
            mu1 = self.gaussian(img1)
            mu2 = self.gaussian(img2)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = self.gaussian(img1 * img1) - mu1_sq
            sigma2_sq = self.gaussian(img2 * img2) - mu2_sq
            sigma12 = self.gaussian(img1 * img2) - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

            ssim_map = ssim_map.mean()

        return ssim_map

    def __call__(self, img1, img2, data_range=16.693302):
        dimensions = img1.dim()-2
        channel = img1.size(1)

        if data_range != self.data_range:
            self.data_range = data_range
            self.C1 = (0.01 ** 2) * (data_range ** 2)
            self.C2 = (0.03 ** 2) * (data_range ** 2)

        if channel != self.channel or self.gaussian.conv.weight.data.type() != img1.data.type() or dimensions != self.dimensions:
            print('Problem')
            self.gaussian = GaussianSmoothing(channel, self.window_size, 1.5, dim=dimensions)

            if img1.is_cuda:
                self.gaussian.cuda()

            self.channel = channel
            self.dimensions = dimensions

        return self._ssim(img1, img2)

if __name__ == '__main__':
    # test=torch.randn(4,4,40,40).cuda()
    # test2=torch.LongTensor(4,40,40).random_(4).cuda()
    test = torch.from_numpy(np.array([[[2, 1, 0], [0, 3, 2], [1, 0, 0]], [[2, 1, 0], [0, 3, 2], [1, 0, 0]]])).cuda()
    test2 = torch.from_numpy(np.array([[[2, 1, 0], [0, 3, 2], [1, 2, 0]], [[3, 1, 0], [0, 3, 2], [1, 0, 1]]])).cuda()

    sm = nn.Softmax(dim=1)

    accuracy, c_matrix = meas_net(test, test2, sm, 4)

    print(accuracy)
