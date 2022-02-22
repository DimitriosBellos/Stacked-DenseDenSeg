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

if __name__ == '__main__':
    # test=torch.randn(4,4,40,40).cuda()
    # test2=torch.LongTensor(4,40,40).random_(4).cuda()
    test = torch.from_numpy(np.array([[[2, 1, 0], [0, 3, 2], [1, 0, 0]], [[2, 1, 0], [0, 3, 2], [1, 0, 0]]])).cuda()
    test2 = torch.from_numpy(np.array([[[2, 1, 0], [0, 3, 2], [1, 2, 0]], [[3, 1, 0], [0, 3, 2], [1, 0, 1]]])).cuda()

    sm = nn.Softmax(dim=1)

    accuracy, c_matrix = meas_net(test, test2, sm, 4)

    print(accuracy)
