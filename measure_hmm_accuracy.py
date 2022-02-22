import numpy as np
import h5py
import sys
import time
import multiprocessing as mp
from functools import partial
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.logger import LoggerLite
from os.path import isfile
import tqdm


class Confusion_Matrix(nn.Module):
    def __init__(self, n_classes):
        super(Confusion_Matrix, self).__init__()
        self.n_classes = n_classes

    def forward(self, yp, y, x2, res): # list(range(torch.cuda.device_count()))[-1]):
        for i in range(0, int(y.shape[0])):
            x = yp[i].flatten() * n_classes + y[i].flatten()
            x = x.unsqueeze(0)
            x = x.type(torch.cuda.LongTensor)
            x2[i] = x2[i].scatter_(0, x, 1)
            res[i] = torch.sum(x2[i], 1).reshape((n_classes, n_classes))
        return res


class Databringer(Dataset):
    def __init__(self, n_classes, time_size, slices, ground_truths, before_hmm, after_hmm, after_hmm2, start, cpus):
        super(Databringer, self).__init__()
        self.n_classes = n_classes
        self.time_size = time_size
        self.slices = slices
        self.ground_truths = ground_truths
        self.before_hmm = before_hmm
        self.after_hmm = after_hmm
        self.after_hmm2 = after_hmm2
        self.cpus = cpus
        self.start = start
        self.iteration = 0
        self.all_size = (self.slices[:, 0] * self.slices[:, 1] * self.slices[:, 2])
        self.b = self.all_size // self.cpus

    def give_iteration(self, iteration):
        self.iteration = iteration

    def __getitem__(self, idx):
        if idx != self.cpus - 1:
            # Sam = np.zeros((self.time_size, self.b[self.iteration]))
            Samy = np.zeros((self.time_size, self.b[self.iteration]))
            Samy2 = np.zeros((self.time_size, self.b[self.iteration]))
            Samy3 = np.zeros((self.time_size, self.b[self.iteration]))
            tmp = np.zeros((self.b[self.iteration],))

            f2 = h5py.File(self.ground_truths, 'r')
            data = f2.get(list(f2.keys())[0].encode('ascii', 'ignore'))
            while isinstance(data, h5py.Group):
                data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))
            for i in range(idx * self.b[self.iteration], (idx + 1) * self.b[self.iteration]):
                z = i // (self.slices[self.iteration, 1] * self.slices[self.iteration, 2])
                tr = i - z * self.slices[self.iteration, 1] * self.slices[self.iteration, 2]
                y = tr // self.slices[self.iteration, 2]
                x = tr - y * self.slices[self.iteration, 2]
                tmp[i - idx * self.b[self.iteration]] = data[int(z) + self.start[self.iteration, 0], int(y) + self.start[self.iteration, 1] + 450, int(x) + self.start[self.iteration, 2] + 400]
                tmp[tmp == 4] = 3
                #Sam[:, i - idx * self.b[self.iteration]] = data[:, int(z) + self.start[self.iteration, 0], int(y) + self.start[self.iteration, 1], int(x) + self.start[self.iteration, 2]]
            f2.close()
            #Sam = Sam.flatten()
            Sam = np.repeat(tmp[np.newaxis, :], self.time_size, axis=0).flatten()

            del tmp

            f2 = h5py.File(self.before_hmm, 'r')
            data = f2.get(list(f2.keys())[0].encode('ascii', 'ignore'))
            while isinstance(data, h5py.Group):
                data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))
            for i in range(idx * self.b[self.iteration], (idx + 1) * self.b[self.iteration]):
                z = i // (self.slices[self.iteration, 1] * self.slices[self.iteration, 2])
                tr = i - z * self.slices[self.iteration, 1] * self.slices[self.iteration, 2]
                y = tr // self.slices[self.iteration, 2]
                x = tr - y * self.slices[self.iteration, 2]
                Samy[:, i - idx * self.b[self.iteration]] = data[:, int(z) + self.start[self.iteration, 0], int(y) + self.start[self.iteration, 1], int(x) + self.start[self.iteration, 2]]
            f2.close()
            Samy = Samy.flatten()

            f2 = h5py.File(self.after_hmm, 'r')
            data = f2.get(list(f2.keys())[0].encode('ascii', 'ignore'))
            while isinstance(data, h5py.Group):
                data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))
            for i in range(idx * self.b[self.iteration], (idx + 1) * self.b[self.iteration]):
                z = i // (self.slices[self.iteration, 1] * self.slices[self.iteration, 2])
                tr = i - z * self.slices[self.iteration, 1] * self.slices[self.iteration, 2]
                y = tr // self.slices[self.iteration, 2]
                x = tr - y * self.slices[self.iteration, 2]
                Samy2[:, i - idx * self.b[self.iteration]] = data[:, int(z) + self.start[self.iteration, 0], int(y) + self.start[self.iteration, 1], int(x) + self.start[self.iteration, 2]]
            f2.close()
            Samy2 = Samy2.flatten()

            f2 = h5py.File(self.after_hmm2, 'r')
            data = f2.get(list(f2.keys())[0].encode('ascii', 'ignore'))
            while isinstance(data, h5py.Group):
                data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))
            for i in range(idx * self.b[self.iteration], (idx + 1) * self.b[self.iteration]):
                z = i // (self.slices[self.iteration, 1] * self.slices[self.iteration, 2])
                tr = i - z * self.slices[self.iteration, 1] * self.slices[self.iteration, 2]
                y = tr // self.slices[self.iteration, 2]
                x = tr - y * self.slices[self.iteration, 2]
                Samy3[:, i - idx * self.b[self.iteration]] = data[:, int(z) + self.start[self.iteration, 0], int(y) + self.start[self.iteration, 1], int(x) + self.start[self.iteration, 2]]
            f2.close()
            Samy3 = Samy3.flatten()

        else:
            sz = self.slices[self.iteration, 0] * self.slices[self.iteration, 1] * self.slices[self.iteration, 2]
            ss = sz - idx * self.b[self.iteration]
            Sam = np.zeros((self.time_size, ss))
            Samy = np.zeros((self.time_size, ss))
            Samy2 = np.zeros((self.time_size, ss))
            Samy3 = np.zeros((self.time_size, ss))
            tmp = np.zeros((ss,))

            f2 = h5py.File(self.ground_truths, 'r')
            data = f2.get(list(f2.keys())[0].encode('ascii', 'ignore'))
            while isinstance(data, h5py.Group):
                data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))
            for i in range(idx * self.b[self.iteration], sz):
                z = i // (self.slices[self.iteration, 1] * self.slices[self.iteration, 2])
                tr = i - z * self.slices[self.iteration, 1] * self.slices[self.iteration, 2]
                y = tr // self.slices[self.iteration, 2]
                x = tr - y * self.slices[self.iteration, 2]
                tmp[i - idx * self.b[self.iteration]] = data[int(z) + self.start[self.iteration, 0], int(y) + self.start[self.iteration, 1] + 450, int(x) + self.start[self.iteration, 2] + 400]
                tmp[tmp == 4] = 3
                #Sam[:, i - idx * self.b[self.iteration]] = data[:, int(z) + self.start[self.iteration, 0], int(y) + self.start[self.iteration, 1], int(x) + self.start[self.iteration, 2]]
            f2.close()
            Sam = np.repeat(tmp[np.newaxis, :], self.time_size, axis=0).flatten()
            #Sam = Sam.flatten()

            del tmp

            f2 = h5py.File(self.before_hmm, 'r')
            data = f2.get(list(f2.keys())[0].encode('ascii', 'ignore'))
            while isinstance(data, h5py.Group):
                data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))
            for i in range(idx * self.b[self.iteration], ss):
                z = i // (self.slices[self.iteration, 1] * self.slices[self.iteration, 2])
                tr = i - z * self.slices[self.iteration, 1] * self.slices[self.iteration, 2]
                y = tr // self.slices[self.iteration, 2]
                x = tr - y * self.slices[self.iteration, 2]
                Samy[:, i - idx * self.b[self.iteration]] = data[:, int(z) + self.start[self.iteration, 0], int(y) + self.start[self.iteration, 1], int(x) + self.start[self.iteration, 2]]
            f2.close()
            Samy = Samy.flatten()

            f2 = h5py.File(self.after_hmm, 'r')
            data = f2.get(list(f2.keys())[0].encode('ascii', 'ignore'))
            while isinstance(data, h5py.Group):
                data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))
            for i in range(idx * self.b[self.iteration], sz):
                z = i // (self.slices[self.iteration, 1] * self.slices[self.iteration, 2])
                tr = i - z * self.slices[self.iteration, 1] * self.slices[self.iteration, 2]
                y = tr // self.slices[self.iteration, 2]
                x = tr - y * self.slices[self.iteration, 2]
                Samy2[:, i - idx * self.b[self.iteration]] = data[:, int(z) + self.start[self.iteration, 0], int(y) + self.start[self.iteration, 1], int(x) + self.start[self.iteration, 2]]
            f2.close()
            Samy2 = Samy2.flatten()

            f2 = h5py.File(self.after_hmm2, 'r')
            data = f2.get(list(f2.keys())[0].encode('ascii', 'ignore'))
            while isinstance(data, h5py.Group):
                data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))
            for i in range(idx * self.b[self.iteration], sz):
                z = i // (self.slices[self.iteration, 1] * self.slices[self.iteration, 2])
                tr = i - z * self.slices[self.iteration, 1] * self.slices[self.iteration, 2]
                y = tr // self.slices[self.iteration, 2]
                x = tr - y * self.slices[self.iteration, 2]
                Samy3[:, i - idx * self.b[self.iteration]] = data[:, int(z) + self.start[self.iteration, 0], int(y) + self.start[self.iteration, 1], int(x) + self.start[self.iteration, 2]]
            f2.close()
            Samy3 = Samy3.flatten()

        return torch.from_numpy(Sam), torch.from_numpy(Samy), torch.from_numpy(Samy2), torch.from_numpy(Samy3)

    def __len__(self):
        return self.cpus


def calc_conf(tcm, n_classes):
    accuracy = np.zeros((n_classes + 1),)
    for i in range(0, n_classes):
        if tcm[:, i].sum() + tcm[i, :].sum() != 0:
            accuracy[i] = (tcm[i, i] / (tcm[:, i].sum() + tcm[i, :].sum() - tcm[i, i]))
        else:
            accuracy[i] = 1
    accuracy[n_classes] = accuracy[:n_classes].mean()
    return accuracy


if __name__ == '__main__':
    ground_truths = '/db/user/db1/annotations/annotations0.h5'
    before_hmm = '/db/user/4D/original.h5'
    after_hmm = '/db/user/4D/hmm1_5.h5'
    after_hmm2 = '/db/user/4D/hmm2_5.h5'
    time_size = 21
    start = np.mgrid[1660:1661, 806:807, 50:51]  # 1692:1702, 806:807, 50:51]  # 1702
    start = np.rollaxis(start, 0, 4)
    start = start.reshape(1, 3)
    end = np.mgrid[1670:1671, 1414:1415, 562:563]  # 1693:1703, 1414:1415, 562:563]  # 1703
    end = np.rollaxis(end, 0, 4)
    end = end.reshape(1, 3)
    iterations = 1 # start.shape[0]
    slices = end - start
    cpus = 32
    gpus = 4
    n_classes = 4

    val = LoggerLite('/home/user/Code_Folder/validationshmm_new', 'w')
    t = ('Methods',)
    for l in range(0, n_classes + 1):
        if np.mod(l + 1, n_classes + 1) == 0:
            r = 'mIoU'
        else:
            r = 'IoU_%d' % l
        t = t + (r,)
    val.setNames(t)

    cms = LoggerLite('/home/user/Code_Folder/cmshmm_new', 'w')
    t = ('Methods',)
    for k in range(0, n_classes):
        for l in range(0, n_classes):
            if l < k:
                r = 'F_%d%d' % (k, l)
            elif l == k:
                r = 'T_%d' % k
            else:
                r = 'F_%d%d' % (k, l)
            t = t + (r,)
    cms.setNames(t)

    cm1 = np.zeros((n_classes, n_classes)).astype(np.ulonglong)
    cm2 = np.zeros((n_classes, n_classes)).astype(np.ulonglong)
    cm3 = np.zeros((n_classes, n_classes)).astype(np.ulonglong)

    data = Databringer(n_classes, time_size, slices, ground_truths, before_hmm, after_hmm, after_hmm2, start, int(cpus*2))
    conf = Confusion_Matrix(n_classes=n_classes)
    conf = torch.nn.DataParallel(conf, device_ids=list(range(torch.cuda.device_count()))[:4], output_device=list(range(torch.cuda.device_count()))[0])
    with torch.no_grad():
        for k in range(0, iterations):
            sys.stdout.write('Iteration %d\n' % k)
            data.give_iteration(k)
            loader = iter(DataLoader(dataset=data, batch_size=cpus, num_workers=cpus))
            for j in range(0, 2):
                Sam, Samy, Samy2, Samy3 = next(loader)
                x2 = torch.LongTensor(cpus, n_classes * n_classes, int(sum(list(Sam.shape[1:])))).zero_()
                res = torch.LongTensor(cpus, n_classes, n_classes).zero_()
                Sam = Sam.cuda()
                Samy = Samy.cuda()
                Samy2 = Samy2.cuda()
                Samy3 = Samy3.cuda()
                x2 = x2.cuda()
                res = res.cuda()

                tcm1 = conf(Samy, Sam, x2, res)

                x2 = torch.LongTensor(cpus, n_classes * n_classes, int(sum(list(Sam.shape[1:])))).zero_()
                res = torch.LongTensor(cpus, n_classes, n_classes).zero_()
                x2 = x2.cuda()
                res = res.cuda()

                tcm2 = conf(Samy2, Sam, x2, res)

                x2 = torch.LongTensor(cpus, n_classes * n_classes, int(sum(list(Sam.shape[1:])))).zero_()
                res = torch.LongTensor(cpus, n_classes, n_classes).zero_()
                x2 = x2.cuda()
                res = res.cuda()

                tcm3 = conf(Samy3, Sam, x2, res)

                cm1 += torch.sum(tcm1, 0).cpu().detach().numpy().astype(np.ulonglong)
                cm2 += torch.sum(tcm2, 0).cpu().detach().numpy().astype(np.ulonglong)
                cm3 += torch.sum(tcm3, 0).cpu().detach().numpy().astype(np.ulonglong)
            sys.stdout.write('\n')

    cms.add(['Before'] + list(cm1.flatten()))
    cms.add(['After'] + list(cm2.flatten()))
    cms.add(['After2'] + list(cm3.flatten()))
    acc1 = calc_conf(cm1, n_classes)
    acc2 = calc_conf(cm2, n_classes)
    acc3 = calc_conf(cm3, n_classes)
    val.add(['Before'] + list(acc1))
    val.add(['After'] + list(acc2))
    val.add(['After2'] + list(acc3))
    print('Done')
