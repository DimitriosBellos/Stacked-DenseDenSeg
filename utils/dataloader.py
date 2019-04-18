#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

from os import listdir
from os.path import isfile, join
import sys
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import collections
import scipy.misc as m
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
import ctypes
import tqdm
from functools import partial
from scipy.signal import find_peaks
from .lcn import LCN_Jarret, LCN_Pinto

# import opts
from PIL import Image
from scipy.ndimage import rotate

dataAnnotation = []
dataInput = []


class DataTemplate(Dataset):
    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __init__(self, options, mode):
        super(DataTemplate, self).__init__()
        self.input_size = options.input_size
        self.output_area = options.output_area
        self.criterion = options.dice
        self.mode = mode
        self.prunned_nodes = options.prunned_classes
        self.filenameI = options.input_filename
        self.filenameA = options.annotations_filename
        self.normalize = options.normalize
        fileI = h5py.File(self.filenameI, 'r')
        dataI = fileI.get(list(fileI.keys())[0].encode('ascii', 'ignore'))
        while isinstance(dataI, h5py.Group):
            dataI = dataI.get(list(dataI.keys())[0].encode('ascii', 'ignore'))
        self.data_size = options.input_area[1, :] - options.input_area[0, :]
        self.excess = np.array([np.mod((self.data_size[0] - self.input_size[0]), options.input_stride[0]),
                                np.mod((self.data_size[1] - self.input_size[1]), options.input_stride[1]),
                                np.mod((self.data_size[2] - self.input_size[2]), options.input_stride[2])])
        self.start_points = np.floor(self.excess / 2).astype(int) + options.input_area[0, :]
        self.numel = np.array([((self.data_size[0] - self.input_size[0]) // options.input_stride[0]) + 1,
                               ((self.data_size[1] - self.input_size[1]) // options.input_stride[1]) + 1,
                               ((self.data_size[2] - self.input_size[2]) // options.input_stride[2]) + 1])
        self.pad = dataI[0, 0, 0]
        fileI.close()
        save_flag = True
        weights = np.ones((3,))
        rotstate = []
        perm = []
        order = []
        self.dataAnnotation = []
        self.dataInput = []
        if isfile(options.cp_dest + '2D_slices.npz') and options.clean_go is False:
            files = np.load(options.cp_dest + '2D_slices.npz')
            input_size = files['input_size']
            input_area = files['input_area']
            output_area = files['output_area']
            input_stride = files['input_stride']
            if (np.array_equal(input_size, options.input_size) and
                    np.array_equal(input_area, options.input_area) and
                    np.array_equal(input_stride, options.input_stride)):
                order = files['order']
                perm = files['perm']
                rotstate = files['rotstate']
                if options.weights[0] == -1:
                    weights = files['weights']
                elif options.weights[0] == -2:
                    weights = self.recalc(options, order)
                else:
                    weights = options.weights
                files.close()
                save_flag = False
        if save_flag:
            order, perm, rotstate, weights = self.processing(options)
        self.weights = torch.from_numpy(weights).float()
        self.num_train = (np.floor((1 - options.val_precentage - options.test_precentage) * len(perm))).astype(int)
        self.num_val = (np.floor(options.val_precentage * len(perm))).astype(int)
        self.num_test = len(perm) - self.num_val - self.num_train
        permTrain = perm[0:self.num_train]
        permVal = perm[self.num_train:(self.num_train + self.num_val)]
        permTest = perm[(self.num_train + self.num_val):len(perm)]
        # perm = permTrain[0:(len(permTrain)-np.mod(len(permTrain), options.batchsize).astype(int))]
        self.batchTrain = np.empty([self.num_train.astype(int), 4])
        self.batchTrain[:] = np.nan
        self.batchVal = np.empty([self.num_val.astype(int), 4])
        self.batchVal[:] = np.nan
        self.batchTest = np.empty([self.num_test.astype(int), 4])
        self.batchTest[:] = np.nan
        for i in range(0, self.num_train):
            self.batchTrain[i, 3] = rotstate[i]
            self.batchTrain[i, 2] = order[permTrain[i], 2]
            self.batchTrain[i, 1] = order[permTrain[i], 1]
            self.batchTrain[i, 0] = order[permTrain[i], 0]
        for i in range(0, self.num_val):
            self.batchVal[i, 3] = rotstate[i + self.num_train]
            self.batchVal[i, 2] = order[permVal[i], 2]
            self.batchVal[i, 1] = order[permVal[i], 1]
            self.batchVal[i, 0] = order[permVal[i], 0]
        for i in range(0, self.num_test):
            self.batchTest[i, 3] = rotstate[i + self.num_train + self.num_val]
            self.batchTest[i, 2] = order[permTest[i], 2]
            self.batchTest[i, 1] = order[permTest[i], 1]
            self.batchTest[i, 0] = order[permTest[i], 0]
        if save_flag:
            np.savez(options.cp_dest + '2D_slices.npz',
                     order=order,
                     perm=perm,
                     rotstate=rotstate,
                     weights=weights,
                     input_size=options.input_size,
                     input_area=options.input_area,
                     output_area=options.output_area,
                     input_stride=options.input_stride)

    def processing(self, options):
        order, weights = self.preprocess(options)
        perm = np.random.permutation(order.shape[0])
        rotstate = np.random.randint(8, size=order.shape[0])
        return order, perm, rotstate, weights

    def change_mode(self, mode):
        self.mode = mode
        return

    @staticmethod
    def progress(count, total, status=''):
        bar_len = 40
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('\r[%s] %s%s ...%s' % (bar, percents, '%', status))
        sys.stdout.flush()

    @staticmethod
    def rotate(window, rtst):
        dim = window.ndim
        if rtst == 0:
            return window
        elif rtst == 1:
            return np.flip(window, dim - 2)
        elif rtst == 2:
            return np.flip(window, dim - 1)
        elif rtst == 3:
            return np.rot90(window, k=1, axes=(dim - 2, dim - 1))
        elif rtst == 4:
            return np.flip(np.flip(window, dim - 2), dim - 1)
        elif rtst == 5:
            return np.rot90(window, k=3, axes=(dim - 2, dim - 1))
        elif rtst == 6:
            return np.flip(np.rot90(window, k=1, axes=(dim - 2, dim - 1)), dim - 2)
        elif rtst == 7:
            return np.flip(np.rot90(window, k=1, axes=(dim - 2, dim - 1)), dim - 1)

    @staticmethod
    def normalize(window):
        avg = window.sum() / window.numel()
        std = torch.std(window)
        result = (window - avg) / (4 * std)
        return result

    @staticmethod
    def getWindow(filename, z, y, x, depth, height, width, pad):
        window = np.ones((depth, height, width)) * pad
        start = np.array([z, y, x])
        startNew = np.copy(start)
        size = np.array([depth, height, width])
        end = start + size
        endNew = np.copy(end)
        if filename is not True and filename is not False:
            file = h5py.File(filename, 'r', libver='latest', swmr=True)
            data = file.get(list(file.keys())[0].encode('ascii', 'ignore'))
            while isinstance(data, h5py.Group):
                data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))
            for i in range(0, 3):
                if start[i] < 0:
                    startNew[i] = 0
                if end[i] > data.shape[i]:
                    endNew[i] = data.shape[i]
            extr = data[int(startNew[0]):int(endNew[0]), int(startNew[1]):int(endNew[1]), int(startNew[2]):int(endNew[2])]
            file.close()
        else:
            if filename:
                global dataInput
                ceiling = dataInput.shape
            else:
                global dataAnnotation
                ceiling = dataAnnotation.shape
            for i in range(0, 3):
                if start[i] < 0:
                    startNew[i] = 0
                if end[i] > ceiling[i]:
                    endNew[i] = ceiling[i]
            if filename:
                extr = dataInput[int(startNew[0]):int(endNew[0]), int(startNew[1]):int(endNew[1]), int(startNew[2]):int(endNew[2])]
            else:
                extr = dataAnnotation[int(startNew[0]):int(endNew[0]), int(startNew[1]):int(endNew[1]), int(startNew[2]):int(endNew[2])]
        startDiff = startNew - start
        endDiff = size - (end - endNew)
        window[int(startDiff[0]):int(endDiff[0]), int(startDiff[1]):int(endDiff[1]),
        int(startDiff[2]):int(endDiff[2])] = extr
        if np.isnan(pad):
            window[np.isnan(window)] = 0
        else:
            window[np.isnan(window)] = pad
        return window

    @staticmethod
    def par_preprocess_weighted(vec, filenameA, n_classes, input_stride, numel, start_points, output_area, prunned_nodes, itr):
        i = int(np.floor(itr / (numel[1] * numel[2])))
        left = itr - i * (numel[1] * numel[2])
        j = int(np.floor(left / (numel[2])))
        k = int(left - j * numel[2])
        x = k * input_stride[2] + vec[i, 0] + start_points[2]
        y = j * input_stride[1] + vec[i, 1] + start_points[1]
        z = i * input_stride[0] + start_points[0]
        ordr = np.empty((3,))
        ordr[:] = np.nan
        weight = np.zeros((n_classes,))
        patch_flags = np.zeros((n_classes,))
        # out = False
        ex_data = torch.from_numpy(DataTemplate.getWindow(filenameA,
                                                          z + output_area[0, 0],
                                                          y + output_area[0, 1],
                                                          x + output_area[0, 2],
                                                          output_area[1, 0],
                                                          output_area[1, 1],
                                                          output_area[1, 2],
                                                          0))
        if prunned_nodes[0] != -1:
            for l in range(0, prunned_nodes.shape[1]):
                ex_data[ex_data == float(prunned_nodes[0, l])] = float(prunned_nodes[1, l])
        if ex_data.max() > 0:  # other class present except Air: 0
            ordr = np.array([z, y, x])
            for r in range(0, n_classes):
                if (ex_data == r).sum() > 0:
                    weight[r] += (ex_data == r).sum()
                    patch_flags[r] += 1
        '''
        else:
            if np.random.uniform(0, 100, 1) < 5:  # Add Air wot 15% chance provided the 512x512 lays completely in the data
                ordr = np.array([z, y, x])
                weight[0] += ex_data.numel()
                patch_flags[0] += 1
        '''
        res = (ordr, weight, patch_flags)
        return res

    @staticmethod
    def par_preprocess(vec, filenameA, input_stride, numel, start_points, output_area, prunned_nodes, itr):
        i = int(np.floor(itr / (numel[1] * numel[2])))
        left = itr - i * (numel[1] * numel[2])
        j = int(np.floor(left / (numel[2])))
        k = int(left - j * numel[2])
        x = k * input_stride[2] + vec[i, 0] + start_points[2]
        y = j * input_stride[1] + vec[i, 1] + start_points[1]
        z = i * input_stride[0] + start_points[0]
        ordr = np.empty((1, 3))
        ordr[:] = np.nan
        # out = False
        ex_data = torch.from_numpy(DataTemplate.getWindow(filenameA,
                                                          z + output_area[0, 0],
                                                          y + output_area[0, 1],
                                                          x + output_area[0, 2],
                                                          output_area[1, 0],
                                                          output_area[1, 1],
                                                          output_area[1, 2],
                                                          0))
        if prunned_nodes[0] != -1:
            for l in range(0, prunned_nodes.shape[1]):
                ex_data[ex_data == float(prunned_nodes[0, l])] = float(prunned_nodes[1, l])
        if ex_data.max() > 0:  # other class present except Air: 0
            ordr = np.array([z, y, x])
        '''
        else:
            if np.random.uniform(0, 100, 1) < 5:  # Add Air wot 15% chance provided the 512x512 lays completely in the data
                ordr = np.array([z, y, x])
        '''
        return ordr

    def preprocess(self, options):
        vec = np.random.normal(0, options.random_std, (self.numel[0], 2)).astype(int)
        if options.weights[0] == -1:
            func = partial(DataTemplate.par_preprocess_weighted, vec, self.filenameA, options.n_classes, options.input_stride, self.numel, self.start_points, self.output_area, self.prunned_nodes)
        else:
            func = partial(DataTemplate.par_preprocess, vec, self.filenameA, options.input_stride, self.numel, self.start_points, self.output_area, self.prunned_nodes)
        pool = mp.Pool(processes=options.workers)
        if options.weights[0] == -1:
            out = list(tqdm.tqdm(pool.imap(func, range(0, self.numel[0] * self.numel[1] * self.numel[2])), total=self.numel[0] * self.numel[1] * self.numel[2]))
            order = np.stack([out[i][0] for i in range(len(out))])
            order = order[~np.isnan(order).any(axis=1)]
            weights = np.stack([out[i][1] for i in range(len(out))]).sum(0)
            patchcount = np.stack([out[i][2] for i in range(len(out))]).sum(0)
            weights = weights / patchcount
            weights = np.median(weights) / weights
        else:
            out = np.vstack(list(tqdm.tqdm(pool.imap(func, range(0, self.numel[0] * self.numel[1] * self.numel[2])), total=self.numel[0] * self.numel[1] * self.numel[2])))
            weights = options.weights
            order = out
            order = order[~np.isnan(order).any(axis=1)]
        return order, weights

    @staticmethod
    def par_recalc(filenameA, order, n_classes, output_area, prunned_nodes, itr):
        weight = np.zeros((n_classes,))
        patch_flags = np.zeros((n_classes,))
        # out = False
        z = order[itr, 0]
        y = order[itr, 1]
        x = order[itr, 2]
        ex_data = torch.from_numpy(DataTemplate.getWindow(filenameA,
                                                          z + output_area[0, 0],
                                                          y + output_area[0, 1],
                                                          x + output_area[0, 2],
                                                          output_area[1, 0],
                                                          output_area[1, 1],
                                                          output_area[1, 2],
                                                          0))
        if prunned_nodes[0] != -1:
            for l in range(0, prunned_nodes.shape[1]):
                ex_data[ex_data == float(prunned_nodes[0, l])] = float(prunned_nodes[1, l])
        for r in range(0, n_classes):
            if (ex_data == r).sum() > 0:
                weight[r] += (ex_data == r).sum()
                patch_flags[r] += 1
        res = (weight, patch_flags)
        return res

    def recalc(self, options, order):
        func = partial(DataTemplate.par_recalc, self.filenameA, order, options.n_classes, self.output_area, self.prunned_nodes)
        pool = mp.Pool(processes=options.workers)
        out = list(tqdm.tqdm(pool.imap(func, range(0, len(order))), total=len(order)))
        weights = np.stack([out[i][0] for i in range(len(out))]).sum(0)
        patchcount = np.stack([out[i][1] for i in range(len(out))]).sum(0)
        weights = weights / patchcount
        weights = np.median(weights) / weights
        return weights


class Data(DataTemplate):
    def __getitem__(self, idx):
        # Returns a batch for each epoch
        # Edited
        if self.mode is 'train':
            data = self.batchTrain
        elif self.mode is 'val':
            data = self.batchVal
        else:
            data = self.batchTest
        torch_dataI = torch.empty(int(self.input_size[0]),
                                  int(self.input_size[1]),
                                  int(self.input_size[2]))
        torch_dataA = torch.empty(int(self.output_area[1, 1]),
                                  int(self.output_area[1, 2]))
        if self.criterion == 'Mixed':
            torch_dataA2 = torch.empty(int(self.output_area[1, 1]),
                                       int(self.output_area[1, 2]))

        z = data[idx, 0]
        y = data[idx, 1]
        x = data[idx, 2]
        rtst = int(data[idx, 3])
        if self.normalize:
            torch_dataI[:, :, :] = Data.normalize(torch.from_numpy(DataTemplate.rotate(DataTemplate.getWindow(self.filenameI,
                                                                                                              z,
                                                                                                              y,
                                                                                                              x,
                                                                                                              self.input_size[0],
                                                                                                              self.input_size[1],
                                                                                                              self.input_size[2],
                                                                                                              self.pad), rtst).copy()))
        else:
            torch_dataI[:, :, :] = torch.from_numpy(DataTemplate.rotate(DataTemplate.getWindow(self.filenameI,
                                                                                               z,
                                                                                               y,
                                                                                               x,
                                                                                               self.input_size[0],
                                                                                               self.input_size[1],
                                                                                               self.input_size[2],
                                                                                               self.pad), rtst).copy())
        # self.dataI[z:z+self.input_size[0], y:y+self.input_size[1], x:x+self.input_size[2]])
        torch_dataA[:, :] = torch.from_numpy(DataTemplate.rotate(DataTemplate.getWindow(self.filenameA,
                                                                                        z + self.output_area[0, 0],
                                                                                        y + self.output_area[0, 1],
                                                                                        x + self.output_area[0, 2],
                                                                                        self.output_area[1, 0],
                                                                                        self.output_area[1, 1],
                                                                                        self.output_area[1, 2],
                                                                                        0)[0, :, :], rtst).copy())
        if self.criterion == 'Mixed':
            torch_dataA2[:, :] = torch.from_numpy(DataTemplate.rotate(DataTemplate.getWindow('/db/psxdb3/havok/db1/tomo_p2_astra_recon_gpu.h5',
                                                                                               z + self.output_area[0, 0],
                                                                                               y + self.output_area[0, 1],
                                                                                               x + self.output_area[0, 2],
                                                                                               self.output_area[1, 0],
                                                                                               self.output_area[1, 1],
                                                                                               self.output_area[1, 2],
                                                                                               0)[0, :, :], rtst).copy())
        if self.criterion != 'Mixed':
            if self.criterion == 'MSE':
                torch_dataA = torch_dataA.unsqueeze(0)
            else:
                if self.prunned_nodes[0] != -1:
                    for l in range(0, self.prunned_nodes.shape[1]):
                        torch_dataA[torch_dataA == float(self.prunned_nodes[0, l])] = float(self.prunned_nodes[1, l])
                torch_dataA = torch_dataA.long()
            return torch_dataI, torch_dataA
        else:
            torch_dataA2 = torch_dataA2.unsqueeze(0)
            if self.prunned_nodes[0] != -1:
                for l in range(0, self.prunned_nodes.shape[1]):
                    torch_dataA[torch_dataA == float(self.prunned_nodes[0, l])] = float(self.prunned_nodes[1, l])
            torch_dataA = torch_dataA.long()
            return torch_dataI, torch_dataA, torch_dataA2

    def __len__(self):
        if self.mode is 'train':
            return self.num_train
        elif self.mode is 'val':
            return self.num_val
        else:
            return self.num_test


class Data3D(DataTemplate):
    def __getitem__(self, idx):
        # Returns a batch for each epoch
        # Edited
        if self.mode is 'train':
            data = self.batchTrain
        elif self.mode is 'val':
            data = self.batchVal
        else:
            data = self.batchTest
        torch_dataI = torch.empty(1,
                                  int(self.input_size[0]),
                                  int(self.input_size[1]),
                                  int(self.input_size[2]))
        torch_dataA = torch.empty(int(self.output_area[1, 0]),
                                  int(self.output_area[1, 1]),
                                  int(self.output_area[1, 2]))
        if self.criterion == 'Mixed':
            torch_dataA2 = torch.empty(int(self.output_area[1, 0]),
                                      int(self.output_area[1, 1]),
                                      int(self.output_area[1, 2]))
        z = data[idx, 0]
        y = data[idx, 1]
        x = data[idx, 2]
        rtst = int(data[idx, 3])
        if self.normalize:
            torch_dataI[0, :, :, :] = DataTemplate.normalize(torch.from_numpy(DataTemplate.rotate(DataTemplate.getWindow(self.filenameI,
                                                                                                                         z,
                                                                                                                         y,
                                                                                                                         x,
                                                                                                                         self.input_size[0],
                                                                                                                         self.input_size[1],
                                                                                                                         self.input_size[2],
                                                                                                                         self.pad), rtst).copy()))
        else:
            torch_dataI[0, :, :, :] = torch.from_numpy(DataTemplate.rotate(DataTemplate.getWindow(self.filenameI,
                                                                                                  z,
                                                                                                  y,
                                                                                                  x,
                                                                                                  self.input_size[0],
                                                                                                  self.input_size[1],
                                                                                                  self.input_size[2],
                                                                                                  self.pad), rtst).copy())
        # self.dataI[z:z+self.input_size[0], y:y+self.input_size[1], x:x+self.input_size[2]])
        torch_dataA[:, :, :] = torch.from_numpy(DataTemplate.rotate(DataTemplate.getWindow(self.filenameA,
                                                                                           z + self.output_area[0, 0],
                                                                                           y + self.output_area[0, 1],
                                                                                           x + self.output_area[0, 2],
                                                                                           self.output_area[1, 0],
                                                                                           self.output_area[1, 1],
                                                                                           self.output_area[1, 2],
                                                                                           0), rtst).copy())
        if self.criterion == 'Mixed':
            torch_dataA2[:, :, :] = torch.from_numpy(DataTemplate.rotate(DataTemplate.getWindow('/db/psxdb3/havok/db1/tomo_p2_astra_recon_gpu.h5',
                                                                                               z + self.output_area[0, 0],
                                                                                               y + self.output_area[0, 1],
                                                                                               x + self.output_area[0, 2],
                                                                                               self.output_area[1, 0],
                                                                                               self.output_area[1, 1],
                                                                                               self.output_area[1, 2],
                                                                                               0), rtst).copy())
        if self.criterion != 'Mixed':
            if self.criterion == 'MSE':
                torch_dataA = torch_dataA.unsqueeze(0) # A2 for stack
            else:
                if self.prunned_nodes[0] != -1:
                    for l in range(0, self.prunned_nodes.shape[1]):
                        torch_dataA[torch_dataA == float(self.prunned_nodes[0, l])] = float(self.prunned_nodes[1, l])
                torch_dataA = torch_dataA.long()
            return torch_dataI, torch_dataA
        else:
            torch_dataA2 = torch_dataA2.unsqueeze(0) # A2 for stack
            if self.prunned_nodes[0] != -1:
                for l in range(0, self.prunned_nodes.shape[1]):
                    torch_dataA[torch_dataA == float(self.prunned_nodes[0, l])] = float(self.prunned_nodes[1, l])
            torch_dataA = torch_dataA.long()
            return torch_dataI, torch_dataA, torch_dataA2

    def __len__(self):
        if self.mode is 'train':
            return self.num_train
        elif self.mode is 'val':
            return self.num_val
        else:
            return self.num_test


class Data_fast_template(DataTemplate):
    def processing(self, options):
        fileA = h5py.File(self.filenameA, 'r')
        dataA = fileA.get(list(fileA.keys())[0].encode('ascii', 'ignore'))
        while isinstance(dataA, h5py.Group):
            dataA = dataA.get(list(dataA.keys())[0].encode('ascii', 'ignore'))
        fileI = h5py.File(self.filenameI, 'r')
        dataI = fileI.get(list(fileI.keys())[0].encode('ascii', 'ignore'))
        while isinstance(dataI, h5py.Group):
            dataI = dataI.get(list(dataI.keys())[0].encode('ascii', 'ignore'))
        global dataAnnotation
        dataAnnotation = dataA[self.start_points[0]:self.end_points[0], self.start_points[1]:self.end_points[1], self.start_points[2]:self.end_points[2]]
        global dataInput
        dataInput = dataI[self.start_points[0]:self.end_points[0], self.start_points[1]:self.end_points[1], self.start_points[2]:self.end_points[2]]
        order, weights = self.preprocess(options)
        del dataAnnotation, dataInput
        perm = np.random.permutation(order.shape[0])
        rotstate = np.random.randint(8, size=order.shape[0])
        self.dataAnnotation = dataA[self.start_points[0]:self.end_points[0], self.start_points[1]:self.end_points[1], self.start_points[2]:self.end_points[2]]
        self.dataInput = dataI[self.start_points[0]:self.end_points[0], self.start_points[1]:self.end_points[1], self.start_points[2]:self.end_points[2]]
        fileA.close()
        fileI.close()
        return order, perm, rotstate, weights

    def getWindowSelf(self, dataset, z, y, x, depth, height, width, pad):
        window = np.ones((depth, height, width)) * pad
        start = np.array([z, y, x])
        startNew = np.copy(start)
        size = np.array([depth, height, width])
        end = start + size
        endNew = np.copy(end)
        if dataset:
            ceiling = self.dataInput.shape
        else:
            ceiling = self.dataAnnotation.shape
        for i in range(0, 3):
            if start[i] < 0:
                startNew[i] = 0
            if end[i] > ceiling[i]:
                endNew[i] = ceiling[i]
        if dataset:
            extr = self.dataInput[int(startNew[0]):int(endNew[0]), int(startNew[1]):int(endNew[1]), int(startNew[2]):int(endNew[2])]
        else:
            extr = self.dataAnnotation[int(startNew[0]):int(endNew[0]), int(startNew[1]):int(endNew[1]), int(startNew[2]):int(endNew[2])]
        startDiff = startNew - start
        endDiff = size - (end - endNew)
        window[int(startDiff[0]):int(endDiff[0]), int(startDiff[1]):int(endDiff[1]),
        int(startDiff[2]):int(endDiff[2])] = extr
        if np.isnan(pad):
            window[np.isnan(window)] = 0
        else:
            window[np.isnan(window)] = pad
        return window


class Data_fast(Data_fast_template):
    def __getitem__(self, idx):
        # Returns a batch for each epoch
        # Edited
        if self.mode is 'train':
            data = self.batchTrain
        elif self.mode is 'val':
            data = self.batchVal
        else:
            data = self.batchTest
        torch_dataI = torch.empty(int(self.input_size[0]),
                                  int(self.input_size[1]),
                                  int(self.input_size[2]))
        torch_dataA = torch.empty(int(self.output_area[1, 1]),
                                  int(self.output_area[1, 2]))
        z = data[idx, 0]
        y = data[idx, 1]
        x = data[idx, 2]
        rtst = int(data[idx, 3])
        if self.normalize:
            torch_dataI[:, :, :] = DataTemplate.normalize(torch.from_numpy(DataTemplate.rotate(self.getWindowSelf(True,
                                                                                                                  z,
                                                                                                                  y,
                                                                                                                  x,
                                                                                                                  self.input_size[0],
                                                                                                                  self.input_size[1],
                                                                                                                  self.input_size[2],
                                                                                                                  self.pad), rtst).copy()))
        else:
            torch_dataI[:, :, :] = torch.from_numpy(DataTemplate.rotate(self.getWindowSelf(True,
                                                                                           z,
                                                                                           y,
                                                                                           x,
                                                                                           self.input_size[0],
                                                                                           self.input_size[1],
                                                                                           self.input_size[2],
                                                                                           self.pad), rtst).copy())
        # self.dataI[z:z+self.input_size[0], y:y+self.input_size[1], x:x+self.input_size[2]])
        torch_dataA[:, :] = torch.from_numpy(DataTemplate.rotate(self.getWindowSelf(False,
                                                                                    z + self.output_area[0, 0],
                                                                                    y + self.output_area[0, 1],
                                                                                    x + self.output_area[0, 2],
                                                                                    self.output_area[1, 0],
                                                                                    self.output_area[1, 1],
                                                                                    self.output_area[1, 2],
                                                                                    0)[0, :, :], rtst).copy())
        if self.prunned_nodes[0] != -1:
            for l in range(0, self.prunned_nodes.shape[1]):
                torch_dataA[torch_dataA == float(self.prunned_nodes[0, l])] = float(self.prunned_nodes[1, l])
        return torch_dataI, torch_dataA.long()

    def __len__(self):
        if self.mode is 'train':
            return self.num_train
        elif self.mode is 'val':
            return self.num_val
        else:
            return self.num_test


class Data3D_fast(Data_fast_template):
    def __getitem__(self, idx):
        # Returns a batch for each epoch
        # Edited
        if self.mode is 'train':
            data = self.batchTrain
        elif self.mode is 'val':
            data = self.batchVal
        else:
            data = self.batchTest
        torch_dataI = torch.empty(1, int(self.input_size[0]), int(self.input_size[1]), int(self.input_size[2]))
        torch_dataA = torch.empty(int(self.output_area[1, 0]), int(self.output_area[1, 1]), int(self.output_area[1, 2]))
        z = data[idx, 0]
        y = data[idx, 1]
        x = data[idx, 2]
        rtst = int(data[idx, 3])
        if self.normalize:
            torch_dataI[0, :, :, :] = Data3D_fast.normalize(torch.from_numpy(Data3D_fast.rotate(self.getWindowSelf(True,
                                                                                                                   z,
                                                                                                                   y,
                                                                                                                   x,
                                                                                                                   self.input_size[0],
                                                                                                                   self.input_size[1],
                                                                                                                   self.input_size[2],
                                                                                                                   self.pad), rtst).copy()))
        else:
            torch_dataI[0, :, :, :] = torch.from_numpy(Data3D_fast.rotate(self.getWindowSelf(True,
                                                                                             z,
                                                                                             y,
                                                                                             x,
                                                                                             self.input_size[0],
                                                                                             self.input_size[1],
                                                                                             self.input_size[2],
                                                                                             self.pad), rtst).copy())
        # self.dataI[z:z+self.input_size[0], y:y+self.input_size[1], x:x+self.input_size[2]])
        torch_dataA[:, :, :] = torch.from_numpy(Data3D_fast.rotate(self.getWindowSelf(False,
                                                                                      z + self.output_area[0, 0],
                                                                                      y + self.output_area[0, 1],
                                                                                      x + self.output_area[0, 2],
                                                                                      self.output_area[1, 0],
                                                                                      self.output_area[1, 1],
                                                                                      self.output_area[1, 2],
                                                                                      0), rtst).copy())
        if self.prunned_nodes[0] != -1:
            for l in range(0, self.prunned_nodes.shape[1]):
                torch_dataA[torch_dataA == float(self.prunned_nodes[0, l])] = float(self.prunned_nodes[1, l])
        return torch_dataI, torch_dataA.long()

    def __len__(self):
        if self.mode is 'train':
            return self.num_train
        elif self.mode is 'val':
            return self.num_val
        else:
            return self.num_test


class Data_infer(Dataset):
    def __init__(self, options):
        super(Data_infer, self).__init__()
        self.input_size = options.input_size
        self.output_area = options.output_area
        self.filenameI = options.input_filename
        self.filenameA = options.root + '/infer.h5'
        self.normalize = options.normalize
        self.output_size = options.infer_output
        self.pad = np.empty(self.filenameI.shape)
        fileI = h5py.File(self.filenameI, 'r')
        dataI = fileI.get(list(fileI.keys())[0].encode('ascii', 'ignore'))
        while isinstance(dataI, h5py.Group):
            dataI = dataI.get(list(dataI.keys())[0].encode('ascii', 'ignore'))
            self.pad = np.nanmin(dataI[int(options.input_area[0, 0]):int(options.input_area[1, 0]), int(options.input_area[0, 1]):int(options.input_area[1, 1]),
                                    int(options.input_area[0, 2]):int(options.input_area[1, 2])])
        fileI.close()
        self.data_size = options.input_area[1, :] - options.input_area[0, :]
        self.excess = np.array([np.mod((self.data_size[0] - self.input_size[0]), options.input_stride[0]),
                                np.mod((self.data_size[1] - self.input_size[1]), options.input_stride[1]),
                                np.mod((self.data_size[2] - self.input_size[2]), options.input_stride[2])])
        self.start_points = np.floor(self.excess / 2).astype(int) + options.input_area[0, :]
        self.numel = np.array([((self.data_size[0] - self.input_size[0]) // options.input_stride[0]) + 1,
                               ((self.data_size[1] - self.input_size[1]) // options.input_stride[1]) + 1,
                               ((self.data_size[2] - self.input_size[2]) // options.input_stride[2]) + 1])
        if isfile(options.cp_dest + '2D_slices.npz') and options.clean_go is False:
            files = np.load(options.cp_dest + '2D_slices.npz')
            self.order = files['order']
        else:
            self.order = self.preprocess(options)
            np.savez(options.cp_dest + '2D_slices.npz',
                     order=self.order)
        self.temp_stor = np.zeros((self.output_size[0], self.output_area[1, 0], self.output_size[2], self.output_size[3]))
        self.delim = np.zeros((self.output_size[0], self.output_area[1, 0], self.output_size[2], self.output_size[3]))
        self.save_point = self.order[0, 0]

    @staticmethod
    def getWindow(filename, z, y, x, depth, height, width, pad):
        window = np.ones((depth, height, width)) * pad
        start = np.array([z, y, x])
        startNew = np.copy(start)
        size = np.array([depth, height, width])
        end = start + size
        endNew = np.copy(end)
        file = h5py.File(filename, 'r')  # libver='latest', swmr=True)
        data = file.get(list(file.keys())[0].encode('ascii', 'ignore'))
        while isinstance(data, h5py.Group):
            data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))
        for i in range(0, 3):
            if start[i] < 0:
                startNew[i] = 0
            if end[i] > data.shape[i]:
                endNew[i] = data.shape[i]
        extr = data[int(startNew[0]):int(endNew[0]), int(startNew[1]):int(endNew[1]), int(startNew[2]):int(endNew[2])]
        file.close()
        startDiff = startNew - start
        endDiff = size - (end - endNew)
        window[int(startDiff[0]):int(endDiff[0]), int(startDiff[1]):int(endDiff[1]), int(startDiff[2]):int(endDiff[2])] = extr
        if np.isnan(pad):
            window[np.isnan(window)] = 0
        else:
            window[np.isnan(window)] = pad
        return window

    @staticmethod
    def par_preprocess(filename, stride, start_point, input_size, itr):
        file = h5py.File(filename, 'r')  # libver='latest', swmr=True)
        data = file.get(list(file.keys())[0].encode('ascii', 'ignore'))
        while isinstance(data, h5py.Group):
            data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))
        vol = np.nansum(data[(itr * stride[0] + start_point[0]):(itr * stride[0] + start_point[0] + input_size[0]), :, :], 0)
        file.close()
        for_x = np.nansum(vol, 0)
        for_y = np.nansum(vol, 1)
        x_start, x_numel = Data_infer.boundaries(for_x, stride[2], start_point[2], input_size[2])
        y_start, y_numel = Data_infer.boundaries(for_y, stride[1], start_point[1], input_size[1])
        if x_start is None or y_start is None:
            order = np.empty((3,))
            order[:] = np.nan
        else:
            order = np.mgrid[0:1, 0:y_numel, 0:x_numel]
            order = np.rollaxis(order, 0, 4)
            order = order.reshape((y_numel * x_numel, 3))
            order[:, 0] = order[:, 0] + itr * stride[0] + start_point[0]
            order[:, 1] = order[:, 1] * stride[1] + y_start
            order[:, 2] = order[:, 2] * stride[2] + x_start
        return order

    @staticmethod
    def boundaries(proj, stride, start_point, input_size):
        peaks, _ = find_peaks(-proj, prominence=proj.max() / 50)
        start = peaks[0]
        end = peaks[-1]
        if end < start_point:
            return None, None
        else:
            if start < start_point:
                start = start_point
            numel = int(np.ceil((((end - start) - input_size) / stride) + 2))
            area = (numel - 1) * stride + input_size
            middle = (end + start) / 2
            starting = int(np.round((middle - area / 2)))
            return starting, numel

    @staticmethod
    def normalize(window):
        avg = window.sum() / window.numel()
        std = torch.std(window)
        result = (window - avg) / (4 * std)
        return result

    def preprocess(self, options):
        func = partial(Data_infer.par_preprocess, self.filenameI, options.input_stride, self.start_points, self.input_size)
        pool = mp.Pool(processes=options.workers)
        order = np.vstack(list(tqdm.tqdm(pool.imap(func, range(0, self.numel[0])), total=self.numel[0])))
        return order[~np.isnan(order).any(axis=1)]

    def save(self, y, idx, options):
        # y = F.softmax(y, dim=1)
        for i in range(0, len(idx)):
            if self.order[idx[i], 0] != self.save_point:
                if isfile(self.filenameA):
                    fileA = h5py.File(self.filenameA, 'a')  # libver='latest', swmr=True)
                    dataA = fileA.get(list(fileA.keys())[0].encode('ascii', 'ignore'))
                else:
                    fileA = h5py.File(self.filenameA, 'w')  # libver='latest', swmr=True)
                    dataA = fileA.create_dataset('/data', tuple(self.output_size[self.output_size!=1]), chunks=True)
                div = self.delim[:, :options.input_stride[0], :, :]
                div[div == 0] = 1
                output = self.temp_stor[:, :options.input_stride[0], :, :] / div
                #output = np.argmax(output, axis=0)
                if dataA.ndim == 3:
                    dataA[(self.save_point + self.output_area[0, 0]):(self.save_point + self.output_area[0, 0] + options.input_stride[0]), :, :] = output[0, :, :, :]
                else:
                    for j in range(0, self.output_size[0]):
                        dataA[j, (self.save_point + self.output_area[0, 0]):(self.save_point + self.output_area[0, 0] + options.input_stride[0]), :, :] = output[j, :, :, :]
                self.temp_stor = np.pad(self.temp_stor, ((0, 0), (0, options.input_stride[0]), (0, 0), (0, 0)), mode='constant')[:, options.input_stride[0]:, :, :]
                self.delim = np.pad(self.delim, ((0, 0), (0, options.input_stride[0]), (0, 0), (0, 0)), mode='constant')[:, options.input_stride[0]:, :, :]
                fileA.close()
            for_file = y[i,
                         :,
                         (self.output_area[0, 1]):(self.output_area[0, 1] + self.output_area[1, 1]),
                         (self.output_area[0, 2]):(self.output_area[0, 2] + self.output_area[1, 2])]
            if options.gpu:
                self.temp_stor[:,
                0,
                (self.order[idx[i], 1] + self.output_area[0, 1]):(self.order[idx[i], 1] + self.output_area[0, 1] + self.output_area[1, 1]),
                (self.order[idx[i], 2] + self.output_area[0, 2]):(self.order[idx[i], 2] + self.output_area[0, 2] + self.output_area[1, 2])] += for_file.cpu().detach().numpy()
            else:
                self.temp_stor[:,
                0,
                (self.order[idx[i], 1] + self.output_area[0, 1]):(self.order[idx[i], 1] + self.output_area[0, 1] + self.output_area[1, 1]),
                (self.order[idx[i], 2] + self.output_area[0, 2]):(self.order[idx[i], 2] + self.output_area[0, 2] + self.output_area[1, 2])] += for_file.detach().numpy()
            self.delim[:,
            0,
            (self.order[idx[i], 1] + self.output_area[0, 1]):(self.order[idx[i], 1] + self.output_area[0, 1] + self.output_area[1, 1]),
            (self.order[idx[i], 2] + self.output_area[0, 2]):(self.order[idx[i], 2] + self.output_area[0, 2] + self.output_area[1, 2])] += np.ones((for_file.shape[0], for_file.shape[1], for_file.shape[2]))
            self.save_point = self.order[idx[i], 0]

    def __getitem__(self, idx):
        # Returns a batch for each epoch
        # Edited
        torch_dataI = torch.empty(int(self.input_size[0]),
                                  int(self.input_size[1]),
                                  int(self.input_size[2]))
        z = self.order[idx, 0]
        y = self.order[idx, 1]
        x = self.order[idx, 2]
        torch_dataI[:, :, :] = torch.from_numpy(Data_infer.getWindow(self.filenameI,
                                                                        z,
                                                                        y,
                                                                        x,
                                                                        self.input_size[0],
                                                                        self.input_size[1],
                                                                        self.input_size[2],
                                                                        self.pad).copy())
        if self.normalize:
            torch_dataI = Data.normalize(torch_dataI)
        return torch_dataI, idx

    def __len__(self):
        return len(self.order)


class Data3D_infer(Dataset):
    def __init__(self, options):
        super(Data3D_infer, self).__init__()
        self.input_size = options.input_size
        self.output_area = options.output_area
        self.filenameI = options.input_filename
        self.filenameA = options.root + 'infer.h5'
        self.normalize = options.normalize
        self.pad = np.empty(self.filenameI.shape)
        self.output_size = options.infer_output
        fileI = h5py.File(self.filenameI, 'r')
        dataI = fileI.get(list(fileI.keys())[0].encode('ascii', 'ignore'))
        while isinstance(dataI, h5py.Group):
            dataI = dataI.get(list(dataI.keys())[0].encode('ascii', 'ignore'))
            self.pad = np.nanmin(dataI[int(options.input_area[0, 0]):int(options.input_area[1, 0]), int(options.input_area[0, 1]):int(options.input_area[1, 1]),
                                    int(options.input_area[0, 2]):int(options.input_area[1, 2])])
        fileI.close()
        self.data_size = options.input_area[1, :] - options.input_area[0, :]
        self.excess = np.array([np.mod((self.data_size[0] - self.input_size[0]), options.input_stride[0]),
                                np.mod((self.data_size[1] - self.input_size[1]), options.input_stride[1]),
                                np.mod((self.data_size[2] - self.input_size[2]), options.input_stride[2])])
        self.start_points = np.floor(self.excess / 2).astype(int) + options.input_area[0, :]
        self.numel = np.array([((self.data_size[0] - self.input_size[0]) // options.input_stride[0]) + 1,
                               ((self.data_size[1] - self.input_size[1]) // options.input_stride[1]) + 1,
                               ((self.data_size[2] - self.input_size[2]) // options.input_stride[2]) + 1])
        self.order = self.preprocess(options)
        self.temp_stor = np.zeros((options.n_classes, self.output_area[1, 0], self.output_size[1], self.output_size[2]))
        self.delim = np.zeros((options.n_classes, self.output_area[1, 0], self.output_size[1], self.output_size[2]))
        self.save_point = self.order[0, 0]

    @staticmethod
    def getWindow(filename, z, y, x, depth, height, width, pad):
        window = np.ones((depth, height, width)) * pad
        start = np.array([z, y, x])
        startNew = np.copy(start)
        size = np.array([depth, height, width])
        end = start + size
        endNew = np.copy(end)
        file = h5py.File(filename, 'r')  # libver='latest', swmr=True)
        data = file.get(list(file.keys())[0].encode('ascii', 'ignore'))
        while isinstance(data, h5py.Group):
            data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))
        for i in range(0, 3):
            if start[i] < 0:
                startNew[i] = 0
            if end[i] > data.shape[i]:
                endNew[i] = data.shape[i]
        extr = data[int(startNew[0]):int(endNew[0]), int(startNew[1]):int(endNew[1]), int(startNew[2]):int(endNew[2])]
        file.close()
        startDiff = startNew - start
        endDiff = size - (end - endNew)
        window[int(startDiff[0]):int(endDiff[0]), int(startDiff[1]):int(endDiff[1]), int(startDiff[2]):int(endDiff[2])] = extr
        if np.isnan(pad):
            window[np.isnan(window)] = 0
        else:
            window[np.isnan(window)] = pad
        return window

    @staticmethod
    def par_preprocess(filename, stride, start_point, input_size, itr):
        file = h5py.File(filename, 'r')  # libver='latest', swmr=True)
        data = file.get(list(file.keys())[0].encode('ascii', 'ignore'))
        while isinstance(data, h5py.Group):
            data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))
        vol = np.nansum(data[(itr * stride[0] + start_point[0]):(itr * stride[0] + start_point[0] + input_size[0]), :, :], 0)
        file.close()
        for_x = np.nansum(vol, 0)
        for_y = np.nansum(vol, 1)
        x_start, x_numel = Data3D_infer.boundaries(for_x, stride[2], start_point[2], input_size[2])
        y_start, y_numel = Data3D_infer.boundaries(for_y, stride[1], start_point[1], input_size[1])
        if x_start is None or y_start is None:
            order = np.empty((3,))
            order[:] = np.nan
        else:
            order = np.mgrid[0:1, 0:y_numel, 0:x_numel]
            order = np.rollaxis(order, 0, 4)
            order = order.reshape((y_numel * x_numel, 3))
            order[:, 0] = order[:, 0] + itr * stride[0] + start_point[0]
            order[:, 1] = order[:, 1] * stride[1] + y_start
            order[:, 2] = order[:, 2] * stride[2] + x_start
        return order

    @staticmethod
    def boundaries(proj, stride, start_point, input_size):
        peaks, _ = find_peaks(-proj, prominence=proj.max() / 50)
        start = peaks[0]
        end = peaks[-1]
        if end < start_point:
            return None, None
        else:
            if start < start_point:
                start = start_point
            numel = int(np.ceil((((end - start) - input_size) / stride) + 1))
            area = (numel - 1) * stride + input_size
            middle = (end + start) / 2
            starting = int(np.round((middle - area / 2)))
            return starting, numel

    @staticmethod
    def normalize(window):
        avg = window.sum() / window.numel()
        std = torch.std(window)
        result = (window - avg) / (4 * std)
        return result

    def preprocess(self, options):
        func = partial(Data3D_infer.par_preprocess, self.filenameI, options.input_stride, self.start_points, self.input_size)
        pool = mp.Pool(processes=options.workers)
        order = np.vstack(list(tqdm.tqdm(pool.imap(func, range(0, self.numel[0])), total=self.numel[0])))
        return order[~np.isnan(order).any(axis=1)]

    def save(self, y, idx, options):
        # y = F.softmax(y, dim=1)
        for i in range(0, len(idx)):
            if self.order[idx[i], 0] != self.save_point:
                if self.save_point != self.order[0, 0]:
                    fileA = h5py.File(self.filenameA, 'a')
                    dataA = fileA.get(list(fileA.keys())[0].encode('ascii', 'ignore'))
                else:
                    fileA = h5py.File(self.filenameA, 'w')
                    dataA = fileA.create_dataset('/data', tuple(self.output_size), chunks=True)
                div = self.delim[:, :options.input_stride[0], :, :]
                div[div == 0] = 1
                output = self.temp_stor[:, :options.input_stride[0], :, :] / div
                # output = np.argmax(output, axis=0)
                dataA[:, (self.save_point + self.output_area[0, 0]):(self.save_point + self.output_area[0, 0] + options.input_stride[0]), :, :] = output
                self.temp_stor = np.pad(self.temp_stor, ((0, 0), (0, options.input_stride[0]), (0, 0), (0, 0)), mode='constant')[:, options.input_stride[0]:, :, :]
                self.delim = np.pad(self.delim, ((0, 0), (0, options.input_stride[0]), (0, 0), (0, 0)), mode='constant')[:, options.input_stride[0]:, :, :]
                fileA.close()
            for_file = y[i,
                         :,
                         :,
                         (self.output_area[0, 1]):(self.output_area[0, 1] + self.output_area[1, 1]),
                         (self.output_area[0, 2]):(self.output_area[0, 2] + self.output_area[1, 2])]
            if options.gpu:
                self.temp_stor[:,
                :,
                (self.order[idx[i], 1] + self.output_area[0, 1]):(self.order[idx[i], 1] + self.output_area[0, 1] + self.output_area[1, 1]),
                (self.order[idx[i], 2] + self.output_area[0, 2]):(self.order[idx[i], 2] + self.output_area[0, 2] + self.output_area[1, 2])] += for_file.cpu().detach().numpy()
            else:
                self.temp_stor[:,
                :,
                (self.order[idx[i], 1] + self.output_area[0, 1]):(self.order[idx[i], 1] + self.output_area[0, 1] + self.output_area[1, 1]),
                (self.order[idx[i], 2] + self.output_area[0, 2]):(self.order[idx[i], 2] + self.output_area[0, 2] + self.output_area[1, 2])] += for_file.detach().numpy()
            self.delim[:,
            :,
            (self.order[idx[i], 1] + self.output_area[0, 1]):(self.order[idx[i], 1] + self.output_area[0, 1] + self.output_area[1, 1]),
            (self.order[idx[i], 2] + self.output_area[0, 2]):(self.order[idx[i], 2] + self.output_area[0, 2] + self.output_area[1, 2])] += np.ones((for_file.shape[0], for_file.shape[1], for_file.shape[2], for_file.shape[3]))
            self.save_point = self.order[idx[i], 0]

    def __getitem__(self, idx):
        # Returns a batch for each epoch
        # Edited
        torch_dataI = torch.empty(1,
                                  int(self.input_size[0]),
                                  int(self.input_size[1]),
                                  int(self.input_size[2]))
        z = self.order[idx, 0]
        y = self.order[idx, 1]
        x = self.order[idx, 2]
        torch_dataI[0, :, :, :] = torch.from_numpy(Data3D_infer.getWindow(self.filenameI,
                                                                          z,
                                                                          y,
                                                                          x,
                                                                          self.input_size[0],
                                                                          self.input_size[1],
                                                                          self.input_size[2],
                                                                          self.pad).copy())
        if self.normalize:
            torch_dataI = Data3D_infer.normalize(torch_dataI)
        return torch_dataI, idx

    def __len__(self):
        return len(self.order)

class DataCamVid(Dataset):
    def __init__(self, input_filename, mode="train", is_transform=False):
        super(DataCamVid, self).__init__()
        self.root = input_filename
        self.mode = mode
        self.is_transform = is_transform
        self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.lcn = LCN_Jarret()
        self.weights = torch.Tensor([0.2595,
                                     0.1826,
                                     4.564,
                                     0.1417,
                                     0.9051,
                                     0.3826,
                                     9.6446,
                                     1.8418,
                                     0.6823,
                                     6.2478,
                                     7.3614,
                                     0])

        for mode in ["train", "test", "val"]:
            file_list = os.listdir(self.root + '/' + mode)
            self.files[mode] = file_list

    def change_mode(self, mode):
        self.mode = mode
        return

    def __len__(self):
        return len(self.files[self.mode])

    def __getitem__(self, index):
        img_name = self.files[self.mode][index]
        img_path = self.root + '/' + self.mode + '/' + img_name
        lbl_path = self.root + '/' + self.mode + 'annot/' + img_name

        img = m.imread(img_path)
        img = torch.from_numpy(np.array(img, dtype=np.float32)/255)
        img = self.lcn(img.permute(2, 0, 1))

        lbl = m.imread(lbl_path)
        lbl = torch.from_numpy(np.array(lbl, dtype=np.int32))

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl.long()

    def transform(self, img, lbl):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    @staticmethod
    def decode_segmap(temp, plot=False):
        Sky = [128, 128, 128]
        Building = [128, 0, 0]
        Pole = [192, 192, 128]
        Road_marking = [255, 69, 0]
        Road = [128, 64, 128]
        Pavement = [60, 40, 222]
        Tree = [128, 128, 0]
        SignSymbol = [192, 128, 128]
        Fence = [64, 64, 128]
        Car = [64, 0, 128]
        Pedestrian = [64, 64, 0]
        Bicyclist = [0, 128, 192]
        Unlabelled = [0, 0, 0]

        label_colours = np.array([Sky, Building, Pole, Road, Road_marking, Pavement,
                                  Tree, SignSymbol, Fence, Car, Pedestrian,
                                  Bicyclist, Unlabelled])
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, 12):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = (r / 255.0)
        rgb[:, :, 1] = (g / 255.0)
        rgb[:, :, 2] = (b / 255.0)
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb


class Data_Brats_template(Dataset):
    def __init__(self, options, mode):
        super(Data_Brats_template, self).__init__()
        self.input_size = options.input_size
        self.input_area = options.input_area
        self.output_area = options.output_area
        self.filenameI = options.input_filename
        self.filenameA = options.annotations_filename
        self.normalize = options.normalize
        self.data_3D = options.data_3D
        self.mode = mode
        self.prunned_nodes = options.prunned_classes
        exc = np.floor(np.mod((self.input_area[1, 0] - self.input_area[0, 0] - self.input_size[0]), options.input_stride[0]) / 2).astype(int)
        numel = int(((self.input_area[1, 0] - self.input_area[0, 0] - self.input_size[0]) // options.input_stride[0]) + 1)
        self.heights = np.arange(0, numel) * options.input_stride[0] + exc
        save_flag = False
        if isfile(options.cp_dest + 'Brats.npz') and options.clean_go is False:
            files = np.load(options.cp_dest + 'Brats.npz')
            order = files['order']
            if options.weights[0] == -1:
                weights = files['weights']
            else:
                weights = options.weights
        else:
            order, weights = self.preprocess(options)
            save_flag = True
        self.weights = torch.from_numpy(weights).float()
        self.num_train = (np.floor((1 - options.val_precentage - options.test_precentage) * len(order))).astype(int)
        self.num_val = (np.floor(options.val_precentage * len(order))).astype(int)
        self.num_test = len(order) - self.num_val - self.num_train
        self.batchTrain = order[0:self.num_train]
        self.batchVal = order[self.num_train:(self.num_train + self.num_val)]
        self.batchTest = order[(self.num_train + self.num_val):len(order)]
        self.num_train = self.num_train * len(self.heights)
        self.num_val = self.num_val * len(self.heights)
        self.num_test = self.num_test * len(self.heights)
        if save_flag:
            np.savez(options.cp_dest + 'Brats.npz',
                     order=order,
                     weights=weights)

    def change_mode(self, mode):
        self.mode = mode
        return

    def shuffle(self):
        permTrain = np.random.permutation(len(self.batchTrain))
        tmpBatchTrain = np.empty(int(len(self.batchTrain)), )
        for i in range(0, len(self.batchTrain)):
            tmpBatchTrain[i] = self.batchTrain[permTrain[i]]
        self.batchTrain = tmpBatchTrain
        return

    @staticmethod
    def par_preprocess(filename, order, heights, input_area, output_area, n_classes, data_3D, itr):
        file = h5py.File(filename, 'r')
        data = file.get(list(file.keys())[0].encode('ascii', 'ignore'))
        while isinstance(data, h5py.Group):
            data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))
        weight = np.zeros((n_classes,))
        patch_flags = np.zeros((n_classes,))
        for i in range(0, len(heights)):
            if data_3D:
                ex_data = data[order[itr],
                          0,
                          (heights[i] + output_area[0, 0]):(heights[i] + output_area[0, 0] + output_area[1, 0]),
                          (input_area[0, 1] + output_area[0, 1]):(input_area[0, 1] + output_area[0, 1] + output_area[1, 1]),
                          (input_area[0, 2] + output_area[0, 2]):(input_area[0, 2] + output_area[0, 2] + output_area[1, 2])]
            else:
                ex_data = data[order[itr],
                          0,
                          (heights[i] + output_area[0, 0] + int(output_area[1, 0]/2)):(heights[i] + output_area[0, 0] + int(output_area[1, 0]/2) + 1),
                          (input_area[0, 1] + output_area[0, 1]):(input_area[0, 1] + output_area[0, 1] + output_area[1, 1]),
                          (input_area[0, 2] + output_area[0, 2]):(input_area[0, 2] + output_area[0, 2] + output_area[1, 2])]
            for r in range(0, n_classes):
                if (ex_data == r).sum() > 0:
                    weight[r] += (ex_data == r).sum()
                    patch_flags[r] += 1
        file.close()
        res = (weight, patch_flags)
        return res

    @staticmethod
    def normalize(window):
        avg = window.sum() / window.numel()
        std = torch.std(window)
        result = (window - avg) / (4 * std)
        return result

    def preprocess(self, options):
        order = np.random.permutation(np.arange(options.input_size[1], options.input_size[2]))
        if options.weights[0] == -1:
            func = partial(Data_Brats.par_preprocess, self.filenameA, order, self.heights, self.input_area, self.output_area, options.n_classes, self.data_3D)
            pool = mp.Pool(processes=options.workers)
            out = list(tqdm.tqdm(pool.imap(func, range(0, len(order))), total=len(order)))
            weights = np.stack([out[i][0] for i in range(len(out))]).sum(0)
            patchcount = np.stack([out[i][1] for i in range(len(out))]).sum(0)
            weights = weights / patchcount
            weights = np.median(weights) / weights
        else:
            weights = options.weights
        return order, weights

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class Data_Brats(Data_Brats_template):
    def __getitem__(self, idx):
        # Returns a batch for each epoch
        # Edited
        fileI = h5py.File(self.filenameI, 'r')
        dataI = fileI.get(list(fileI.keys())[0].encode('ascii', 'ignore'))
        while isinstance(dataI, h5py.Group):
            dataI = dataI.get(list(dataI.keys())[0].encode('ascii', 'ignore'))
        fileA = h5py.File(self.filenameA, 'r')
        dataA = fileA.get(list(fileA.keys())[0].encode('ascii', 'ignore'))
        while isinstance(dataA, h5py.Group):
            dataA = dataA.get(list(dataA.keys())[0].encode('ascii', 'ignore'))
        if self.mode is 'train':
            order = self.batchTrain
        elif self.mode is 'val':
            order = self.batchVal
        else:
            order = self.batchTest
        input_data = []
        heig = np.mod(idx, len(self.heights)).astype(int)
        vol = int(idx // len(self.heights))
        torch_dataI = torch.from_numpy(dataI[order[vol],
                                       :,
                                       (self.heights[heig] + self.output_area[0, 0]):(self.heights[heig] + self.output_area[0, 0] + self.output_area[1, 0]),
                                       (self.input_area[0, 1] + self.output_area[0, 1]):(self.input_area[0, 1] + self.output_area[0, 1] + self.output_area[1, 1]),
                                       (self.input_area[0, 2] + self.output_area[0, 2]):(self.input_area[0, 2] + self.output_area[0, 2] + self.output_area[1, 2])])
        '''
        for c in range(0, len(self.input_size) - 3):
            if c == 0:
                input_data = np.expand_dims(dataI[order[vol],
                                            self.input_size[c + 3],
                                            (self.heights[heig] + self.output_area[0, 0]):(self.heights[heig] + self.output_area[0, 0] + self.output_area[1, 0]),
                                            (self.input_area[0, 1] + self.output_area[0, 1]):(self.input_area[0, 1] + self.output_area[0, 1] + self.output_area[1, 1]),
                                            (self.input_area[0, 2] + self.output_area[0, 2]):(self.input_area[0, 2] + self.output_area[0, 2] + self.output_area[1, 2])], axis=0)
            else:
                input_data = np.vstack((input_data,
                                        np.expand_dims(dataI[order[vol],
                                                       self.input_size[c + 3],
                                                       (self.heights[heig] + self.output_area[0, 0]):(self.heights[heig] + self.output_area[0, 0] + self.output_area[1, 0]),
                                                       (self.input_area[0, 1] + self.output_area[0, 1]):(self.input_area[0, 1] + self.output_area[0, 1] + self.output_area[1, 1]),
                                                       (self.input_area[0, 2] + self.output_area[0, 2]):(self.input_area[0, 2] + self.output_area[0, 2] + self.output_area[1, 2])], axis=0)))
        '''
        if self.normalize:
            input_data = Data_Brats.normalize(input_data)
        #torch_dataI = torch.from_numpy(input_data)
        # self.dataI[z:z+self.input_size[0], y:y+self.input_size[1], x:x+self.input_size[2]])
        if self.data_3D is False:
            torch_dataI = torch_dataI.reshape(torch_dataI.shape[0] * torch_dataI.shape[1], torch_dataI.shape[2], torch_dataI.shape[3])
            torch_dataA = torch.from_numpy(dataA[order[vol],
                                           0,
                                           (self.heights[heig] + self.output_area[0, 0] + int(self.output_area[1, 0] / 2)):(
                                                       self.heights[heig] + self.output_area[0, 0] + int(self.output_area[1, 0] / 2) + 1),
                                           (self.input_area[0, 1] + self.output_area[0, 1]):(self.input_area[0, 1] + self.output_area[0, 1] + self.output_area[1, 1]),
                                           (self.input_area[0, 2] + self.output_area[0, 2]):(self.input_area[0, 2] + self.output_area[0, 2] + self.output_area[1, 2])])
            torch_dataA = torch_dataA.squeeze()
        else:
            torch_dataA = torch.from_numpy(dataA[order[vol],
                                           0,
                                           (self.heights[heig] + self.output_area[0, 0]):(self.heights[heig] + self.output_area[0, 0] + self.output_area[1, 0]),
                                           (self.input_area[0, 1] + self.output_area[0, 1]):(self.input_area[0, 1] + self.output_area[0, 1] + self.output_area[1, 1]),
                                           (self.input_area[0, 2] + self.output_area[0, 2]):(self.input_area[0, 2] + self.output_area[0, 2] + self.output_area[1, 2])])
        if self.prunned_nodes[0] != -1:
            for l in range(0, self.prunned_nodes.shape[1]):
                torch_dataA[torch_dataA == float(self.prunned_nodes[0, l])] = float(self.prunned_nodes[1, l])
        fileI.close()
        fileA.close()
        return torch_dataI, torch_dataA.long()

    def __len__(self):
        if self.mode is 'train':
            return self.num_train
        elif self.mode is 'val':
            return self.num_val
        else:
            return self.num_test