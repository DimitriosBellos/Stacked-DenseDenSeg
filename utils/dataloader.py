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
        self.filenameIA = options.intermediate_filename
        self.normalize = options.normalize
        self.data_size = np.empty((options.input_area.shape[0], 3))
        self.excess = np.empty((options.input_area.shape[0], 3))
        self.start_points = np.empty((options.input_area.shape[0], 3))
        self.numel = np.empty((options.input_area.shape[0], 3))
        self.end_points = np.empty((options.input_area.shape[0], 3))
        self.pad = np.empty((2, options.input_area.shape[0],))
        for i in range(0, options.input_area.shape[0]):
            fileI = h5py.File(self.filenameI[i], 'r')
            dataI = fileI.get(list(fileI.keys())[0].encode('ascii', 'ignore'))
            while isinstance(dataI, h5py.Group):
                dataI = dataI.get(list(dataI.keys())[0].encode('ascii', 'ignore'))
            self.data_size[i, :] = options.input_area[i, 1, :] - options.input_area[i, 0, :]
            self.excess[i, :] = np.array([np.mod((self.data_size[i, 0] - self.input_size[0]), options.input_stride[0]),
                                    np.mod((self.data_size[i, 1] - self.input_size[1]), options.input_stride[1]),
                                    np.mod((self.data_size[i, 2] - self.input_size[2]), options.input_stride[2])])
            self.start_points[i, :] = np.floor(self.excess[i, :] / 2).astype(int) + options.input_area[i, 0, :]
            self.numel[i, :] = np.array([((self.data_size[i, 0] - self.input_size[0]) // options.input_stride[0]) + 1,
                                   ((self.data_size[i, 1] - self.input_size[1]) // options.input_stride[1]) + 1,
                                   ((self.data_size[i, 2] - self.input_size[2]) // options.input_stride[2]) + 1])
            self.end_points[i, :] = np.array([((self.numel[i, 0] - 1) * options.input_stride[0]) + self.input_size[0],
                                        ((self.numel[i, 1] - 1) * options.input_stride[1]) + self.input_size[1],
                                        ((self.numel[i, 2] - 1) * options.input_stride[2]) + self.input_size[2]]) + self.start_points[i, :]
            self.pad[0, i] = dataI[0, 0, 0]
            fileI.close()
            if self.criterion == 'Mixed':
                fileIA = h5py.File(self.filenameIA[i], 'r')
                dataIA = fileIA.get(list(fileIA.keys())[0].encode('ascii', 'ignore'))
                while isinstance(dataIA, h5py.Group):
                    dataIA = dataIA.get(list(dataIA.keys())[0].encode('ascii', 'ignore'))
                self.pad[1, i] = dataIA[0, 0, 0]
                fileIA.close()
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
            if input_area.ndim == 2:
                input_area = np.expand_dims(input_area, axis=0)
            if (np.array_equal(input_size, options.input_size) and
                    np.array_equal(input_area, options.input_area) and
                    np.array_equal(input_stride, options.input_stride)):
                order = files['order']
                perm = files['perm']
                rotstate = files['rotstate']
                if 'filename' in files.files:
                    filename = files['filename']
                else:
                    filename = np.zeros(len(perm)).astype(int)
                if options.weights[0] == -1:
                    weights = files['weights']
                elif options.weights[0] == -2:
                    weights = self.recalc(options, order, filename)
                else:
                    weights = options.weights
                files.close()
                save_flag = False
                self.load(options)
        if save_flag:
            order, perm, rotstate, weights, filename = self.processing(options)
        sm = filename.sum()
        br = -1
        for i in range(len(order)-1,-1, -1):
            if filename[i] == 0:
                br = i + 1
                if filename[br:].sum() != sm:
                    print('Error %d %d' % (sm, filename[br:].sum()))
        #perm0 = np.random.permutation(br)
        #perm1 = np.random.permutation(order.shape[0]-br)+br
        #perm = np.concatenate((perm0, perm1), axis=0)
        perm = perm[0:int(len(perm))]
        self.weights = torch.from_numpy(weights).float()
        self.num_train_full = (np.floor((1 - options.val_precentage - options.test_precentage) * len(perm))).astype(int)
        self.num_val = (np.floor(options.val_precentage * len(perm))).astype(int)
        self.num_test = len(perm) - self.num_val - self.num_train_full
        self.num_train = int(self.num_train_full)
        permTrain = perm[0:self.num_train]
        permVal = perm[self.num_train_full:(self.num_train_full + self.num_val)]
        permTest = perm[(self.num_train_full + self.num_val):len(perm)]
        #permVal = perm[0:self.num_val]
        #permTest = perm[self.num_val:(self.num_test + self.num_val)]
        #permTrain = perm[(self.num_test + self.num_val):]
        self.batchTrain = np.empty([self.num_train, 5])
        self.batchTrain[:] = np.nan
        self.batchVal = np.empty([self.num_val.astype(int), 5])
        self.batchVal[:] = np.nan
        self.batchTest = np.empty([self.num_test.astype(int), 5])
        self.batchTest[:] = np.nan
        for i in range(0, self.num_train):
            self.batchTrain[i, 4] = filename[permTrain[i]]
            self.batchTrain[i, 3] = rotstate[i]
            self.batchTrain[i, 2] = order[permTrain[i], 2]
            self.batchTrain[i, 1] = order[permTrain[i], 1]
            self.batchTrain[i, 0] = order[permTrain[i], 0]
        for i in range(0, self.num_val):
            self.batchVal[i, 4] = filename[permVal[i]]
            self.batchVal[i, 3] = rotstate[i + self.num_train_full]
            self.batchVal[i, 2] = order[permVal[i], 2]
            self.batchVal[i, 1] = order[permVal[i], 1]
            self.batchVal[i, 0] = order[permVal[i], 0]
        for i in range(0, self.num_test):
            self.batchTest[i, 4] = filename[permTest[i]]
            self.batchTest[i, 3] = rotstate[i + self.num_train_full + self.num_val]
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

    def load(self, options):
        pass

    def processing(self, options):
        order, weights, filename = self.preprocess(options)
        perm = np.random.permutation(order.shape[0])
        rotstate = np.random.randint(8, size=order.shape[0])
        return order, perm, rotstate, weights, filename

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
    def par_preprocess_weighted(vec, filenameA, n_classes, input_stride, nl, numel, start_points, output_area, prunned_nodes, itr):
        filename = 0
        for i in range(0, len(nl)):
            if itr < nl[i]:
                filename = i
                break
        if filename > 0:
            ind = nl[filename-1]
        else:
            ind = 0
        left = itr - ind
        i = int(np.floor(left / (numel[filename, 1] * numel[filename, 2])))
        left = itr - i * (numel[filename, 1] * numel[filename, 2])
        j = int(np.floor(left / (numel[filename, 2])))
        k = int(left - j * numel[filename, 2])
        x = k * input_stride[2] + vec[filename, i, 0] + start_points[filename, 2]
        y = j * input_stride[1] + vec[filename, i, 1] + start_points[filename, 1]
        z = i * input_stride[0] + start_points[filename, 0]
        ordr = np.empty((3,))
        ordr[:] = np.nan
        weight = np.zeros((n_classes,))
        patch_flags = np.zeros((n_classes,))
        # out = False
        ex_data = torch.from_numpy(DataTemplate.getWindow(filenameA[filename],
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
        res = (ordr, weight, patch_flags, filename)
        return res

    @staticmethod
    def par_preprocess(vec, filenameA, input_stride, nl, numel, start_points, output_area, prunned_nodes, itr):
        filename = 0
        for i in range(0, len(nl)):
            if itr < nl[i]:
                filename = i
                break
        if filename > 0:
            ind = nl[int(filename-1)]
        else:
            ind = 0
        left = itr - ind
        i = int(np.floor(left / (numel[filename, 1] * numel[filename, 2])))
        left = itr - i * (numel[filename, 1] * numel[filename, 2])
        j = int(np.floor(left / (numel[filename, 2])))
        k = int(left - j * numel[filename, 2])
        x = k * input_stride[2] + vec[filename, i, 0] + start_points[filename, 2]
        y = j * input_stride[1] + vec[filename, i, 1] + start_points[filename, 1]
        z = i * input_stride[0] + start_points[filename, 0]
        ordr = np.empty((1, 3))
        ordr[:] = np.nan
        # out = False
        ex_data = torch.from_numpy(DataTemplate.getWindow(filenameA[filename],
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
        return ordr, filename

    def preprocess(self, options):
        vec = np.random.normal(0, options.random_std, (int(self.numel.shape[0]), int(self.numel[:, 0].max()), 2)).astype(int)
        nl = self.numel.prod(axis=1)
        tmp = nl.copy()
        for i in range(1, len(nl)):
            nl[i] = tmp[i]+tmp[0:i].sum()
        if options.weights[0] == -1:
            func = partial(DataTemplate.par_preprocess_weighted, vec, self.filenameA, options.n_classes, options.input_stride, nl, self.numel, self.start_points, self.output_area, self.prunned_nodes)
        else:
            func = partial(DataTemplate.par_preprocess, vec, self.filenameA, options.input_stride, nl, self.numel, self.start_points, self.output_area, self.prunned_nodes)
        pool = mp.Pool(processes=options.workers)
        if options.weights[0] == -1:
            out = list(tqdm.tqdm(pool.imap(func, range(0, int(self.numel.prod(axis=1).sum()))), total=int(self.numel.prod(axis=1).sum())))
            order = np.stack([out[i][0] for i in range(len(out))])
            filename = np.stack([out[i][3] for i in range(len(out))])
            filename = filename[~np.isnan(order).any(axis=1)]
            order = order[~np.isnan(order).any(axis=1)]
            weights = np.stack([out[i][1] for i in range(len(out))]).sum(0)
            patchcount = np.stack([out[i][2] for i in range(len(out))]).sum(0)
            weights = weights / patchcount
            weights = np.median(weights) / weights
        else:
            out = np.vstack(list(tqdm.tqdm(pool.imap(func, range(0, int(self.numel.prod(axis=1).sum()))), total=int(self.numel.prod(axis=1).sum()))))
            weights = options.weights
            order = np.stack([out[i][0] for i in range(len(out))])
            filename = np.stack([out[i][1] for i in range(len(out))])
            filename = filename[~np.isnan(order).any(axis=1)]
            order = order[~np.isnan(order).any(axis=1)]
        return order, weights, filename

    @staticmethod
    def par_recalc(filenameA, order, filename, n_classes, output_area, prunned_nodes, itr):
        weight = np.zeros((n_classes,))
        patch_flags = np.zeros((n_classes,))
        # out = False
        z = order[itr, 0]
        y = order[itr, 1]
        x = order[itr, 2]
        ex_data = torch.from_numpy(DataTemplate.getWindow(filenameA[filename[itr]],
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

    def recalc(self, options, order, filename):
        func = partial(DataTemplate.par_recalc, self.filenameA, order, filename, options.n_classes, self.output_area, self.prunned_nodes)
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
        if self.mode == 'train':
            data = self.batchTrain
        elif self.mode == 'val':
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
        filename = int(data[idx, 4])
        if self.normalize:
            torch_dataI[:, :, :] = Data.normalize(torch.from_numpy(DataTemplate.rotate(DataTemplate.getWindow(self.filenameI[filename],
                                                                                                              z,
                                                                                                              y,
                                                                                                              x,
                                                                                                              self.input_size[0],
                                                                                                              self.input_size[1],
                                                                                                              self.input_size[2],
                                                                                                              self.pad[0, filename]), rtst).copy()))
        else:
            torch_dataI[:, :, :] = torch.from_numpy(DataTemplate.rotate(DataTemplate.getWindow(self.filenameI[filename],
                                                                                               z,
                                                                                               y,
                                                                                               x,
                                                                                               self.input_size[0],
                                                                                               self.input_size[1],
                                                                                               self.input_size[2],
                                                                                               self.pad[0, filename]), rtst).copy())
        # self.dataI[z:z+self.input_size[0], y:y+self.input_size[1], x:x+self.input_size[2]])
        torch_dataA[:, :] = torch.from_numpy(DataTemplate.rotate(DataTemplate.getWindow(self.filenameA[filename],
                                                                                        z + self.output_area[0, 0],
                                                                                        y + self.output_area[0, 1],
                                                                                        x + self.output_area[0, 2],
                                                                                        self.output_area[1, 0],
                                                                                        self.output_area[1, 1],
                                                                                        self.output_area[1, 2],
                                                                                        0)[0, :, :], rtst).copy())
        if self.criterion == 'Mixed':
            torch_dataA2[:, :] = torch.from_numpy(DataTemplate.rotate(DataTemplate.getWindow(self.filenameIA[filename],
                                                                                               z + self.output_area[0, 0],
                                                                                               y + self.output_area[0, 1],
                                                                                               x + self.output_area[0, 2],
                                                                                               self.output_area[1, 0],
                                                                                               self.output_area[1, 1],
                                                                                               self.output_area[1, 2],
                                                                                               self.pad[1, filename])[0, :, :], rtst).copy())
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
        if self.mode == 'train':
            return self.num_train
        elif self.mode == 'val':
            return self.num_val
        else:
            return self.num_test


class Data3D(DataTemplate):
    def __getitem__(self, idx):
        # Returns a batch for each epoch
        # Edited
        if self.mode == 'train':
            data = self.batchTrain
        elif self.mode == 'val':
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
        filename = int(data[idx, 4])
        if self.normalize:
            torch_dataI[0, :, :, :] = DataTemplate.normalize(torch.from_numpy(DataTemplate.rotate(DataTemplate.getWindow(self.filenameI[filename],
                                                                                                                         z,
                                                                                                                         y,
                                                                                                                         x,
                                                                                                                         self.input_size[0],
                                                                                                                         self.input_size[1],
                                                                                                                         self.input_size[2],
                                                                                                                         self.pad[0, filename]), rtst).copy()))
        else:
            torch_dataI[0, :, :, :] = torch.from_numpy(DataTemplate.rotate(DataTemplate.getWindow(self.filenameI[filename],
                                                                                                  z,
                                                                                                  y,
                                                                                                  x,
                                                                                                  self.input_size[0],
                                                                                                  self.input_size[1],
                                                                                                  self.input_size[2],
                                                                                                  self.pad[0, filename]), rtst).copy())
        # self.dataI[z:z+self.input_size[0], y:y+self.input_size[1], x:x+self.input_size[2]])
        torch_dataA[:, :, :] = torch.from_numpy(DataTemplate.rotate(DataTemplate.getWindow(self.filenameA[filename],
                                                                                           z + self.output_area[0, 0],
                                                                                           y + self.output_area[0, 1],
                                                                                           x + self.output_area[0, 2],
                                                                                           self.output_area[1, 0],
                                                                                           self.output_area[1, 1],
                                                                                           self.output_area[1, 2],
                                                                                           0), rtst).copy())
        if self.criterion == 'Mixed':
            torch_dataA2[:, :, :] = torch.from_numpy(DataTemplate.rotate(DataTemplate.getWindow(self.filenameIA[filename],
                                                                                               z + self.output_area[0, 0],
                                                                                               y + self.output_area[0, 1],
                                                                                               x + self.output_area[0, 2],
                                                                                               self.output_area[1, 0],
                                                                                               self.output_area[1, 1],
                                                                                               self.output_area[1, 2],
                                                                                               self.pad[1, filename]), rtst).copy())
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
        if self.mode == 'train':
            return self.num_train
        elif self.mode == 'val':
            return self.num_val
        else:
            return self.num_test


class Data_fast_template(DataTemplate):
    def load(self, options):
        fileA = h5py.File(self.filenameA, 'r')
        dataA = fileA.get(list(fileA.keys())[0].encode('ascii', 'ignore'))
        while isinstance(dataA, h5py.Group):
            dataA = dataA.get(list(dataA.keys())[0].encode('ascii', 'ignore'))
        fileI = h5py.File(self.filenameI, 'r')
        dataI = fileI.get(list(fileI.keys())[0].encode('ascii', 'ignore'))
        while isinstance(dataI, h5py.Group):
            dataI = dataI.get(list(dataI.keys())[0].encode('ascii', 'ignore'))
        self.dataAnnotation = dataA[self.start_points[0]:self.end_points[0], self.start_points[1]:self.end_points[1], self.start_points[2]:self.end_points[2]]
        self.dataInput = dataI[self.start_points[0]:self.end_points[0], self.start_points[1]:self.end_points[1], self.start_points[2]:self.end_points[2]]
        fileA.close()
        fileI.close()
        return

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
        if self.mode == 'train':
            data = self.batchTrain
        elif self.mode == 'val':
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
        if self.mode == 'train':
            return self.num_train
        elif self.mode == 'val':
            return self.num_val
        else:
            return self.num_test


class Data3D_fast(Data_fast_template):
    def __getitem__(self, idx):
        # Returns a batch for each epoch
        # Edited
        if self.mode == 'train':
            data = self.batchTrain
        elif self.mode == 'val':
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
        if self.mode == 'train':
            return self.num_train
        elif self.mode == 'val':
            return self.num_val
        else:
            return self.num_test


class Data_infer(Dataset):
    def __init__(self, options):
        super(Data_infer, self).__init__()
        self.input_size = options.input_size
        self.output_area = options.output_area
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
        self.order = self.preprocess(options)
        self.temp_stor = np.zeros((options.n_classes, self.output_area[1, 0], dataI.shape[1], dataI.shape[2]))
        self.delim = np.zeros((options.n_classes, self.output_area[1, 0], dataI.shape[1], dataI.shape[2]))
        self.output_size = dataI.shape
        self.save_point = 0
        fileI.close()

    @staticmethod
    def getWindow(data, z, y, x, depth, height, width, pad):
        window = np.ones((depth, height, width)) * pad
        start = np.array([z, y, x])
        startNew = np.copy(start)
        size = np.array([depth, height, width])
        end = start + size
        endNew = np.copy(end)
        for i in range(0, 3):
            if start[i] < 0:
                startNew[i] = 0
            if end[i] > data.shape[i]:
                endNew[i] = data.shape[i]
        extr = data[int(startNew[0]):int(endNew[0]), int(startNew[1]):int(endNew[1]), int(startNew[2]):int(endNew[2])]
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
    def normalize(window):
        avg = window.sum() / window.numel()
        std = torch.std(window)
        result = (window - avg) / (4 * std)
        return result

    def preprocess(self, options):
        order = np.mgrid[0:self.numel[0], 0:self.numel[1], 0:self.numel[2]]
        order = np.rollaxis(order, 0, 4)
        order = order.reshape((self.numel[0] * self.numel[1] * self.numel[2], 3))
        order[:, 0] = order[:, 0] * options.input_stride[0] + self.start_points[0]
        order[:, 1] = order[:, 1] * options.input_stride[1] + self.start_points[1]
        order[:, 2] = order[:, 2] * options.input_stride[2] + self.start_points[2]
        return order

    def save(self, y, idx, options):
        for i in range(0, len(idx)):
            if self.order[idx[i], 1] == self.start_points[1] and self.order[idx[i], 2] == self.start_points[2]:
                if self.order[idx[i], 0] != self.start_points[0]:
                    if isfile(self.filenameA):
                        fileA = h5py.File(self.filenameA, 'a', libver='latest', swmr=True)
                        dataA = fileA.get(list(fileA.keys())[0].encode('ascii', 'ignore'))
                    else:
                        fileA = h5py.File(self.filenameA, 'w', libver='latest', swmr=True)
                        dataA = fileA.create_dataset('/data', tuple(self.output_size), chunks=True)
                    output = self.temp_stor[:, :options.input_stride[0], :, :] / self.delim[:, :options.input_stride[0], :, :]
                    output = np.argmax(output, axis=0)
                    dataA[(self.save_point + self.output_area[0, 0]):(self.save_point + self.output_area[0, 0] + options.input_stride[0]), :, :] = output
                    self.temp_stor = np.pad(self.temp_stor, ((0, 0), (0, options.input_stride[0]), (0, 0), (0, 0)), mode='constant')[:, options.input_stride[0]:, :, :]
                    self.delim = np.pad(self.delim, ((0, 0), (0, options.input_stride[0]), (0, 0), (0, 0)), mode='constant')[:, options.input_stride[0]:, :, :]
                    fileA.close()
                self.save_point = self.order[idx[i], 0]
            if options.gpu:
                self.temp_stor[:,
                0,
                (self.order[idx[i], 1] + self.output_area[0, 1]):(self.order[idx[i], 1] + self.output_area[0, 1] + self.output_area[1, 1]),
                (self.order[idx[i], 2] + self.output_area[0, 2]):(self.order[idx[i], 2] + self.output_area[0, 2] + self.output_area[1, 2])] += y[i, :, :, :].cpu().detach().numpy()
            else:
                self.temp_stor[:,
                0,
                (self.order[idx[i], 1] + self.output_area[0, 1]):(self.order[idx[i], 1] + self.output_area[0, 1] + self.output_area[1, 1]),
                (self.order[idx[i], 2] + self.output_area[0, 2]):(self.order[idx[i], 2] + self.output_area[0, 2] + self.output_area[1, 2])] += y[i, :, :, :].detach().numpy()
            self.delim[:,
            0,
            (self.order[idx[i], 1] + self.output_area[0, 1]):(self.order[idx[i], 1] + self.output_area[0, 1] + self.output_area[1, 1]),
            (self.order[idx[i], 2] + self.output_area[0, 2]):(self.order[idx[i], 2] + self.output_area[0, 2] + self.output_area[1, 2])] += np.ones((y.shape[1], y.shape[2], y.shape[3]))

    def __getitem__(self, idx):
        # Returns a batch for each epoch
        # Edited
        fileI = h5py.File(self.filenameI, 'r', libver='latest', swmr=True)
        dataI = fileI.get(list(fileI.keys())[0].encode('ascii', 'ignore'))
        while isinstance(dataI, h5py.Group):
            dataI = dataI.get(list(dataI.keys())[0].encode('ascii', 'ignore'))
        torch_dataI = torch.empty(int(self.input_size[0]),
                                  int(self.input_size[1]),
                                  int(self.input_size[2]))
        z = self.order[idx, 0]
        y = self.order[idx, 1]
        x = self.order[idx, 2]
        if self.normalize:
            torch_dataI[:, :, :] = Data_infer.normalize(torch.from_numpy(Data_infer.getWindow(dataI,
                                                                                              z,
                                                                                              y,
                                                                                              x,
                                                                                              self.input_size[0],
                                                                                              self.input_size[1],
                                                                                              self.input_size[2],
                                                                                              self.pad).copy()))
        else:
            torch_dataI[:, :, :] = torch.from_numpy(Data_infer.getWindow(dataI,
                                                                         z,
                                                                         y,
                                                                         x,
                                                                         self.input_size[0],
                                                                         self.input_size[1],
                                                                         self.input_size[2],
                                                                         self.pad).copy())
        fileI.close()
        return torch_dataI, idx

    def __len__(self):
        return len(self.order)


class Dataset3D_infer(Dataset):
    def __init__(self, options):
        super(Dataset3D_infer, self).__init__()
        self.input_size = options.input_size
        self.output_area = options.output_area
        self.filenameI = options.input_filename[0]
        self.filenameA = options.annotations_filename[0]
        self.filenameIA = options.intermediate_filename[0]
        self.normalize = options.normalize
        fileI = h5py.File(self.filenameI, 'r')
        dataI = fileI.get(list(fileI.keys())[0].encode('ascii', 'ignore'))
        while isinstance(dataI, h5py.Group):
            dataI = dataI.get(list(dataI.keys())[0].encode('ascii', 'ignore'))
        self.data_size = options.input_area[0, 1, :] - options.input_area[0, 0, :]
        self.excess = np.array([np.mod((self.data_size[0] - self.input_size[0]), options.input_stride[0]),
                                np.mod((self.data_size[1] - self.input_size[1]), options.input_stride[1]),
                                np.mod((self.data_size[2] - self.input_size[2]), options.input_stride[2])])
        self.start_points = np.floor(self.excess / 2).astype(int) + options.input_area[0, 0, :]
        self.numel = np.array([((self.data_size[0] - self.input_size[0]) // options.input_stride[0]) + 1,
                               ((self.data_size[1] - self.input_size[1]) // options.input_stride[1]) + 1,
                               ((self.data_size[2] - self.input_size[2]) // options.input_stride[2]) + 1])
        self.pad = dataI[0, 0, 0]
        self.array_size = np.array([((self.numel[0] - 1) * options.input_stride[0]) + self.input_size[0],
                                    ((self.numel[1] - 1) * options.input_stride[1]) + self.input_size[1],
                                    ((self.numel[2] - 1) * options.input_stride[2]) + self.input_size[2]])
        self.order = self.preprocess(options)
        self.temp_stor = np.zeros((options.n_classes, self.output_area[1, 0], self.array_size[1], self.array_size[2]))
        self.temp_stor2 = np.zeros((1, self.output_area[1, 0], self.array_size[1], self.array_size[2]))
        self.delim = np.zeros((options.n_classes, self.output_area[1, 0], self.array_size[1], self.array_size[2]))
        self.delim2 = np.zeros((1, self.output_area[1, 0], self.array_size[1], self.array_size[2]))
        self.output_size = (options.n_classes,) + dataI.shape
        self.save_point = self.order[0, 0]
        fileI.close()

    @staticmethod
    def getWindow(data, z, y, x, depth, height, width, pad):
        window = np.ones((depth, height, width)) * pad
        start = np.array([z, y, x])
        startNew = np.copy(start)
        size = np.array([depth, height, width])
        end = start + size
        endNew = np.copy(end)
        for i in range(0, 3):
            if start[i] < 0:
                startNew[i] = 0
            if end[i] > data.shape[i]:
                endNew[i] = data.shape[i]
        extr = data[int(startNew[0]):int(endNew[0]), int(startNew[1]):int(endNew[1]), int(startNew[2]):int(endNew[2])]
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
    def normalize(window):
        avg = window.sum() / window.numel()
        std = torch.std(window)
        result = (window - avg) / (4 * std)
        return result

    def preprocess(self, options):
        order = np.mgrid[0:self.numel[0], 0:self.numel[1], 0:self.numel[2]]
        order = np.rollaxis(order, 0, 4)
        order = order.reshape((self.numel[0] * self.numel[1] * self.numel[2], 3))
        order[:, 0] = order[:, 0] * options.input_stride[0] + self.start_points[0]
        order[:, 1] = order[:, 1] * options.input_stride[1] + self.start_points[1]
        order[:, 2] = order[:, 2] * options.input_stride[2] + self.start_points[2]
        return order

    def save(self, y, yd, idx, options):
        for i in range(0, len(idx)):
            if options.gpu:
                self.temp_stor[:,
                :,
                (self.order[idx[i], 1] + self.output_area[0, 1] - self.start_points[1]):(self.order[idx[i], 1] + self.output_area[0, 1] + self.output_area[1, 1] - self.start_points[1]),
                (self.order[idx[i], 2] + self.output_area[0, 2] - self.start_points[2]):(self.order[idx[i], 2] + self.output_area[0, 2] + self.output_area[1, 2] - self.start_points[2])] += y[i, :, :, :, :].cpu().detach().numpy()
            else:
                self.temp_stor[:,
                :,
                (self.order[idx[i], 1] + self.output_area[0, 1] - self.start_points[1]):(self.order[idx[i], 1] + self.output_area[0, 1] + self.output_area[1, 1] - self.start_points[1]),
                (self.order[idx[i], 2] + self.output_area[0, 2] - self.start_points[2]):(self.order[idx[i], 2] + self.output_area[0, 2] + self.output_area[1, 2] - self.start_points[2])] += y[i, :, :, :, :].detach().numpy()
            self.delim[:,
            :,
            (self.order[idx[i], 1] + self.output_area[0, 1] - self.start_points[1]):(self.order[idx[i], 1] + self.output_area[0, 1] + self.output_area[1, 1] - self.start_points[1]),
            (self.order[idx[i], 2] + self.output_area[0, 2] - self.start_points[2]):(self.order[idx[i], 2] + self.output_area[0, 2] + self.output_area[1, 2] - self.start_points[2])] += np.ones((y.shape[1], y.shape[2], y.shape[3], y.shape[4]))
            '''
            if options.gpu:
                self.temp_stor2[:,
                :,
                (self.order[idx[i], 1] + self.output_area[0, 1] - self.start_points[1]):(self.order[idx[i], 1] + self.output_area[0, 1] + self.output_area[1, 1] - self.start_points[1]),
                (self.order[idx[i], 2] + self.output_area[0, 2] - self.start_points[2]):(self.order[idx[i], 2] + self.output_area[0, 2] + self.output_area[1, 2] - self.start_points[2])] += yd[i, :, :, :, :].cpu().detach().numpy()
            else:
                self.temp_stor2[:,
                :,
                (self.order[idx[i], 1] + self.output_area[0, 1] - self.start_points[1]):(self.order[idx[i], 1] + self.output_area[0, 1] + self.output_area[1, 1] - self.start_points[1]),
                (self.order[idx[i], 2] + self.output_area[0, 2] - self.start_points[2]):(self.order[idx[i], 2] + self.output_area[0, 2] + self.output_area[1, 2] - self.start_points[2])] += yd[i, :, :, :, :].detach().numpy()
            self.delim2[:,
            :,
            (self.order[idx[i], 1] + self.output_area[0, 1] - self.start_points[1]):(self.order[idx[i], 1] + self.output_area[0, 1] + self.output_area[1, 1] - self.start_points[1]),
            (self.order[idx[i], 2] + self.output_area[0, 2] - self.start_points[2]):(self.order[idx[i], 2] + self.output_area[0, 2] + self.output_area[1, 2] - self.start_points[2])] += np.ones((yd.shape[1], yd.shape[2], yd.shape[3], yd.shape[4]))
            '''
            if np.count_nonzero(self.delim) - self.delim.size == 0:  # self.order[idx[i], 1] == self.start_points[1] and self.order[idx[i], 2] == self.start_points[2]:
                # if self.order[idx[i], 0] != self.start_points[0]:
                if isfile(self.filenameA):
                    fileA = h5py.File(self.filenameA, 'a', libver='latest', swmr=True)
                    dataA = fileA.get(list(fileA.keys())[0].encode('ascii', 'ignore'))
                else:
                    fileA = h5py.File(self.filenameA, 'w', libver='latest', swmr=True)
                    dataA = fileA.create_dataset('/data', tuple(self.output_size), chunks=True)
                output = self.temp_stor[:, :options.input_stride[0], :, :] / self.delim[:, :options.input_stride[0], :, :]
                # output = np.argmax(output, axis=0)
                dataA[:, (self.save_point + self.output_area[0, 0]):(self.save_point + self.output_area[0, 0] + options.input_stride[0]), self.start_points[1]:(output.shape[2]+self.start_points[1]), self.start_points[2]:(output.shape[3]+self.start_points[2])] = output
                self.temp_stor = np.pad(self.temp_stor, ((0, 0), (0, options.input_stride[0]), (0, 0), (0, 0)), mode='constant')[:, options.input_stride[0]:, :, :]
                self.delim = np.pad(self.delim, ((0, 0), (0, options.input_stride[0]), (0, 0), (0, 0)), mode='constant')[:, options.input_stride[0]:, :, :]
                fileA.close()
                '''
                if isfile(self.filenameIA):
                    fileIA = h5py.File(self.filenameIA, 'a', libver='latest', swmr=True)
                    dataIA = fileIA.get(list(fileIA.keys())[0].encode('ascii', 'ignore'))
                else:
                    fileIA = h5py.File(self.filenameIA, 'w', libver='latest', swmr=True)
                    dataIA = fileIA.create_dataset('/data', tuple(self.output_size), chunks=True)
                output = self.temp_stor2[:, :options.input_stride[0], :, :] / self.delim2[:, :options.input_stride[0], :, :]
                output = output.squeeze()
                dataIA[(self.save_point + self.output_area[0, 0]):(self.save_point + self.output_area[0, 0] + options.input_stride[0]), self.start_points[1]:(output.shape[1]+self.start_points[1]), self.start_points[2]:(output.shape[2]+self.start_points[2])] = output
                self.temp_stor2 = np.pad(self.temp_stor2, ((0, 0), (0, options.input_stride[0]), (0, 0), (0, 0)), mode='constant')[:, options.input_stride[0]:, :, :]
                self.delim2 = np.pad(self.delim2, ((0, 0), (0, options.input_stride[0]), (0, 0), (0, 0)), mode='constant')[:, options.input_stride[0]:, :, :]
                fileIA.close()
                '''
                self.save_point = self.save_point + options.input_stride[0]

    def end_save(self, options):
        if isfile(self.filenameA):
            fileA = h5py.File(self.filenameA, 'a', libver='latest', swmr=True)
            dataA = fileA.get(list(fileA.keys())[0].encode('ascii', 'ignore'))
        else:
            fileA = h5py.File(self.filenameA, 'w', libver='latest', swmr=True)
            dataA = fileA.create_dataset('/data', tuple(self.output_size), chunks=True)
        sz = self.temp_stor[:, :-options.input_stride[0], :, :].shape[1]
        output = self.temp_stor[:, :-options.input_stride[0], :, :] / self.delim[:, :-options.input_stride[0], :, :]
        # output = np.argmax(output, axis=0)
        dataA[:, (self.save_point + self.output_area[0, 0]):(self.save_point + self.output_area[0, 0] + sz), self.start_points[1]:(output.shape[2]+self.start_points[1]), self.start_points[2]:(output.shape[3]+self.start_points[2])] = output
        fileA.close()
        '''
        if isfile(self.filenameIA):
            fileIA = h5py.File(self.filenameIA, 'a', libver='latest', swmr=True)
            dataIA = fileIA.get(list(fileIA.keys())[0].encode('ascii', 'ignore'))
        else:
            fileIA = h5py.File(self.filenameIA, 'w', libver='latest', swmr=True)
            dataIA = fileA.create_dataset('/data', tuple(self.output_size), chunks=True)
        sz = self.temp_stor2[:, :-options.input_stride[0], :, :].shape[1]
        output = self.temp_stor2[:, :-options.input_stride[0], :, :] / self.delim2[:, :-options.input_stride[0], :, :]
        output = output.squeeze()
        dataIA[(self.save_point + self.output_area[0, 0]):(self.save_point + self.output_area[0, 0] + sz), self.start_points[1]:(output.shape[1]++self.start_points[1]), self.start_points[2]:(output.shape[2]++self.start_points[2])] = output
        fileIA.close()
        '''

    def __getitem__(self, idx):
        # Returns a batch for each epoch
        # Edited
        fileI = h5py.File(self.filenameI, 'r', libver='latest', swmr=True)
        dataI = fileI.get(list(fileI.keys())[0].encode('ascii', 'ignore'))
        while isinstance(dataI, h5py.Group):
            dataI = dataI.get(list(dataI.keys())[0].encode('ascii', 'ignore'))
        data = self.order
        torch_dataI = torch.empty(1,
                                  int(self.input_size[0]),
                                  int(self.input_size[1]),
                                  int(self.input_size[2]))
        z = data[idx, 0]
        y = data[idx, 1]
        x = data[idx, 2]
        if self.normalize:
            torch_dataI[0, :, :, :] = Dataset3D_infer.normalize(torch.from_numpy(Dataset3D_infer.getWindow(dataI,
                                                                                                     z,
                                                                                                     y,
                                                                                                     x,
                                                                                                     self.input_size[0],
                                                                                                     self.input_size[1],
                                                                                                     self.input_size[2],
                                                                                                     self.pad).copy()))
        else:
            torch_dataI[0, :, :, :] = torch.from_numpy(Dataset3D_infer.getWindow(dataI,
                                                                              z,
                                                                              y,
                                                                              x,
                                                                              self.input_size[0],
                                                                              self.input_size[1],
                                                                              self.input_size[2],
                                                                              self.pad).copy())
        fileI.close()
        return torch_dataI, idx

    def __len__(self):
        return len(self.order)


class DataloaderDeNo(object):
    def __init__(self, options):
        super(DataloaderDeNo, self).__init__()
        self.fileI = h5py.File(options.input_filename, 'r')
        self.fileI2 = h5py.File(options.input_filename2, 'r')
        self.fileA = h5py.File(options.annotations_filename, 'r')
        self.dataI = self.fileI.get(list(self.fileI.keys())[0].encode('ascii', 'ignore'))
        self.dataI2 = self.fileI.get(list(self.fileI2.keys())[0].encode('ascii', 'ignore'))
        self.dataA = self.fileA.get(list(self.fileA.keys())[0].encode('ascii', 'ignore'))
        self.data_size = options.input_area[1, :] - options.input_area[0, :]
        self.excess = np.array([np.mod((self.data_size[0] - options.input_size[0]), options.input_stride[0]),
                                np.mod((self.data_size[1] - options.input_size[1]), options.input_stride[1]),
                                np.mod((self.data_size[2] - options.input_size[2]), options.input_stride[2])])
        self.start_points = np.floor(self.excess / 2).astype(int) + options.input_area[0, :]
        self.numel = np.array([((self.data_size[0] - options.input_size[0]) // options.input_stride[0]) + 1,
                               ((self.data_size[1] - options.input_size[1]) // options.input_stride[1]) + 1,
                               ((self.data_size[2] - options.input_size[2]) // options.input_stride[2]) + 1])
        self.pad = self.dataI[0, 0, 0]
        if isfile(options.cp_dest + 'dataloader2.npz') and options.clean_go is False:
            files = np.load(options.cp_dest + 'dataloader2.npz')
            self.order = files['order']
            perm = files['perm']
            rotstate = files['rotstate']
            files.close()
        else:
            self.order = self.preprocess(options)
            perm = np.random.permutation(self.order.shape[0])
            rotstate = np.random.randint(8, size=self.order.shape[0])
        num_train = (np.floor((1 - options.val_precentage - options.test_precentage) * len(perm))).astype(int)
        num_val = (np.floor(options.val_precentage * len(perm))).astype(int)
        num_test = len(perm) - num_val - num_train
        self.permTrain = perm[0:num_train]
        self.permVal = perm[num_train:(num_train + num_val)]
        self.permTest = perm[(num_train + num_val):len(perm)]
        # perm = permTrain[0:(len(permTrain)-np.mod(len(permTrain), options.batchsize).astype(int))]
        self.batchTrain = np.empty([np.ceil(num_train / float(options.batchsize)).astype(int), options.batchsize, 4])
        self.batchTrain[:] = np.nan
        self.batchVal = np.empty([np.ceil(num_val / float(options.batchsize)).astype(int), options.batchsize, 4])
        self.batchVal[:] = np.nan
        self.batchTest = np.empty([np.ceil(num_test / float(options.batchsize)).astype(int), options.batchsize, 4])
        self.batchTest[:] = np.nan
        for i in range(0, num_train):
            instance = np.mod(i, options.batchsize).astype(int)
            batch = i // options.batchsize
            self.batchTrain[batch, instance, 3] = rotstate[i]
            self.batchTrain[batch, instance, 2] = self.order[self.permTrain[i], 2]
            self.batchTrain[batch, instance, 1] = self.order[self.permTrain[i], 1]
            self.batchTrain[batch, instance, 0] = self.order[self.permTrain[i], 0]
        for i in range(0, num_val):
            instance = np.mod(i - num_train, options.batchsize).astype(int)
            batch = i // options.batchsize
            self.batchVal[batch, instance, 3] = rotstate[i + num_train]
            self.batchVal[batch, instance, 2] = self.order[self.permVal[i], 2]
            self.batchVal[batch, instance, 1] = self.order[self.permVal[i], 1]
            self.batchVal[batch, instance, 0] = self.order[self.permVal[i], 0]
        for i in range(0, num_test):
            instance = np.mod(i - num_train - num_val, options.batchsize).astype(int)
            batch = i // options.batchsize
            self.batchTest[batch, instance, 3] = rotstate[i + num_train + num_val]
            self.batchTest[batch, instance, 2] = self.order[self.permTest[i], 2]
            self.batchTest[batch, instance, 1] = self.order[self.permTest[i], 1]
            self.batchTest[batch, instance, 0] = self.order[self.permTest[i], 0]
        np.savez(options.cp_dest + 'dataloader2.npz', order=self.order, perm=perm, rotstate=rotstate)

    def shuffle(self, options):
        shperm = np.random.permutation(self.permTrain.shape[0])
        newrot = np.random.randint(8, size=self.permTrain.shape[0])
        for i in range(0, self.permTrain.shape[0]):
            instance = np.mod(i, options.batchsize).astype(int)
            batch = i // options.batchsize
            self.batchTrain[batch, instance, 3] = newrot[i]
            self.batchTrain[batch, instance, 2] = self.order[self.permTrain[shperm[i]], 2]
            self.batchTrain[batch, instance, 1] = self.order[self.permTrain[shperm[i]], 1]
            self.batchTrain[batch, instance, 0] = self.order[self.permTrain[shperm[i]], 0]

    @staticmethod
    def getWindow(data, z, y, x, depth, height, width, pad):
        window = np.ones((depth, height, width)) * pad
        start = np.array([z, y, x])
        startNew = np.copy(start)
        size = np.array([depth, height, width])
        end = start + size
        endNew = np.copy(end)
        for i in range(0, 3):
            if start[i] < 0:
                startNew[i] = 0
            if end[i] > data.shape[i]:
                endNew[i] = data.shape[i]
        extr = data[int(startNew[0]):int(endNew[0]), int(startNew[1]):int(endNew[1]), int(startNew[2]):int(endNew[2])]
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
    def progress(count, total, status=''):
        bar_len = 40
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
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
            return np.rot90(window, axes=(dim - 2, dim - 1))
        elif rtst == 4:
            return np.rot90(np.rot90(window, axes=(dim - 2, dim - 1)), axes=(dim - 2, dim - 1))
        elif rtst == 5:
            return np.rot90(np.rot90(np.rot90(window, axes=(dim - 2, dim - 1)), axes=(dim - 2, dim - 1)),
                            axes=(dim - 2, dim - 1))
        elif rtst == 6:
            return np.flip(np.rot90(window, axes=(dim - 2, dim - 1)), dim - 2)
        elif rtst == 7:
            return np.flip(np.rot90(window, axes=(dim - 2, dim - 1)), dim - 1)

    @staticmethod
    def normalize(window):
        avg = window.sum() / window.numel()
        std = torch.std(window)
        result = (window - avg) / (4 * std)
        return result

    def preprocess(self, options):
        order = np.empty((0, 3)).astype(int)
        count = 0
        total = self.numel[0] * self.numel[1] * self.numel[2]
        for i in range(0, self.numel[0]):
            vec = np.random.normal(0, 25, 2).astype(int)
            for j in range(0, self.numel[1]):
                for k in range(0, self.numel[2]):
                    x = k * options.input_stride[2] + vec[0] + self.start_points[2]
                    y = j * options.input_stride[1] + vec[1] + self.start_points[1]
                    z = i * options.input_stride[0] + self.start_points[0]
                    # out = False
                    ex_data = DataloaderDeNo.getWindow(self.dataA,
                                                       z + options.output_area[0, 0],
                                                       y + options.output_area[0, 1],
                                                       x + options.output_area[0, 2],
                                                       options.output_area[1, 0],
                                                       options.output_area[1, 1],
                                                       options.output_area[1, 2],
                                                       0)
                    # self.dataA[z+2, y:y+options.input_size[1], x:x+options.input_size[2]])
                    if (ex_data != 0).astype(int).sum() > 0.2 * ex_data.size:  # other class present except Air: 0
                        order = np.vstack([order, [z, y, x]])
                    DataloaderDeNo.progress(count, total, status='Data Preprocessing')
        sys.stdout.write("\n")
        return order

    def load(self, epoch, options, mode):
        # Returns a batch for each epoch
        # Edited
        if mode == 'train':
            data = self.batchTrain
        elif mode == 'val':
            data = self.batchVal
        else:
            data = self.batchTest
        batchsize = 0
        for i in range(0, options.batchsize):
            if not np.isnan(data[epoch, i, 0]):
                batchsize += 1
        torch_dataI = torch.empty(batchsize,
                                  int(options.input_size[0]),
                                  int(options.input_size[1]),
                                  int(options.input_size[2]))
        torch_dataA = torch.empty(batchsize,
                                  int(options.input_size[0]),
                                  int(options.input_size[1]),
                                  int(options.input_size[2]))
        for i in range(0, batchsize):
            z = data[epoch, i, 0]
            y = data[epoch, i, 1]
            x = data[epoch, i, 2]
            rtst = int(data[epoch, i, 3])
            if options.normalize:
                torch_dataI[i, :, :, :] = DataloaderDeNo.normalize(
                    torch.from_numpy(DataloaderDeNo.rotate(DataloaderDeNo.getWindow(self.dataI,
                                                                                    z,
                                                                                    y,
                                                                                    x,
                                                                                    options.input_size[0],
                                                                                    options.input_size[1],
                                                                                    options.input_size[2],
                                                                                    self.pad), rtst).copy()))
                torch_dataA[i, :, :, :] = DataloaderDeNo.normalize(
                    torch.from_numpy(DataloaderDeNo.rotate(DataloaderDeNo.getWindow(self.dataI2,
                                                                                    z,
                                                                                    y,
                                                                                    x,
                                                                                    options.input_size[0],
                                                                                    options.input_size[1],
                                                                                    options.input_size[2],
                                                                                    self.pad), rtst).copy()))
            else:
                torch_dataI[i, :, :, :] = torch.from_numpy(DataloaderDeNo.rotate(DataloaderDeNo.getWindow(self.dataI,
                                                                                                          z,
                                                                                                          y,
                                                                                                          x,
                                                                                                          options.input_size[
                                                                                                              0],
                                                                                                          options.input_size[
                                                                                                              1],
                                                                                                          options.input_size[
                                                                                                              2],
                                                                                                          self.pad),
                                                                                 rtst).copy())
                torch_dataA[i, :, :, :] = torch.from_numpy(DataloaderDeNo.rotate(DataloaderDeNo.getWindow(self.dataI2,
                                                                                                          z,
                                                                                                          y,
                                                                                                          x,
                                                                                                          options.input_size[
                                                                                                              0],
                                                                                                          options.input_size[
                                                                                                              1],
                                                                                                          options.input_size[
                                                                                                              2],
                                                                                                          self.pad),
                                                                                 rtst).copy())
            return torch_dataI, torch_dataA