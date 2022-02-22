import importlib
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from os.path import isfile, join
import torch.nn.functional as F
# from torch.autograd import Variable
from torch import optim
import numpy as np
import sys
import os
import gc
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
import shutil
import multiprocessing as mp
from functools import partial
import csv

'''
import utils.dataloader as dataload
from opts import Options
from opts2 import Options as Options2
from utils.logger import LoggerLite
from utils.timer import Timer
from validation import eval_net
from utils.accuracy import meas_net, meas_cm_weighted
from utils.data_vis import save_img
import models
'''


class PolyLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, iter_max, power=0.9, last_epoch=-1):
        self.iter_max = iter_max
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * ((1 - (self.last_epoch / self.iter_max)) ** self.power) for base_lr in self.base_lrs]


def par_3D_to_2D(x, options, itr):
    i = int(np.floor(itr / int(x.shape[2] - options.input_channels + 1)))
    j = itr - i * int(x.shape[2] - options.input_channels + 1)
    return x[i, :, j:j + options.input_channels, :, :]


def data_3D_to_2D(x, options, pool):
    func = partial(par_3D_to_2D, x, options)
    data = torch.cat(list(pool.map(func, range(0, x.shape[0] * (x.shape[2] - options.input_channels + 1)))), dim=0)
    return data


def calc_conf(tcm, n_classes, mask_class, gpu):
    accuracy = np.zeros((2 * (n_classes + 1),))
    for i in range(0, n_classes):
        if gpu:
            accuracy[i] = (tcm[i, i] / (tcm[:, i].sum() + tcm[i, :].sum() - tcm[i, i])).cpu().detach().numpy()
        else:
            accuracy[i] = (tcm[i, i] / (tcm[:, i].sum() + tcm[i, :].sum() - tcm[i, i])).detach().numpy()
    if mask_class > 0:
        accuracy[n_classes] = (accuracy[:mask_class].sum() + accuracy[mask_class + 1:n_classes].sum()) / (n_classes - 1)
    else:
        accuracy[n_classes] = accuracy[:n_classes].sum() / n_classes
    tp_tn = tcm.trace()
    for i in range(0, n_classes):
        if gpu:
            accuracy[i + n_classes + 1] = (tp_tn / (tp_tn + tcm[:, i].sum() + tcm[i, :].sum() - 2 * tcm[i, i])).cpu().detach().numpy()
        else:
            accuracy[i + n_classes + 1] = (tp_tn / (tp_tn + tcm[:, i].sum() + tcm[i, :].sum() - 2 * tcm[i, i])).detach().numpy()
    if mask_class > 0:
        accuracy[2 * n_classes + 1] = (accuracy[(n_classes + 1):(n_classes + 1 + mask_class)].sum() + accuracy[
                                                                                                      (n_classes + mask_class + 2):(2 * n_classes + 1)].sum()) / (
                                              n_classes - 1)
    else:
        accuracy[2 * n_classes + 1] = accuracy[(n_classes + 1):(2 * n_classes + 1)].sum() / n_classes
    return accuracy


def get_model(configuration, models):
    if configuration.down_samp == 'Convolution':
        down_samp = False
    else:
        down_samp = True

    if configuration.up_samp == 'UpConvolution':
        up_samp = False
    else:
        up_samp = True

    if configuration.model_type == 'DenseModel':
        network = models.DenseModel(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=32, Theta=0.5, Dropout=0, Growth_rate=configuration.gr)
    elif configuration.model_type == 'DenseModel5L':
        network = models.DenseModel5L(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=configuration.gr)
    elif configuration.model_type == 'Dense2DModel':
        network = models.Dense2DModel(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=configuration.gr)
    elif configuration.model_type == 'Dense2DModel5L':
        network = models.Dense2DModel5L(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=configuration.gr)
    elif configuration.model_type == 'DenseModel2_5':
        network = models.DenseModel2_5(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=configuration.gr)
    elif configuration.model_type == 'DenseModelHalfScope':
        network = models.DenseModelHalfScope(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=configuration.gr)
    elif configuration.model_type == 'DenseUSeg':
        # print(configuration.input_channels)
        network = models.DenseUSeg(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=configuration.gr)
    elif configuration.model_type == 'DenseUSeg2D':
        network = models.DenseUSeg2D(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=configuration.gr)
    elif configuration.model_type == 'DenseModelStack':
        network = models.DenseModelStack(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=configuration.gr)
    elif configuration.model_type == 'DenseModelStack_2D':
        network = models.DenseModelStack_2D(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=configuration.gr)
    elif configuration.model_type == 'DenseUSegStack':
        network = models.DenseUSegStack(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=configuration.gr)
    else:
        network = models.DenseUSegStack_2D(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=64, Theta=0.5, Dropout=0.2, Growth_rate=configuration.gr)

    return network


def train_net(net, options, prenet, options2, dataload):
    # dataload = importlib.import_module('utils.dataloader')
    Timer = getattr(importlib.import_module('utils.timer'), 'Timer')
    LoggerLite = getattr(importlib.import_module('utils.logger'), 'LoggerLite')
    eval_net = getattr(importlib.import_module('validation'), 'eval_net')
    meas_cm_weighted = getattr(importlib.import_module('utils.accuracy'), 'meas_cm_weighted')
    # save_img = getattr(importlib.import_module('utils.data_vis'), 'save_img')

    if options.CamVid:
        data = dataload.Data_Brats(options, 'train')
    elif options.dataSep:
        if options.data_3D:
            data = dataload.Data3DSep(options, 'train')
        else:
            data = dataload.DataSep(options, 'train')
    else:
        if options.data_3D:
            if options.input_fast:
                data = dataload.Data3D_fast(options, 'train')
            else:
                data = dataload.Data3D(options, 'train')
        else:
            if options.input_fast:
                data = dataload.Data_fast(options, 'train')
            else:
                data = dataload.Data(options, 'train')

    if options.prenet:
        if options2.CamVid:
            data2 = dataload.Data_Brats(options2, 'train', filenam='2D_slices_Real.npz')
        elif options2.dataSep:
            if options2.data_3D:
                data2 = dataload.Data3DSep(options2, 'train', filenam='2D_slices_Real.npz')
            else:
                data2 = dataload.DataSep(options2, 'train', filenam='2D_slices_Real.npz')
        else:
            if options2.data_3D:
                if options2.input_fast:
                    data2 = dataload.Data3D_fast(options2, 'train', filenam='2D_slices_Real.npz')
                else:
                    data2 = dataload.Data3D(options2, 'train', filenam='2D_slices_Real.npz')
            else:
                if options2.input_fast:
                    data2 = dataload.Data_fast(options2, 'train', filenam='2D_slices_Real.npz')
                else:
                    data2 = dataload.Data(options2, 'train', filenam='2D_slices_Real.npz')

    # hist = Logger(options.root+'/history', 'w')
    # val = Logger(options.root+'/validations', 'w')
    if options.testing_case:
        val = LoggerLite(options.root + '/validationsClasses', 'a')
        if options.dice != 'MSE':
            conf_matrV = LoggerLite(options.root + '/conf_matrixesV2', 'a')
        if options.prenet:
            val2 = LoggerLite(options.root + '/validationsClassesReal', 'a')
            if options.dice != 'MSE':
                conf_matrV2 = LoggerLite(options.root + '/conf_matrixesVReal', 'a')
    else:
        hist = LoggerLite(options.root + '/historyClasses', 'w')
        val = LoggerLite(options.root + '/validationsClasses', 'w')
        if options.dice != 'MSE':
            conf_matrH = LoggerLite(options.root + '/conf_matrixesH2', 'w')
            conf_matrV = LoggerLite(options.root + '/conf_matrixesV2', 'w')
        if options.prenet:
            val2 = LoggerLite(options.root + '/validationsClassesReal', 'w')
            if options.dice != 'MSE':
                conf_matrV2 = LoggerLite(options.root + '/conf_matrixesVReal', 'w')

    # hist.setNames(('instance', 'Accuracy', 'Time'))
    # val.setNames(('epoch', 'Accuracy', 'Time'))#
    n_classes = options.n_classes
    if options.mask_class > 0:
        n_classes += 1
    if options.dice != 'MSE':
        t = ('',)
        for k in range(0, n_classes):
            for l in range(0, n_classes):
                if l < k:
                    r = 'F_%d%d' % (k, l)
                elif l == k:
                    r = 'T_%d' % k
                else:
                    r = 'F_%d%d' % (k, l)
                if k + l == 0:
                    t = (r,)
                else:
                    t = t + (r,)
        if options.testing_case is False:
            conf_matrH.setNames(t)
            conf_matrV.setNames(t)
            if options.prenet:
                conf_matrV2.setNames(t)
    t = ('Epoch', 'Loss')
    if options.dice != 'MSE':
        for k in range(0, 2):
            for l in range(0, n_classes + 1):
                if k == 0:
                    if np.mod(l + 1, n_classes + 1) == 0:
                        r = 'mIoU'
                    else:
                        r = 'IoU_%d' % l
                else:
                    if np.mod(l + 1, n_classes + 1) == 0:
                        r = 'Global_Acc'
                    else:
                        r = 'Acc_%d' % l
                t = t + (r,)
    t = t + ('Time',)
    # hist.setNames(('Epoch', 'Acc', 'Time'))
    if options.testing_case is False:
        hist.setNames(t)
        val.setNames(t)
        if options.prenet:
            val2.setNames(t)

    if options.dice == 'Mixed':
        if options.testing_case:
            valD = LoggerLite(options.root + '/validationsClassesDenoise', 'a')
            if options.prenet:
                valD2 = LoggerLite(options.root + '/validationsClassesDenoiseReal', 'a')
        else:
            histD = LoggerLite(options.root + '/historyClassesDenoise', 'w')
            valD = LoggerLite(options.root + '/validationsClassesDenoise', 'w')
            t = ('Epoch', 'Loss', 'Time')
            histD.setNames(t)
            valD.setNames(t)
            if options.prenet:
                valD2 = LoggerLite(options.root + '/validationsClassesDenoiseReal', 'w')
                valD2.setNames(t)

    if options.optmethod is 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=options.lr, momentum=options.momentum)
    elif options.optmethod is 'rmsprop':
        optimizer = optim.RMSprop(net.parameters())
    else:
        if options.dice != 'Mixed':
            optimizer = optim.Adam(net.parameters(), lr=options.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=options.weig_dec)
        else:
            optimizer = optim.Adam([{'params': net.module.inc.parameters(), 'lr': options.lr * 10},
                                    {'params': net.module.D1.parameters(), 'lr': options.lr * 10},
                                    {'params': net.module.c1.parameters()},
                                    {'params': net.module.c2.parameters()},
                                    {'params': net.module.D2.parameters()}],
                                   lr=options.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=options.weig_dec)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=options.epochs / 2, gamma=0.1)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 40], gamma=0.1)
    # scheduler.last_epoch = 7
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)
    # scheduler = PolyLR(optimizer, options.epochs*instances)

    criterion2 = None

    if options.dice == 'MSE':
        criterion = nn.MSELoss()
    else:
        if options.weighted:
            if options.gpu:
                data.weights = data.weights.cuda()
            criterion = nn.CrossEntropyLoss(weight=data.weights,
                                            ignore_index=options.mask_class)
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=options.mask_class)
        if options.dice == 'Mixed':
            criterion2 = nn.MSELoss()
            if options.gpu:
                criterion2.cuda()

    if options.gpu:
        criterion.cuda()


    timer = Timer()
    epoch = 1
    counter_before_epoch = 0
    instances = int(np.ceil(data.__len__() / options.batchsize))
    itr = 0
    tcm = None
    tloss = None
    tlossD = None
    critical_value = 0
    idx = 0
    # scheduler = PolyLR(optimizer, options.epochs*instances)
    if options.prenet != 'None':
        pool = mp.Pool(processes=options.workers)
    else:
        pool = 'None'

    net.train()

    while epoch <= options.epochs and options.testing_case is False:
        trainDataloader = iter(DataLoader(dataset=data, batch_size=options.batchsize, num_workers=options.workers))
        for instance in range(0, instances):
            if epoch <= options.epochs:
                if (counter_before_epoch == 0) or (instance == 0):
                    sys.stdout.write('\n')
                    sys.stdout.write('Training')
                    sys.stdout.write('\n\n')

                if options.dice == 'Mixed':
                    x, y, y_de = next(trainDataloader)
                else:
                    x, y = next(trainDataloader)

                if prenet != 'None':
                    if options.data_3D == True and options2.data_3D == False:
                        batchsize = x.shape[0]
                        y_d = data_3D_to_2D(x, options2, pool)
                        y_d = y_d.cuda()

                if options.gpu:
                    x = x.cuda()
                    y = y.cuda()
                    if options.dice == 'Mixed':
                        y_de = y_de.cuda()

                # scheduler.step()

                optimizer.zero_grad()

                if prenet != 'None':
                    with torch.no_grad():
                        x = prenet(x)
                        if options.data_3D == True and options2.data_3D == False:
                            y_d = y_d.reshape(batchsize, 1, options.input_size[0], options.input_size[1], options.input_size[2])
                        # x = torch.cat([y_d, x[:,:,5:-5,:,:]], dim=1)
                        # x = torch.cat([y_d, x], dim=1)
                        # x = trans_data(y_d, x)

                if options.dice == 'Mixed':
                    y_pred, y_pred2 = net(x)
                    lossD = criterion2(y_pred2, y_de)
                    loss = criterion(y_pred, y)
                    loss = loss + 10 * lossD
                    # el = loss/lossD
                    # if el > 1:
                    #     el = np.round(el)
                    # else:
                    #     el = 1 / np.round(1 / el)
                    # loss = loss + el * lossD
                else:
                    y_pred = net(x)
                    loss = criterion(y_pred, y)

                loss.backward()

                optimizer.step()

                # acc = meas_net(y_pred, y, options.mask_class, options.gpu)
                # acc=cm[0,0]
                if options.dice != 'MSE':
                    cm = meas_cm_weighted(y_pred, y, n_classes, options.gpu)
                    if tloss is None:
                        tcm = cm
                        tloss = loss
                    else:
                        tcm += cm
                        tloss += loss
                    acc = (cm.trace() / cm.sum()) * 100
                    statement = '%3.2f' % acc
                    if options.dice == 'Mixed':
                        if tlossD is None:
                            tlossD = lossD
                        else:
                            tlossD += lossD
                else:
                    if tloss is None:
                        tloss = loss
                    else:
                        tloss += loss
                    acc = loss
                    statement = '%f' % acc

                # wr.add_scalar('%s/Train/accuracy' % options.name, acc, itr*instances+instance)

                if (counter_before_epoch == 0) or (instance == 0):
                    sys.stdout.write(("Minbatch: %d/%d,  " % (instance + 1, instances)) + ' Acc: ' + statement)
                else:
                    sys.stdout.write('\r')
                    sys.stdout.write(("Minbatch: %d/%d,  " % (instance + 1, instances)) + ' Acc: ' + statement)
                sys.stdout.flush()

                counter_before_epoch += 1
                if counter_before_epoch == options.val_freq:
                    if options.gpu:
                        tloss = tloss.cpu().detach().numpy() / options.val_freq
                    else:
                        tloss = tloss.detach().numpy() / options.val_freq
                    time = timer.get_value()
                    if options.dice != 'MSE':
                        accuracy = calc_conf(tcm, n_classes, options.mask_class)
                        if options.gpu:
                            conf_matrH.add(tcm.cpu().detach().numpy().flatten())
                        else:
                            conf_matrH.add(tcm.detach().numpy().flatten())
                        hist.add([epoch] + [tloss] + list(accuracy) + [time])
                        if options.dice == 'Mixed':
                            histD.add([epoch] + [tlossD] + [time])
                    else:
                        hist.add([epoch] + [tloss] + [time])

                    counter_before_epoch = 0
                    tcm, tloss, tlossD = eval_net(net, prenet, criterion, criterion2, options, options2, pool, data, ['Validation', 'val'], epoch)
                    torch.save(net.state_dict(), options.cp_dest + 'CP{}.pth'.format(epoch))

                    scheduler.step()

                    time = timer.get_value()
                    if options.dice != 'MSE':
                        accuracy = calc_conf(tcm, n_classes, options.mask_class, options.gpu)
                        if options.gpu:
                            conf_matrV.add(tcm.cpu().detach().numpy().flatten())
                        else:
                            conf_matrV.add(tcm.detach().numpy().flatten())
                        val.add([epoch] + [tloss] + list(accuracy) + [time])
                        if options.dice == 'Mixed':
                            valD.add([epoch] + [tlossD] + [time])
                        if critical_value is None:
                            critical_value = accuracy[n_classes]
                            idx = epoch
                        else:
                            if critical_value < accuracy[n_classes]:
                                critical_value = accuracy[n_classes]
                                idx = epoch
                    else:
                        val.add([epoch] + [tloss] + [time])
                        if critical_value is None:
                            critical_value = tloss
                            idx = epoch
                        else:
                            if critical_value > tloss:
                                critical_value = tloss
                                idx = epoch
                    # wr.add_scalar('%s/Validation/accuracy' % options.name, accuracy[2*n_classes+1], epoch)

                    if options.prenet:
                        tcm, tloss, tlossD = eval_net(net, prenet, criterion, criterion2, options2, options, pool, data2, ['Validation', 'val'], epoch)
                        if options.dice != 'MSE':
                            accuracy = calc_conf(tcm, n_classes, options.mask_class, options.gpu)
                            conf_matrV2.add(tcm.flatten())
                            val2.add([epoch] + [tloss] + list(accuracy) + [time])
                            if options.dice == 'Mixed':
                                valD2.add([epoch] + [tlossD] + [time])
                        else:
                            val2.add([epoch] + [tloss] + [time])

                    tcm = None
                    tloss = None
                    epoch += 1
            else:
                break
        itr += 1

    if options.testing_case:
        x = []
        with open(options.root + '/validationsClasses.csv', 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter='\t')
            rk = 0
            for row in plots:
                if rk == 1:
                    x.append(float(row[6]))
                else:
                    rk = 1
        idx = x.index(max(x)) + 1
        epoch = len(x) + 1
    if not isfile(options.cp_dest + 'Best_model.pth'):
        net.load_state_dict(torch.load(options.cp_dest + 'CP{}.pth'.format(idx)))
    else:
        net.load_state_dict(torch.load(options.cp_dest + 'Best_model.pth'))
    tcm, tloss, tlossD= eval_net(net, prenet, criterion, criterion2, options, options2, pool, data, ['Testing', 'test'], epoch)
    if not isfile(options.cp_dest + 'Best_model.pth'):
        torch.save(net.state_dict(), options.cp_dest + 'Best_model.pth')
    time = timer.get_value()
    if options.dice != 'MSE':
        accuracy = calc_conf(tcm, n_classes, options.mask_class)
        if options.gpu:
            conf_matrV.add(tcm.cpu().detach().numpy().flatten())
        else:
            conf_matrV.add(tcm.detach().numpy().flatten())
        val.add([epoch] + [tloss] + list(accuracy) + [time])
        if options.dice == 'Mixed':
            valD.add([epoch] + [tlossD] + [time])
    else:
        val.add([epoch] + [tloss] + [time])
    if options.prenet:
        tcm, tloss, tlossD = eval_net(net, prenet, criterion, criterion2, options2, options, pool, data2, ['Testing', 'test'], epoch)
        if options.dice != 'MSE':
            accuracy = calc_conf(tcm, n_classes, options.mask_class, options.gpu)
            if options.gpu:
                conf_matrV2.add(tcm.cpu().detach().numpy().flatten())
            else:
                conf_matrV2.add(tcm.cpu().detach().numpy().flatten())
            val2.add([epoch] + [tloss] + list(accuracy) + [time])
            if options.dice == 'Mixed':
                valD2.add([epoch] + [tlossD] + [time])
        else:
            val2.add([epoch] + [tloss] + [time])

    return


if __name__ == '__main__':

    diname = sys.argv[0].split("/")[-1]
    opname = 'opts%s' % diname[5:]
    daname = 'dataloader%s' % diname[5:]
    opname = opname[:-3]
    daname = daname[:-3]

    Options = getattr(importlib.import_module(opname), 'Options')
    dataload = importlib.import_module(daname)

    parser = Options()
    (configuration, args) = parser.parse_args()

    if configuration.prenet:
        opname = sys.argv[0].split("/")[-1]
        opname = 'opts2%s' % opname[5:]
        opname = opname[:-3]

        Options2 = getattr(importlib.import_module(opname), 'Options')

        parser = Options2()
        (configuration2, args) = parser.parse_args()

    sys.path.append('/home/user/Code_Folder/SemSegOldDiamondX')

    models = importlib.import_module('models')

    torch.manual_seed(configuration.manual_seed)

    tmp = configuration.input_filename.split(",")
    configuration.input_filename = np.array(tmp)

    tmp = configuration.annotations_filename.split(",")
    configuration.annotations_filename = np.array(tmp)

    tmp = configuration.intermediate_filename.split(",")
    configuration.intermediate_filename = np.array(tmp)

    tmp = configuration.input_size.split(",")
    configuration.input_size = [int(x.strip()) for x in tmp]
    configuration.input_size = np.array(configuration.input_size)

    tmp = configuration.input_stride.split(",")
    configuration.input_stride = [int(x.strip()) for x in tmp]
    configuration.input_stride = np.array(configuration.input_stride)

    if configuration.weights != 'None':
        tmp = configuration.weights.split(",")
        configuration.weights = [float(x.strip()) for x in tmp]
        configuration.weights = np.array(configuration.weights)
    else:
        configuration.weights = np.array([-1])

    if configuration.lcn != 'None':
        tmp = configuration.lcn.split(",")
        configuration.lcn = [int(x.strip()) for x in tmp]
        configuration.lcn = np.array(configuration.lcn)
    else:
        configuration.lcn = np.array([-1])

    tmp = configuration.input_area.split(",")
    configuration.input_area = [int(x.strip()) for x in tmp]
    for i in range(0, (int(len(configuration.input_area) / 6))):
        tmp = np.vstack((configuration.input_area[i * 6:i * 6 + 3], configuration.input_area[i * 6 + 3:i * 6 + 6]))
        if i == 0:
            tmp2 = np.expand_dims(tmp, axis=0)
        else:
            tmp2 = np.append(tmp2, np.expand_dims(tmp, axis=0), axis=0)
    configuration.input_area = tmp2

    tmp = configuration.output_area.split(",")
    configuration.output_area = [int(x.strip()) for x in tmp]
    configuration.output_area = np.vstack((configuration.output_area[0:3], configuration.output_area[3:6]))

    if configuration.prunned_classes != 'None':
        tmp = configuration.prunned_classes.split(",")
        configuration.prunned_classes = [int(x.strip()) for x in tmp]
        configuration.prunned_classes = np.vstack((configuration.prunned_classes[0:len(configuration.prunned_classes):2], configuration.prunned_classes[1:len(configuration.prunned_classes):2]))
    else:
        configuration.prunned_classes = np.array([[-1], [-1]])

    configuration.root = configuration.root + configuration.name
    configuration.cp_dest = configuration.root + configuration.cp_dest
    configuration.im_dest = configuration.root + configuration.im_dest
    configuration.tb_dest = configuration.root + configuration.tb_dest

    network = get_model(configuration, models)

    # writer = SummaryWriter(options.tb_dest)
    # dummy_input = torch.rand((1, 5, 512, 512))
    # writer.add_graph(net, (dummy_input,))

    if configuration.gpu:
        network.cuda()
        cudnn.benchmark = not configuration.deterministic
        cudnn.deterministic = configuration.deterministic
        # os.environ['CUDA_VISIBLE_DEVICES'] = configuration.gpu_devices
        if configuration.parallel:
            network = torch.nn.DataParallel(network, device_ids=list(range(torch.cuda.device_count()))[:4], output_device=list(range(torch.cuda.device_count()))[0])

    if configuration.load != '0':
        network.load_state_dict(torch.load(configuration.load))
        print('Model loaded from {}'.format(configuration.load))

    if configuration.prenet:
        tmp = configuration2.input_filename.split(",")
        configuration2.input_filename = np.array(tmp)

        tmp = configuration2.annotations_filename.split(",")
        configuration2.annotations_filename = np.array(tmp)

        tmp = configuration2.intermediate_filename.split(",")
        configuration2.intermediate_filename = np.array(tmp)

        tmp = configuration2.input_size.split(",")
        configuration2.input_size = [int(x.strip()) for x in tmp]
        configuration2.input_size = np.array(configuration2.input_size)

        tmp = configuration2.input_stride.split(",")
        configuration2.input_stride = [int(x.strip()) for x in tmp]
        configuration2.input_stride = np.array(configuration2.input_stride)

        if configuration2.weights != 'None':
            tmp = configuration2.weights.split(",")
            configuration2.weights = [float(x.strip()) for x in tmp]
            configuration2.weights = np.array(configuration2.weights)
        else:
            configuration2.weights = np.array([-1])

        if configuration2.lcn != 'None':
            tmp = configuration2.lcn.split(",")
            configuration2.lcn = [int(x.strip()) for x in tmp]
            configuration2.lcn = np.array(configuration2.lcn)
        else:
            configuration2.lcn = np.array([-1])

        tmp = configuration2.input_area.split(",")
        configuration2.input_area = [int(x.strip()) for x in tmp]
        for i in range(0, (int(len(configuration2.input_area) / 6))):
            tmp = np.vstack((configuration2.input_area[i * 6:i * 6 + 3], configuration2.input_area[i * 6 + 3:i * 6 + 6]))
            if i == 0:
                tmp2 = np.expand_dims(tmp, axis=0)
            else:
                tmp2 = np.append(tmp2, np.expand_dims(tmp, axis=0), axis=0)
        configuration2.input_area = tmp2

        tmp = configuration2.output_area.split(",")
        configuration2.output_area = [int(x.strip()) for x in tmp]
        configuration2.output_area = np.vstack((configuration2.output_area[0:3], configuration2.output_area[3:6]))

        if configuration2.prunned_classes != 'None':
            tmp = configuration2.prunned_classes.split(",")
            configuration2.prunned_classes = [int(x.strip()) for x in tmp]
            configuration2.prunned_classes = np.vstack(
                (configuration2.prunned_classes[0:len(configuration2.prunned_classes):2], configuration2.prunned_classes[1:len(configuration2.prunned_classes):2]))
        else:
            configuration2.prunned_classes = np.array([[-1], [-1]])

        configuration2.cp_dest = configuration.root + configuration2.cp_dest
        configuration2.im_dest = configuration.root + configuration2.im_dest
        configuration2.tb_dest = configuration.root + configuration2.tb_dest
        '''
        prenet = get_model(configuration2, models)

        if configuration.gpu:
            prenet.cuda()
            if configuration2.parallel:
                prenet = torch.nn.DataParallel(prenet, device_ids=list(range(torch.cuda.device_count()))[:4], output_device=list(range(torch.cuda.device_count()))[0])

        if configuration2.load != '0':
            prenet.load_state_dict(torch.load(configuration2.cp_dest+configuration2.load))
            print('Model loaded from {}'.format(configuration2.load))

        prenet.eval()
        '''
        prenet = 'None'
    else:
        configuration2 = 'None'
        prenet = 'None'

    try:
        train_net(network, configuration, prenet, configuration2, dataload)
    except KeyboardInterrupt:
        torch.save(network.state_dict(), configuration.root + '/INTERRUPTED.pth')
        print('Saved interrupt')
        sys.exit(0)
