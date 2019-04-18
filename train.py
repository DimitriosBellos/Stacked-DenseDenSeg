import importlib
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable
from torch import optim
import numpy as np
import sys
import os
import gc
from torch.utils.data import DataLoader
import shutil
import multiprocessing as mp
from functools import partial

'''
import utils.dataloader as dataload
from utils.dice import DiceLoss
from opts import Options
from opts2 import Options as Options2
from utils.logger import LoggerLite
from utils.presenter import PrintProgress
from utils.timer import Timer
from utils.colors import Colors
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
    i = int(np.floor(itr / int(x.shape[2]-options.input_channels+1)))
    j = itr - i * int(x.shape[2]-options.input_channels+1)
    return x[i, :, j:j+options.input_channels, :, :]


def data_3D_to_2D(x, options, pool):
    func = partial(par_3D_to_2D, x, options)
    data = torch.cat(list(pool.map(func, range(0, x.shape[0]*(x.shape[2]-options.input_channels+1)))), dim=0)
    return data


def get_model(configuration, models):
    if configuration.down_samp == 'Convolution':
        down_samp = False
    else:
        down_samp = True

    if configuration.up_samp == 'UpConvolution':
        up_samp = False
    else:
        up_samp = True

    if configuration.model_type == 'UNet':
        network = models.UNet(configuration.input_channels, ignore_class=configuration.mask_class, n_classes=configuration.n_classes, n_layers=configuration.n_layers, fls=configuration.gr, MxPool=2,
                              MaxPool=down_samp, Kernel_size=3, Batchnorm=True, Stride=1, StridePool=2, Padding=1, Bias=False, trilinear=up_samp)
    elif configuration.model_type == 'UNet2D':
        network = models.UNet2D(configuration.input_channels, ignore_class=configuration.mask_class, n_classes=configuration.n_classes, n_layers=configuration.n_layers, fls=configuration.gr, MxPool=2,
                              MaxPool=down_samp, Kernel_size=3, Batchnorm=False, Stride=1, StridePool=2, Padding=1, Bias=True, trilinear=up_samp)
    elif configuration.model_type == 'VoxResNet':
        network = models.VoxResNet(in_channels=1, n_classes=configuration.n_classes)
    elif configuration.model_type == 'DenseNet':
        network = models.DenseNet(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=32, Theta=0.5, Dropout=0, Growth_rate=configuration.gr)
    elif configuration.model_type == 'DenseNet5L':
        network = models.DenseNet5L(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=configuration.gr)
    elif configuration.model_type == 'Dense2DNet':
        network = models.Dense2DNet(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=configuration.gr)
    elif configuration.model_type == 'Dense2DNet5L':
        network = models.Dense2DNet5L(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=configuration.gr)
    elif configuration.model_type == 'DenseNet2_5':
        network = models.DenseNet2_5(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=configuration.gr)
    elif configuration.model_type == 'DenseNetHalfScope':
        network = models.DenseNetHalfScope(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=configuration.gr)
    elif configuration.model_type == 'DenseUNet':
        network = models.DenseUNet(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=configuration.gr)
    elif configuration.model_type == 'DenseUNet2D':
        network = models.DenseUNet2D(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=configuration.gr)
    elif configuration.model_type == 'DenseNetStack':
        network = models.DenseNetStack(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=configuration.gr)
    elif configuration.model_type == 'DenseNetStack_2D':
        network = models.DenseNetStack_2D(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=configuration.gr)
    elif configuration.model_type == 'DenseUNetStack':
        network = models.DenseUNetStack(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=32, Theta=0.5, Dropout=0.2, Growth_rate=configuration.gr)
    elif configuration.model_type == 'DenseUNetStack_2D':
        network = models.DenseUNetStack_2D(configuration.input_channels, configuration.n_classes, ignore_class=configuration.mask_class, k0=64, Theta=0.5, Dropout=0.2, Growth_rate=configuration.gr)
    elif configuration.model_type == 'MSDrop':
        network = models.MSDrop(configuration.input_channels, configuration.n_classes, num_layers=configuration.n_layers, growth_rate=configuration.gr, ignore_class=configuration.mask_class)
    else:
        if configuration.data_3D:
            network = models.MSD3D(configuration.input_channels, configuration.n_classes, num_layers=configuration.n_layers, growth_rate=configuration.gr, ignore_class=configuration.mask_class)
        else:
            network = models.MSD(configuration.input_channels, configuration.n_classes, num_layers=configuration.n_layers, growth_rate=configuration.gr, ignore_class=configuration.mask_class)

    return network


def train_net(net, options, prenet, options2):

    dataload = importlib.import_module('utils.dataloader')
    DiceLoss = getattr(importlib.import_module('utils.dice'), 'DiceLoss')
    Timer = getattr(importlib.import_module('utils.timer'), 'Timer')
    LoggerLite = getattr(importlib.import_module('utils.logger'), 'LoggerLite')
    PrintProgress = getattr(importlib.import_module('utils.presenter'), 'PrintProgress')
    Colors = getattr(importlib.import_module('utils.colors'), 'Colors')
    eval_net = getattr(importlib.import_module('validation'), 'eval_net')
    meas_cm_weighted = getattr(importlib.import_module('utils.accuracy'), 'meas_cm_weighted')
    SSIM = getattr(importlib.import_module('utils.accuracy'), 'SSIM')
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

    # hist = Logger(options.root+'/history', 'w')
    # val = Logger(options.root+'/validations', 'w')
    hist = LoggerLite(options.root + '/historyClasses', 'w')
    val = LoggerLite(options.root + '/validationsClasses', 'w')
    if options.dice != 'MSE':
        conf_matrH = LoggerLite(options.root + '/conf_matrixesH', 'w')
        conf_matrV = LoggerLite(options.root + '/conf_matrixesV', 'w')

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
        conf_matrH.setNames(t)
        conf_matrV.setNames(t)
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
    else:
        t = t + ('SSIM',)
    t = t + ('Time',)
    # hist.setNames(('Epoch', 'Acc', 'Time'))
    hist.setNames(t)
    val.setNames(t)
    if options.dice == 'Mixed':
        histD = LoggerLite(options.root + '/historyClassesDenoise', 'w')
        valD = LoggerLite(options.root + '/validationsClassesDenoise', 'w')
        t = ('Epoch', 'Loss', 'SSIM', 'Time')
        histD.setNames(t)
        valD.setNames(t)

    if options.optmethod is 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=options.lr, momentum=options.momentum)
    elif options.optmethod is 'rmsprop':
        optimizer = optim.RMSprop(net.parameters())
    else:
        optimizer = optim.Adam(net.parameters(), lr=options.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=options.weig_dec)
        # optimizer = optim.Adam([{'params': net.module.inc.parameters(), 'lr': options.lr*10},
        #                         {'params': net.module.D1.parameters(), 'lr': options.lr*10},
        #                         {'params': net.module.c1.parameters()},
        #                         {'params': net.module.c2.parameters()},
        #                         {'params': net.module.D2.parameters()}],
        #                        lr=options.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=options.weig_dec)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=options.epochs / 2, gamma=0.1)
    #scheduler.last_epoch = 7
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)
    #scheduler = PolyLR(optimizer, options.epochs*instances)

    criterion2 = None

    if options.dice == 'Dice':
        if options.weighted:
            if options.gpu:
                data.weights = data.weights.cuda()
            criterion = DiceLoss(weight=data.weights)
        else:
            criterion = DiceLoss()
    elif options.dice == 'MSE':
        criterion = nn.MSELoss()
        criterion1 = nn.MSELoss()
        criterion2 = nn.MSELoss()
        criterion3 = nn.MSELoss()
        criterion4 = nn.MSELoss()
        if options.gpu:
            criterion1.cuda()
            criterion2.cuda()
            criterion3.cuda()
            criterion4.cuda()
    else:
        if options.model_type == 'VoxResNet':
            if options.gpu:
                data.weights = data.weights.cuda()
            criterion = nn.CrossEntropyLoss(weight=data.weights, ignore_index=options.mask_class)
            criterion1 = nn.CrossEntropyLoss(weight=data.weights, ignore_index=options.mask_class)
            criterion2 = nn.CrossEntropyLoss(weight=data.weights, ignore_index=options.mask_class)
            criterion3 = nn.CrossEntropyLoss(weight=data.weights, ignore_index=options.mask_class)
            criterion4 = nn.CrossEntropyLoss(weight=data.weights, ignore_index=options.mask_class)
            if options.gpu:
                criterion1.cuda()
                criterion2.cuda()
                criterion3.cuda()
                criterion4.cuda()
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

    syss = Colors()
    timer = Timer()
    ssim = SSIM()
    epoch = 1
    counter_before_epoch = 0
    instances = int(np.ceil(data.__len__() / options.batchsize))
    itr = 0
    tcm = None
    tloss = None
    tlossD = None
    tSSIM = None
    critical_value = None
    idx = None
    #scheduler = PolyLR(optimizer, options.epochs*instances)
    if options.prenet != 'None':
        pool = mp.Pool(processes=options.workers)
    else:
        pool = 'None'

    net.train()
    printTrain = PrintProgress()
    while epoch <= options.epochs:
        trainDataloader = iter(DataLoader(dataset=data, batch_size=options.batchsize, num_workers=mp.cpu_count()))
        for instance in range(0, instances):
            if epoch <= options.epochs:
                if (counter_before_epoch == 0) or (instance == 0):
                    leng = int((PrintProgress.getTermLength() / 2) - 4)
                    sys.stdout.write(syss.Blue + '\n')
                    for _ in range(leng):
                        sys.stdout.write(syss.Blue + '-')
                    sys.stdout.write(syss.Blue + 'Training')
                    for _ in range(leng):
                        sys.stdout.write(syss.Blue + '-')
                    sys.stdout.write(syss.Blue + '\n\n')

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

                #scheduler.step()

                optimizer.zero_grad()

                if prenet != 'None':
                    with torch.no_grad():
                        x = prenet(x)
                        if options.data_3D == True and options2.data_3D == False:
                            y_d = y_d.reshape(batchsize, 1, options.input_size[0], options.input_size[1], options.input_size[2])
                        #x = torch.cat([y_d, x[:,:,5:-5,:,:]], dim=1)
                        #x = torch.cat([y_d, x], dim=1)
                        #x = trans_data(y_d, x)
                if options.model_type == 'VoxResNet':
                    y_pred, y_pred1, y_pred2, y_pred3, y_pred4 = net(x)
                    loss = criterion(y_pred, y)
                    loss1 = criterion1(y_pred1, y)
                    loss2 = criterion2(y_pred2, y)
                    loss3 = criterion3(y_pred3, y)
                    loss4 = criterion4(y_pred4, y)
                    loss += loss1 + loss2 + loss3 + loss4
                else:
                    if options.dice == 'Mixed':
                        y_pred, y_pred2 = net(x)
                        lossD = 6 * criterion2(y_pred2, y_de)
                        loss = criterion(y_pred, y) + lossD
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
                        ss = ssim(y_de, y_pred2)
                        if tlossD is None:
                            tlossD = lossD
                            tSSIM = ss
                        else:
                            tlossD += lossD
                            tSSIM += ss
                else:
                    ss = ssim(y, y_pred)
                    if tloss is None:
                        tloss = loss
                        tSSIM = ss
                    else:
                        tloss += loss
                        tSSIM += ss
                    acc = loss
                    statement = '%f' % acc


                # wr.add_scalar('%s/Train/accuracy' % options.name, acc, itr*instances+instance)

                printTrain(instance + 1, instances, [statement])

                counter_before_epoch += 1
                if counter_before_epoch == options.val_freq:
                    tloss = tloss.cpu().detach().numpy() / options.val_freq
                    time = timer.get_value()
                    if options.dice != 'MSE':
                        accuracy = np.zeros((2 * (n_classes + 1),))
                        for i in range(0, n_classes):
                            accuracy[i] = (tcm[i, i] / (tcm[:, i].sum() + tcm[i, :].sum() - tcm[i, i])).cpu().detach().numpy()
                        if options.mask_class > 0:
                            accuracy[n_classes] = (accuracy[:options.mask_class].sum() + accuracy[options.mask_class + 1:n_classes].sum()) / (n_classes - 1)
                        else:
                            accuracy[n_classes] = accuracy[:n_classes].sum() / n_classes
                        tp_tn = tcm.trace()
                        for i in range(0, n_classes):
                            accuracy[i + n_classes + 1] = (tp_tn / (tp_tn + tcm[:, i].sum() + tcm[i, :].sum() - 2 * tcm[i, i])).cpu().detach().numpy()
                        if options.mask_class > 0:
                            accuracy[2 * n_classes + 1] = (accuracy[(n_classes + 1):(n_classes + 1 + options.mask_class)].sum() + accuracy[
                                                                                                                                  (n_classes + options.mask_class + 2):(2 * n_classes + 1)].sum()) / (
                                                                      n_classes - 1)
                        else:
                            accuracy[2 * n_classes + 1] = accuracy[(n_classes + 1):(2 * n_classes + 1)].sum() / n_classes
                        conf_matrH.add(tcm.cpu().detach().numpy().flatten())
                        hist.add([epoch] + [tloss] + list(accuracy) + [time])
                        if options.dice == 'Mixed':
                            tlossD = tlossD.cpu().detach().numpy() / options.val_freq
                            tSSIM = tSSIM.cpu().detach().numpy() / options.val_freq
                            histD.add([epoch] + [tlossD] + [tSSIM] + [time])
                    else:
                        tSSIM = tSSIM.cpu().detach().numpy() / options.val_freq
                        hist.add([epoch] + [tloss] + [tSSIM] + [time])

                    counter_before_epoch = 0
                    tcm, tloss, tlossD, tSSIM = eval_net(net, prenet, criterion, criterion2, options, options2, pool, data, ['Validation', 'val'], epoch)
                    torch.save(net.state_dict(), options.cp_dest + 'CP{}.pth'.format(epoch))

                    scheduler.step()

                    time = timer.get_value()
                    if options.dice != 'MSE':
                        accuracy = np.zeros((2 * (n_classes + 1),))
                        for i in range(0, n_classes):
                            accuracy[i] = (tcm[i, i] / (tcm[:, i].sum() + tcm[i, :].sum() - tcm[i, i])).cpu().detach().numpy()
                        if options.mask_class > 0:
                            accuracy[n_classes] = (accuracy[:options.mask_class].sum() + accuracy[options.mask_class + 1:n_classes].sum()) / (n_classes - 1)
                        else:
                            accuracy[n_classes] = accuracy[:n_classes].sum() / n_classes
                        tp_tn = tcm.trace()
                        for i in range(0, n_classes):
                            accuracy[i + n_classes + 1] = (tp_tn / (tp_tn + tcm[:, i].sum() + tcm[i, :].sum() - 2 * tcm[i, i])).cpu().detach().numpy()
                        if options.mask_class > 0:
                            accuracy[2 * n_classes + 1] = (accuracy[(n_classes + 1):(n_classes + 1 + options.mask_class)].sum() + accuracy[
                                                                                                                                  (n_classes + options.mask_class + 2):(2 * n_classes + 1)].sum()) / (
                                                                      n_classes - 1)
                        else:
                            accuracy[2 * n_classes + 1] = accuracy[(n_classes + 1):(2 * n_classes + 1)].sum() / n_classes
                        conf_matrV.add(tcm.cpu().detach().numpy().flatten())
                        val.add([epoch] + [tloss] + list(accuracy) + [time])
                        if options.dice == 'Mixed':
                            valD.add([epoch] + [tlossD] + [tSSIM] + [time])

                        if critical_value is None:
                            critical_value = accuracy[n_classes]
                            idx = epoch
                        else:
                            if critical_value < accuracy[n_classes]:
                                critical_value = accuracy[n_classes]
                                idx = epoch
                    else:

                        val.add([epoch] + [tloss] + [tSSIM] + [time])

                        if critical_value is None:
                            critical_value = tloss
                            idx = epoch
                        else:
                            if critical_value > tloss:
                                critical_value = tloss
                                idx = epoch

                    tcm = None
                    tloss = None
                    tlossD = None
                    tSSIM = None
                    # wr.add_scalar('%s/Validation/accuracy' % options.name, accuracy[2*n_classes+1], epoch)
                    epoch += 1
            else:
                break
        itr += 1

    net.load_state_dict(torch.load(options.cp_dest + 'CP{}.pth'.format(idx)))
    tcm, tloss, tlossD, tSSIM = eval_net(net, prenet, criterion, criterion2, options, options2, pool, data, ['Testing', 'test'], epoch)
    torch.save(net.state_dict(), options.cp_dest + 'Best_model.pth')
    time = timer.get_value()
    if options.dice != 'MSE':
        accuracy = np.zeros((2 * (n_classes + 1),))
        for i in range(0, n_classes):
            accuracy[i] = (tcm[i, i] / (tcm[:, i].sum() + tcm[i, :].sum() - tcm[i, i])).cpu().detach().numpy()
        accuracy[n_classes] = accuracy[:n_classes].sum() / n_classes
        tp_tn = tcm.trace()
        for i in range(0, n_classes):
            accuracy[i + n_classes + 1] = (tp_tn / (tp_tn + tcm[:, i].sum() + tcm[i, :].sum() - 2 * tcm[i, i])).cpu().detach().numpy()
        accuracy[2 * n_classes + 1] = (tcm.trace() / tcm.sum()).cpu().detach().numpy()
        conf_matrV.add(tcm.flatten())
        val.add([epoch] + [tloss] + list(accuracy) + [time])
        if options.dice == 'Mixed':
            valD.add([epoch] + [tlossD] + [tSSIM] + [time])
    else:
        val.add([epoch] + [tloss] + [tSSIM] + [time])

    return


if __name__ == '__main__':

    opname = sys.argv[0].split("/")[-1]
    opname = 'opts%s' % opname[5:]
    opname = opname[:-3]

    Options = getattr(importlib.import_module(opname), 'Options')

    parser = Options()
    (configuration, args) = parser.parse_args()

    if configuration.prenet:
        opname = sys.argv[0].split("/")[-1]
        opname = 'opts2%s' % opname[5:]
        opname = opname[:-3]

        Options2 = getattr(importlib.import_module(opname), 'Options')

        parser = Options2()
        (configuration2, args) = parser.parse_args()

    sys.path.append('/home/psxdb3/Code_Folder/SemSegOld')

    models = importlib.import_module('models')

    torch.manual_seed(configuration.manual_seed)

    tmp = configuration.input_filename.split(",")
    configuration.input_filename = np.array(tmp)[0]

    tmp = configuration.annotations_filename.split(",")
    configuration.annotations_filename = np.array(tmp)[0]

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
    configuration.input_area = np.vstack((configuration.input_area[0:3], configuration.input_area[3:6]))

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
        #os.environ['CUDA_VISIBLE_DEVICES'] = configuration.gpu_devices
        if configuration.parallel:
            network = torch.nn.DataParallel(network, device_ids=list(range(torch.cuda.device_count()))[:4], output_device=list(range(torch.cuda.device_count()))[0])

    if configuration.load != '0':
        network.load_state_dict(torch.load(configuration.cp_dest+configuration.load))
        print('Model loaded from {}'.format(configuration.load))

    if configuration.prenet:

        tmp = configuration2.input_filename.split(",")
        configuration2.input_filename = np.array(tmp)[0]

        tmp = configuration2.annotations_filename.split(",")
        configuration2.annotations_filename = np.array(tmp)[0]

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
        configuration2.input_area = np.vstack((configuration2.input_area[0:3], configuration2.input_area[3:6]))

        tmp = configuration2.output_area.split(",")
        configuration2.output_area = [int(x.strip()) for x in tmp]
        configuration2.output_area = np.vstack((configuration2.output_area[0:3], configuration2.output_area[3:6]))

        if configuration2.prunned_classes != 'None':
            tmp = configuration2.prunned_classes.split(",")
            configuration2.prunned_classes = [int(x.strip()) for x in tmp]
            configuration2.prunned_classes = np.vstack((configuration2.prunned_classes[0:len(configuration2.prunned_classes):2], configuration2.prunned_classes[1:len(configuration2.prunned_classes):2]))
        else:
            configuration2.prunned_classes = np.array([[-1], [-1]])

        configuration2.root = configuration2.root + configuration2.name
        configuration2.cp_dest = configuration2.root + configuration2.cp_dest
        configuration2.im_dest = configuration2.root + configuration2.im_dest
        configuration2.tb_dest = configuration2.root + configuration2.tb_dest

        prenet = get_model(configuration2, models)

        if configuration.gpu:
            prenet.cuda()
            if configuration2.parallel:
                prenet = torch.nn.DataParallel(prenet, device_ids=list(range(torch.cuda.device_count()))[:4], output_device=list(range(torch.cuda.device_count()))[0])

        if configuration2.load != '0':
            prenet.load_state_dict(torch.load(configuration2.cp_dest+configuration2.load))
            print('Model loaded from {}'.format(configuration2.load))

        prenet.eval()
    else:
        configuration2='None'
        prenet='None'

    try:
        train_net(network, configuration, prenet, configuration2)
    except KeyboardInterrupt:
        torch.save(network.state_dict(), configuration.root + '/INTERRUPTED.pth')
        print('Saved interrupt')
        sys.exit(0)

