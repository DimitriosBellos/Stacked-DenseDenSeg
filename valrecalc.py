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
# from tensorboardX import SummaryWriter
import shutil
import multiprocessing as mp
from functools import partial
import csv

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
    i = int(np.floor(itr / int(x.shape[2] - options.input_channels + 1)))
    j = itr - i * int(x.shape[2] - options.input_channels + 1)
    return x[i, :, j:j + options.input_channels, :, :]


def data_3D_to_2D(x, options, pool):
    func = partial(par_3D_to_2D, x, options)
    data = torch.cat(list(pool.map(func, range(0, x.shape[0] * (x.shape[2] - options.input_channels + 1)))), dim=0)
    return data


def calc_conf(tcm, n_classes, mask_class):
    accuracy = np.zeros((2 * (n_classes + 1),))
    for i in range(0, n_classes):
        accuracy[i] = (tcm[i, i] / (tcm[:, i].sum() + tcm[i, :].sum() - tcm[i, i])).cpu().detach().numpy()
    if mask_class > 0:
        accuracy[n_classes] = (accuracy[:mask_class].sum() + accuracy[mask_class + 1:n_classes].sum()) / (n_classes - 1)
    else:
        accuracy[n_classes] = accuracy[:n_classes].sum() / n_classes
    tp_tn = tcm.trace()
    for i in range(0, n_classes):
        accuracy[i + n_classes + 1] = (tp_tn / (tp_tn + tcm[:, i].sum() + tcm[i, :].sum() - 2 * tcm[i, i])).cpu().detach().numpy()
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

    if configuration.model_type == 'UNet':
        network = models.UNet(configuration.input_channels, ignore_class=configuration.mask_class, n_classes=configuration.n_classes, n_layers=configuration.n_layers, fls=configuration.gr, MxPool=2,
                              MaxPool=down_samp, Kernel_size=3, Batchnorm=True, Stride=1, StridePool=2, Padding=1, Bias=False, trilinear=up_samp)
    elif configuration.model_type == 'UNet2D':
        network = models.UNet2D(configuration.input_channels, ignore_class=configuration.mask_class, n_classes=configuration.n_classes, n_layers=configuration.n_layers, fls=configuration.gr, MxPool=2,
                                MaxPool=down_samp, Kernel_size=3, Batchnorm=False, Stride=1, StridePool=2, Padding=1, Bias=True, trilinear=up_samp)
    elif configuration.model_type == 'VoxResNet':
        network = models.VoxResNet()
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
        # print(configuration.input_channels)
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


def recalc_net(net, options, prenet, options2, dataload):
    # dataload = importlib.import_module('utils.dataloader')
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
        data = dataload.Data_Brats(options, 'train', filenam='2D_slices_Real.npz')
    elif options.dataSep:
        if options.data_3D:
            data = dataload.Data3DSep(options, 'train', filenam='2D_slices_Real.npz')
        else:
            data = dataload.DataSep(options, 'train', filenam='2D_slices_Real.npz')
    else:
        if options.data_3D:
            if options.input_fast:
                data = dataload.Data3D_fast(options, 'train', filenam='2D_slices_Real.npz')
            else:
                data = dataload.Data3D(options, 'train', filenam='2D_slices_Real.npz')
        else:
            if options.input_fast:
                data = dataload.Data_fast(options, 'train', filenam='2D_slices_Real.npz')
            else:
                data = dataload.Data(options, 'train', filenam='2D_slices_Real.npz')

    val2 = LoggerLite(options.root + '/validationsClassesReal2', 'w')
    if options.dice != 'MSE':
        conf_matrV2 = LoggerLite(options.root + '/conf_matrixesVReal2', 'w')

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
    else:
        t = t + ('SSIM',)
    t = t + ('Time',)
    val2.setNames(t)

    if options.dice == 'Mixed':
        t = ('Epoch', 'Loss', 'SSIM', 'Time')
        valD2 = LoggerLite(options.root + '/validationsClassesDenoiseReal2', 'w')
        valD2.setNames(t)

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
    epoch = 1
    
    tcm = None
    tloss = None
    tlossD = None
    tSSIM = None

    leng = int((PrintProgress.getTermLength() / 2) - 4)
    sys.stdout.write(syss.Blue + '\n')
    for _ in range(leng):
        sys.stdout.write(syss.Blue + '-')
    sys.stdout.write(syss.Blue + 'Recalculating')
    for _ in range(leng):
        sys.stdout.write(syss.Blue + '-')
    sys.stdout.write(syss.Blue + '\n\n')
    while epoch <= options.epochs and options.testing_case is False:
        network.load_state_dict(torch.load(configuration.root+'/Checkpoints/'+'CP{}.pth'.format(epoch)))
        print('Checkpoint of Epoch {} is loaded'.format(epoch))
        tcm, tloss, tlossD, tSSIM = eval_net(net, prenet, criterion, criterion2, options, options2, 0, data, ['Validation', 'val'], epoch)
        time = timer.get_value()
        if options.dice != 'MSE':
            accuracy = calc_conf(tcm, n_classes, options.mask_class)
            conf_matrV2.add(tcm.flatten())
            val2.add([epoch] + [tloss] + list(accuracy) + [time])
            if options.dice == 'Mixed':
                valD2.add([epoch] + [tlossD] + [tSSIM] + [time])
        else:
            val2.add([epoch] + [tloss] + [tSSIM] + [time])
        tcm = None
        tloss = None
        epoch += 1

    if options.testing_case:
        epoch = options.epochs + 1
    net.load_state_dict(torch.load(configuration.root + '/Checkpoints/' + 'Best_model.pth'))
    print('Checkpoint of the Best Epoch is loaded')
    tcm, tloss, tlossD, tSSIM = eval_net(net, prenet, criterion, criterion2, options, options2, 0, data, ['Testing', 'test'], epoch)
    time = timer.get_value()
    if options.dice != 'MSE':
        accuracy = calc_conf(tcm, n_classes, options.mask_class)
        conf_matrV2.add(tcm.flatten())
        val2.add([epoch] + [tloss] + list(accuracy) + [time])
        if options.dice == 'Mixed':
            valD2.add([epoch] + [tlossD] + [tSSIM] + [time])
    else:
        val2.add([epoch] + [tloss] + [tSSIM] + [time])
    return


if __name__ == '__main__':

    diname = sys.argv[0].split("/")[-1]
    opname = 'optsRe2%s' % diname[10:]
    daname = 'dataloaderRe2%s' % diname[10:]
    opname = opname[:-3]
    daname = daname[:-3]

    Options = getattr(importlib.import_module(opname), 'Options')
    dataload = importlib.import_module(daname)

    parser = Options()
    (configuration, args) = parser.parse_args()

    sys.path.append('/home/psxdb3/Code_Folder/SemSegOldDiamondX')

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

    try:
        recalc_net(network, configuration, 'None', 'None', dataload)
    except KeyboardInterrupt:
        torch.save(network.state_dict(), configuration.root + '/INTERRUPTED.pth')
        print('Saved interrupt')
        sys.exit(0)

