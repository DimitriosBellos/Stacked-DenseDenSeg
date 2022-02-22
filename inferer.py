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


def infer(net, options, dataload):
    # dataload = importlib.import_module('utils.dataloader')
    Timer = getattr(importlib.import_module('utils.timer'), 'Timer')
    LoggerLite = getattr(importlib.import_module('utils.logger'), 'LoggerLite')
    eval_net = getattr(importlib.import_module('validation'), 'eval_net')
    meas_cm_weighted = getattr(importlib.import_module('utils.accuracy'), 'meas_cm_weighted')
    # save_img = getattr(importlib.import_module('utils.data_vis'), 'save_img')
    sm = nn.Softmax(dim=1)

    if options.data_3D:
        data = dataload.Dataset3D_infer(options)
    else:
        data = dataload.Data_infer(options)


    instances = int(np.ceil(len(data.order) / options.batchsize))

    inferDataloader = iter(DataLoader(dataset=data, batch_size=options.batchsize, num_workers=options.workers))
    net.eval()
    with torch.no_grad():
        for instance in range(0, instances):

            if instance == 0:
                sys.stdout.write('\n')
                sys.stdout.write('Infering')
                sys.stdout.write('\n')

            x, idx = next(inferDataloader)

            if options.gpu:
                x = x.cuda()

            y_pred, y_pred2 = net(x)
            y_pred = sm(y_pred)

            data.save(y_pred, y_pred2, idx, options)

            if instance == 0:
                sys.stdout.write(("Minbatch: %d/%d,  " % (instance + 1, instances)))
            else:
                sys.stdout.write('\r')
                sys.stdout.write(("Minbatch: %d/%d,  " % (instance + 1, instances)))
            sys.stdout.flush()

            #del x, y_pred
            #torch.cuda.empty_cache()
            #gc.collect()
    data.end_save(options)


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
        infer(network, configuration, dataload)
    except KeyboardInterrupt:
        torch.save(network.state_dict(), configuration.root + '/INTERRUPTED.pth')
        print('Saved interrupt')
        sys.exit(0)