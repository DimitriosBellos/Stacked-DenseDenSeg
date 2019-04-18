import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
# from torch.autograd import Variable
from torch import optim
import numpy as np
import sys
import os
import gc
from torch.utils.data import DataLoader
import shutil

import utils.dataloader as dataload
from opts import Options
from utils.presenter import PrintProgress
from utils.colors import Colors
import models


def infer(net, sm, options):

    if options.data_3D:
        data = dataload.Data3D_infer(options)
    else:
        data = dataload.Data_infer(options)

    syss = Colors()
    instances = int(np.ceil(len(data.order) / options.batchsize))

    printInfer=PrintProgress()
    inferDataloader = iter(DataLoader(dataset=data, batch_size=options.batchsize, num_workers=options.workers))
    for instance in range(0, instances):

        if instance == 0:
            leng = int((PrintProgress.getTermLength() / 2) - 4)
            sys.stdout.write(syss.Blue + '\n')
            for _ in range(leng):
                sys.stdout.write(syss.Blue + '-')
            sys.stdout.write(syss.Blue + 'Infering')
            for _ in range(leng):
                sys.stdout.write(syss.Blue + '-')
            sys.stdout.write(syss.Blue + '\n\n')

        x, idx = next(inferDataloader)

        if options.gpu:
            x = x.cuda()

        y_pred = net(x)
        y_pred = sm(y_pred)

        data.save(y_pred, idx, options)

        statement = ''
        printInfer(instance + 1, instances, [statement])

        del x, y_pred
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':

    parser = Options()
    (options, args) = parser.parse_args()

    torch.manual_seed(options.manual_seed)

    tmp = options.input_size.split(",")
    options.input_size = [int(x.strip()) for x in tmp]
    options.input_size = np.array(options.input_size)

    tmp = options.input_stride.split(",")
    options.input_stride = [int(x.strip()) for x in tmp]
    options.input_stride = np.array(options.input_stride)

    tmp = options.input_area.split(",")
    options.input_area = [int(x.strip()) for x in tmp]
    options.input_area = np.vstack((options.input_area[0:3], options.input_area[3:6]))

    tmp = options.output_area.split(",")
    options.output_area = [int(x.strip()) for x in tmp]
    options.output_area = np.vstack((options.output_area[0:3], options.output_area[3:6]))

    options.root = options.root + options.name
    options.cp_dest = options.root + options.cp_dest
    options.im_dest = options.root + options.im_dest
    options.tb_dest = options.root + options.tb_dest

    if not os.path.exists(options.root):
        os.makedirs(options.root)

    if not os.path.exists(options.cp_dest):
        os.makedirs(options.cp_dest)

    if not os.path.exists(options.im_dest):
        os.makedirs(options.im_dest)

    if not os.path.exists(options.tb_dest):
        os.makedirs(options.tb_dest)

    shutil.copy2('opts.py', options.root + ('/opts_%s.py' % options.name))
    shutil.copy2('inferer.py', options.root + ('/inferer_%s.py' % options.name))

    if options.down_samp == 'Convolution':
        down_samp=False
    else:
        down_samp=True

    if options.up_samp == 'UpConvolution':
        up_samp=False
    else:
        up_samp=True

    if options.model_type == 'UNet':
        net = models.UNet(options.input_channels, options.n_classes, options.n_layers, options.gr, MxPool=2, MaxPool=down_samp, Kernel_size=3, Batchnorm=True, Stride=1, StridePool=2, Padding=1, Bias=False, trilinear=up_samp)
    elif options.model_type == 'DenseNet':
        net = models.DenseNet(options.input_channels, options.n_classes)
    else:
        net = models.MSD(int(options.input_size[0]), options.n_classes, num_layers=options.n_layers, growth_rate=options.gr)
    sm = nn.Softmax(dim=1)

    if options.gpu:
        net.cuda()
        sm.cuda()
        cudnn.benchmark = not options.deterministic
        cudnn.deterministic = options.deterministic
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu_devices
        net = torch.nn.DataParallel(net)

    if options.load != '0':
        net.load_state_dict(torch.load(options.load))
        print('Model loaded from {}'.format(options.load))

    try:
        infer(net, sm, options)
    except KeyboardInterrupt:
        print('Infer interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)