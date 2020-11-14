import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import numpy as np
import sys
import os
import gc

from utils.dataloader import DataloaderCamVid
from utils.optsCamVid import Options
from utils.logger import Logger
import utils.presenter as presenter
from utils.timer import Timer
from utils.colors import Colors
from validation import eval_net, test_net
from utils.accuracy import meas_net
import models

def train_net(net, sm, options):
    data = DataloaderCamVid(options)

    hist = Logger('history','w')
    val = Logger('validations','w')

    hist.setNames(('Epoch','Accuracy','Time'))
    val.setNames(('Instance','Accuracy','Time'))

    if options.optmethod is 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=options.lr, momentum=options.momentum)
    elif options.optmethod is 'rmsprop':
        optimizer = optim.RMSprop(net.parameters())
    else:
        optimizer = optim.Adam(net.parameters())

    if options.weighted:
        if options.gpu:
            data.weights=data.weights.cuda()
        criterion = nn.CrossEntropyLoss(weight=data.weights, reduce=options.reduce, ignore_index=options.mask_class)
    else:
        criterion = nn.CrossEntropyLoss(reduce=options.reduce, ignore_index=options.mask_class)
    #criterion = nn.BCELoss()

    syss=Colors()
    timer=Timer()
    instance=0
    instances=data.batchTrain.shape[0]
    epochs=options.epochs

    for epoch in range(0, epochs):

        leng = int((presenter.getTermLength() / 2) - 4)
        sys.stdout.write(syss.Blue + '\n')
        for m in range(0, leng):
            sys.stdout.write(syss.Blue + '-')
        sys.stdout.write(syss.Blue + 'Training')
        for m in range(0, leng):
            sys.stdout.write(syss.Blue + '-')
        sys.stdout.write(syss.Blue + '\n\n')
        tot=0
        for instance in range(0,instances):

            x, y = data.load(instance, options, 'train')

            if options.gpu:
                x = x.cuda()
                y = y.cuda()

            optimizer.zero_grad()

            y_pred = net(x)

            acc=meas_net(y_pred,y,sm,options)
            tot += acc

            loss = criterion(y_pred, y)

            loss.backward()

            optimizer.step()

            statement='%3.2f'%(acc*100)

            presenter.PrintProgress(instance+1,instances,[statement])
            time = timer.get_value()
            gc.collect()

            del x, y, y_pred, loss

        tot = tot / instances
        hist.add([epoch, tot, time])

        torch.set_grad_enabled(False)
        acc = eval_net(net,sm,options,data,['Validation','val'])
        torch.cuda.empty_cache()
        torch.set_grad_enabled(True)
        torch.save(net.state_dict(), options.cp_dest + 'CP{}.pth'.format(epoch+1))
        time = timer.get_value()
        val.add([epoch,acc,time])
        data.permutate(options)

    best_model=np.argmax(np.array(hist.symbols[1]))
    net.load_state_dict(torch.load(options.cp_dest + 'CP{}.pth'.format(best_model+1)))
    acc = eval_net(net, sm, options, data,['Testing','test'])
    time = timer.get_value()
    val.add([instance, acc, time])

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
    options.input_area = np.array([options.input_area[0:3],options.input_area[3:6]])

    tmp = options.output_area.split(",")
    options.output_area = [int(x.strip()) for x in tmp]
    options.output_area = np.array([options.output_area[0:3],options.output_area[3:6]])

    #net = models.UNet(options.input_channels, options.n_classes, options.n_layers, options.gr)
    #net = models.DenseNet(options.input_channels, options.n_classes)
    net = models.MSD(int(options.input_size[0]), options.n_classes, num_layers=options.n_layers, growth_rate=options.gr)
    sm = nn.Softmax(dim=1)

    if options.load!='0':
        net.load_state_dict(torch.load(options.load))
        print('Model loaded from {}'.format(options.load))

    if options.gpu:
        net.cuda()
        sm.cuda()
        cudnn.benchmark = True
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu_devices
        #os.system('export CUDA_VISIBLE_DEVICES=0,2') #% options.gpu_devices)
        net = torch.nn.DataParallel(net)
    #print(os.environ['CUDA_VISIBLE_DEVICES'])

    if not os.path.isdir(options.cp_dest):
        os.makedirs(options.cp_dest)

    try:
        train_net(net, sm, options)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
