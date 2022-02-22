import torch
import numpy as np
from torch import optim
#from torch.autograd import Variable
from utils.accuracy import meas_cm, meas_cm_weighted, meas_cm_weighted_hmm
from torch.utils.data import DataLoader
from utils.data_vis import save_img
import sys
import multiprocessing as mp
from functools import partial
import gc


def par_3D_to_2D(x, options, itr):
    i = int(np.floor(itr / int(x.shape[2]-options.input_channels+1)))
    j = itr - i * int(x.shape[2]-options.input_channels+1)
    return x[i, :, j:j+options.input_channels, :, :]

def data_3D_to_2D(x, options, pool):
    func = partial(par_3D_to_2D, x, options)
    data = torch.cat(list(pool.map(func, range(0, x.shape[0]*(x.shape[2]-options.input_channels+1)))), dim=0)
    return data


def eval_net(net, prenet, criterion, criterion2, options, options2, pool, data, strng, instance):

    if options.gpu:
        tcm = torch.DoubleTensor(options.n_classes, options.n_classes).zero_().cuda(device=list(range(torch.cuda.device_count()))[-1])
    else:
        tcm = torch.DoubleTensor(options.n_classes, options.n_classes).zero_()
    tloss = None
    tlossD = None

    sys.stdout.write('\n')
    sys.stdout.write(strng[0])
    sys.stdout.write('\n')

    data.change_mode(strng[1])
    valDataloader = iter(DataLoader(dataset=data, batch_size=options.batchsize,num_workers=8))
    iterations = int(np.ceil(data.__len__()/options.batchsize))


    net.eval()
    with torch.no_grad():
        for i in range(0, iterations):

            if options.dice == 'Mixed':
                x, y, y_de = next(valDataloader)
            else:
                x, y = next(valDataloader)

            if prenet != 'None':
                if options.data_3D == True and options2.data_3D == False:
                    batchsize = x.shape[0]
                    y_d = data_3D_to_2D(x, options2, pool)
                    y_d = y_d.cuda()

            if options.gpu:
                x=x.cuda()
                y=y.cuda()
                if options.dice == 'Mixed':
                    y_de = y_de.cuda()

            if prenet != 'None':
                x = prenet(x)
                if options.data_3D == True and options2.data_3D == False:
                    y_d = y_d.reshape(batchsize, 1, options.input_size[0], options.input_size[1], options.input_size[2])
                #x = torch.cat([y_d, x[:,:,5:-5,:,:]], dim=1)
                #x = torch.cat([y_d, x], dim=1)

            if options.dice == 'Mixed':
                y_pred, y_pred2 = net(x)
                lossD = criterion2(y_pred2, y_de)
                loss = criterion(y_pred, y)
            else:
                y_pred = net(x)
                loss = criterion(y_pred, y)

            if options.dice != 'MSE':
                cm = meas_cm_weighted(y_pred, y, options.n_classes, options.gpu)
                tcm += cm
                if tloss is None:
                    tloss = loss
                else:
                    tloss += loss
                if options.dice == 'Mixed':
                    if tlossD is None:
                        tlossD = lossD
                    else:
                        tlossD += lossD
                acc = (cm.trace() / cm.sum()) * 100
            else:
                if tloss is None:
                    tloss = loss
                else:
                    tloss += loss
                acc = loss

            statement = '%3.2f' % (acc)

            if i==0:
               save_img(('%sImage_%03d.png' % (options.im_dest, instance)), x, y, y_pred, options)
               if options.dice == 'Mixed':
                   options.dice = 'MSE'
                   save_img(('%sImage_D_%03d.png' % (options.im_dest, instance)), x, y_de, y_pred2, options)
                   options.dice = 'Mixed'

            #del y_pred
            #torch.cuda.empty_cache()
            #gc.collect()

            if i == 0:
                sys.stdout.write(("Minbatch: %d/%d,  " % (i + 1, iterations)) + ' Acc: ' + statement)
            else:
                sys.stdout.write('\r')
                sys.stdout.write(("Minbatch: %d/%d,  " % (i + 1, iterations)) + ' Acc: ' + statement)
            sys.stdout.flush()

    data.change_mode('train')
    net.train()
    if options.gpu:
        tloss = tloss.cpu().detach().numpy()
        if options.dice == 'Mixed':
            tlossD = tlossD.cpu().detach().numpy()
    else:
        tloss = tloss.detach().numpy()
        if options.dice == 'Mixed':
            tlossD = tlossD.detach().numpy()

    if options.dice == 'Mixed':
        return tcm, tloss / iterations, tlossD / iterations
    if options.dice == 'MSE':
        return tcm, tloss / iterations, tlossD
    return tcm, tloss / iterations, tlossD