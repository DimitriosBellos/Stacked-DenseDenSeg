import torch
import numpy as np
from torch import optim
#from torch.autograd import Variable
from utils.accuracy import meas_cm, meas_cm_weighted, SSIM
from utils.presenter import PrintProgress
from utils.colors import Colors
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

    tcm = None
    tloss = None
    tlossD = None
    tSSIM = None

    syss=Colors()
    ssim=SSIM()

    sys.stdout.write('\n')
    leng = int((PrintProgress.getTermLength() / 2) - 4)
    sys.stdout.write(syss.Blue + '\n')
    for _ in range(leng):
        sys.stdout.write(syss.Blue + '-')
    sys.stdout.write(syss.Blue + strng[0])
    for _ in range(leng):
        sys.stdout.write(syss.Blue + '-')
    sys.stdout.write(syss.Blue + '\n')

    data.change_mode(strng[1])
    valDataloader = iter(DataLoader(dataset=data, batch_size=options.batchsize,num_workers=8))
    iterations = int(np.ceil(data.__len__()/options.batchsize))

    printVal=PrintProgress()
    net.eval()
    with torch.no_grad():
        for i in range(0,iterations):

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
            if options.model_type == 'VoxResNet':
                y_pred, y1, y2, y3, y4 = net(x)
                loss = criterion(y_pred, y)
            else:
                if options.dice == 'Mixed':
                    y_pred, y_pred2 = net(x)
                    lossD = criterion2(y_pred2, y_de)
                    loss = criterion(y_pred, y)
                else:
                    y_pred = net(x)
                    loss = criterion(y_pred, y)

            if options.dice != 'MSE':
                cm = meas_cm_weighted(y_pred, y, options.n_classes, options.gpu)
                if tloss is None:
                    tcm = cm
                    tloss = loss
                else:
                    tcm += cm
                    tloss += loss
                if options.dice == 'Mixed':
                    ss = ssim(y_de, y_pred2)
                    if tlossD is None:
                        tlossD = lossD
                        tSSIM = ss
                    else:
                        tlossD += lossD
                        tSSIM += ss
                acc = (cm.trace() / cm.sum()) * 100
            else:
                ss = ssim(y, y_pred)
                if tloss is None:
                    tloss = loss
                    tSSIM = ss
                else:
                    tloss += loss
                    tSSIM += ss
                acc = loss

            statement = '%3.2f' % (acc)

            if i == 0:
                save_img(('%sImage_%03d.png' % (options.im_dest, instance)), x, y, y_pred, options)
                if options.dice == 'Mixed':
                    options.dice = 'MSE'
                    save_img(('%sImage_D_%03d.png' % (options.im_dest, instance)), x, y_de, y_pred2, options)
                    options.dice = 'Mixed'

            #del y_pred
            #torch.cuda.empty_cache()
            #gc.collect()

            printVal(i + 1, iterations, [statement])

    data.change_mode('train')
    net.train()
    if options.gpu:
        tloss = tloss.cpu().detach().numpy()
        if options.dice == 'Mixed':
            tlossD = tlossD.cpu().detach().numpy()
            tSSIM = tSSIM.cpu().detach().numpy()
        if options.dice == 'MSE':
            tSSIM = tSSIM.cpu().detach().numpy()
    else:
        tloss = tloss.detach().numpy()
        if options.dice == 'Mixed':
            tlossD = tlossD.detach().numpy()
            tSSIM = tSSIM.detach().numpy()
        if options.dice == 'MSE':
            tSSIM = tSSIM.detach().numpy()

    if options.dice == 'Mixed':
        return tcm, tloss / iterations, tlossD / iterations, tSSIM / iterations
    if options.dice == 'MSE':
        return tcm, tloss / iterations, tlossD, tSSIM / iterations
    return tcm, tloss / iterations, tlossD, tSSIM

