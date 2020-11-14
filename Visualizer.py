import torch.nn as nn
import torch
import numpy as np
from PIL import Image
from pylab import imshow, show

from utils.optsCamVid import Options
from utils.dataloader import DataloaderCamVid
import models

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#matplotlib.pyplot.ion()

def ShowImage(img):
    plt.imshow(img)
    plt.show(block=True)


def ReadImage(imagePath):
    return mpimg.imread(imagePath)

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

    net = models.MSD(int(options.input_size[0]), options.n_classes, num_layers=options.n_layers, growth_rate=options.gr)
    sm = nn.Softmax(dim=1)

    net.cuda()
    net = torch.nn.DataParallel(net)

    data = DataloaderCamVid(options)

    x, y = data.load(0, options, 'test')
    x=x[0:1,:,:,:].cuda()
    y=y[0:1,:,:]

    state=torch.load('/db3/psxdb3/SemSegCT/Checkpoints/CP146.pth')

    net.load_state_dict(state)

    y_pred = net(x)

    x=x.cpu().detach().numpy()
    y=y.detach().numpy()[0,:,:]
    y_pred=y_pred.cpu().detach().numpy()[0,:,:,:]

    y_pred = np.argmax(y_pred, axis=0)

    #a = Image.open('/db3/psxdb3/CamVid/testannot/0001TP_010320.png')
    #a = np.asarray(a)

    imshow(y_pred, interpolation='nearest')
    show()
    imshow(y, interpolation='nearest')
    show()