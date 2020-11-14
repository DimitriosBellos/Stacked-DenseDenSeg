import numpy as np
from scipy.ndimage.filters import convolve as conv3d
from scipy.ndimage import rotate
from skimage import transform as tf
from scipy.signal import convolve2d as conv2d
from PIL import Image
from pylab import imshow, show
import math
import torchvision.transforms as T

def gaussian2dfilter(num_channels=3, radius=9, sigma=2, use_divisor=True):
    x = np.zeros((num_channels, radius, radius))

    def gauss(x, y, sigma=2.0):
        Z = 2 * np.pi * sigma ** 2
        return 1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    mid = np.floor(radius / 2.)
    for kernel_idx in range(0, num_channels):
        for i in range(0, radius):
            for j in range(0, radius):
                x[kernel_idx, i, j] = gauss(i - mid, j - mid, sigma)

    return x / np.sum(x)


def makeGaussian(radius=9, sigma=3):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x, y = np.meshgrid(np.linspace(-1, 1, radius), np.linspace(-1, 1, radius))
    d = np.sqrt(x * x + y * y)
    mu=0
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

    return g/g.sum()

def gaussianFilter(kernel_size=9, kernel_std=2):
    x = np.zeros((kernel_size, kernel_size))

    def gauss(x, y, sigma):
        Z = 2 * math.pi * sigma * sigma
        return 1 / Z * math.exp(-(x*x + y*y) / (2 * sigma*sigma))

    mid = np.floor(kernel_size / 2).astype(int)
    for i in range(0,kernel_size):
        for j in range(0,kernel_size):
            x[i,j] = gauss(i - mid, j - mid, kernel_std)

    return x/x.sum()

def lcn(inp, threshold=1e-4, radius=9, use_divisor=True):

    kernel = gaussian2dfilter(num_channels=3, radius=9)

    local_average = conv3d(inp, kernel, mode='constant')

    local_average[0, :, :] = local_average[1, :, :]
    local_average[2, :, :] = local_average[1, :, :]

    sumNormInp = inp-local_average

    if use_divisor:

        sum_sqr = conv3d(sumNormInp ** 2, kernel, mode='constant')

        sum_sqr[0, :, :] = sum_sqr[1, :, :]
        sum_sqr[2, :, :] = sum_sqr[1, :, :]

        denom = np.sqrt(sum_sqr)

        for i in range(0,3):

            M = denom[i, :, :].mean()

            denom[denom<M]=M

        result = sumNormInp / denom

    else:

        result = sumNormInp

    result[result<threshold]=threshold

    return result

def lcn2(inp, threshold=1, radius=9, use_divisor=True):

    kernel = gaussianFilter()

    local_average = np.empty(inp.shape)
    mid = np.floor(radius / 2).astype(int)
    for i in range(0,inp.shape[0]):
        tmp = conv2d(inp[i,:,:], kernel, mode='full')
        local_average[i, :, :] = tmp[mid:-mid,mid:-mid]

    sumNormInp = inp-local_average

    if use_divisor:

        sum_sqr = np.empty(inp.shape)
        for i in range(0, inp.shape[0]):
            tmp = conv2d(sumNormInp[i, :, :] ** 2, kernel, mode='full')
            sum_sqr[i, :, :] = tmp[mid:-mid,mid:-mid]

        denom = np.sqrt(sum_sqr)

        for i in range(0,inp.shape[0]):

            M = denom[i, :, :].mean()

            temp = denom[i, :, :]

            temp[temp<M]=M
            temp[temp<threshold]=threshold

            denom[i, :, :] = temp

        result = sumNormInp / denom

    else:

        result = sumNormInp

    #result[result<0]=0

    return result

if __name__ == '__main__':


    #tmp = np.asarray(Image.open('/home/psxdb3/Dropbox/Cicek/CamVid/train/0016E5_00480.png'))
    tmp = np.asarray(Image.open('/home/psxdb3/lena.png'))

    lena = np.empty((tmp.shape[2], tmp.shape[0], tmp.shape[1]))

    lena[0, :, :] = tmp[:, :, 0]
    lena[1, :, :] = tmp[:, :, 1]
    lena[2, :, :] = tmp[:, :, 2]

    lena_lcn = np.empty((tmp.shape[2], tmp.shape[0], tmp.shape[1]))

    lena_lcn = lcn2(lena)

    #lena_lcn = (lena_lcn-lena_lcn.min())/(lena_lcn.max() - lena_lcn.min())

    #for i in range(0,3):
        #lena_lcn[i,:,:] = (lena_lcn[i,:,:] - lena_lcn[i,:,:].min())/(lena_lcn[i,:,:].max() - lena_lcn[i,:,:].min())

    #lena_lcn = lena_lcn*255

    lena_img = np.empty((tmp.shape[0], tmp.shape[1], tmp.shape[2]))
    lena_img[:, :, 0] = lena_lcn[0, :, :]
    lena_img[:, :, 1] = lena_lcn[1, :, :]
    lena_img[:, :, 2] = lena_lcn[2, :, :]

    res=(lena_img).astype(np.uint8)

    #res = rotate(res, 10, cval=255)
    # shift_y, shift_x = np.array(res.shape[:2]) / 2.
    # tf_rotate = tf.SimilarityTransform(rotation=np.deg2rad(-5))
    # tf_shift = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
    # tf_shift_inv = tf.SimilarityTransform(translation=[shift_x+5, shift_y+5])
    # tf_trans = tf.SimilarityTransform(translation=[10, 10])
    # image_rotated = tf.warp(res, (tf_shift + (tf_rotate + tf_shift_inv)).inverse)
    #
    # imshow((image_rotated*255).astype(int), interpolation='nearest')
    # show()

    #res = (image_rotated*255).astype(int)
    #img = Image.fromarray(res.astype(np.uint8))
    img = Image.fromarray(res)

    img.show()


