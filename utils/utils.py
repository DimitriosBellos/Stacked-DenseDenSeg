import numpy as np
from scipy.signal import convolve2d as conv2d
from PIL import Image

def makeGaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)

def localcontrastnormalization(inp, kernel_size=7, threshold=1e-4, use_divisor=True):

    #inp_conv = inp.convert('YCbCr')
    image = np.copy(np.asanyarray(inp))
    for i in range(0,3):

        x = image[:,:,i]

        kernel = makeGaussian(kernel_size,kernel_size//2)
        kernel = kernel/kernel.sum()

        tmp = conv2d(x, kernel, 'same')

        v=x-tmp

        if use_divisor:

            sum_sqr = conv2d(np.power(v,2), kernel, 'same')

            denom = np.sqrt(sum_sqr)
            mean = denom.mean()*np.ones(v.shape)
            divisor = np.maximum(denom, mean)
            divisor = np.maximum(divisor, threshold)


            v /= divisor

        image[:,:,i] = v

    return image
