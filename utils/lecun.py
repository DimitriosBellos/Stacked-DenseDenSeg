import math
import numpy as np
import torch

def gaussianFilter(kernel_size, kernel_std):
    x = torch.zeros(kernel_size, kernel_size)

    def gauss(x, y, sigma):
        Z = 2 * math.pi * sigma * sigma
        return 1 / Z * math.exp(-(x*x + y*y) / (2 * sigma*sigma))

    mid = np.ceil(kernel_size / 2)
    for i in range(0,kernel_size):
        for j in range(0,kernel_size):
            x[i,j] = gauss(i - mid, j - mid, kernel_std)

    return x/x.sum()
