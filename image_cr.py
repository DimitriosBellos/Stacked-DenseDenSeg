import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageColor
import multiprocessing as mp
from functools import partial

#mn=-27.501322
#mx=31.297194
mn=0                                                                                             
mx=3
norm = plt.Normalize(vmin=0, vmax=3)
cpus=16

def img(itr):
    #j=int(itr//21)
    #i=int(itr-j*21)
    f2=h5py.File('/db/user/4D/hmm2.h5', 'r') 
    data2=f2['data'][itr, 1190, 806:1414, 50:562]
    image = norm(data2)
    plt.imshow(data2,cmap='hot', vmin=0, vmax=3) 
    fig = plt.gcf()
    fig.set_size_inches(8,8)
    #if j == 0:
    #    name='/db/user/4D/HMM3/hmm_%d_old' % i
    #    plt.title('Time step:%d before HMM' % i, fontsize=20)
    #else:
    name='/db/user/4D/HMM4/hmm_%d_new' % itr
    plt.title('Time step:%d after HMM' % itr, fontsize=20)
    #fig.tight_layout()
    fig.savefig(name+'.png', dpi=300)
    f2.close()
    return np.zeros((2,))
    
pool = mp.Pool(processes=cpus)
#stdimg=partial(img, outl)
ls = list(pool.imap(img, range(0, 21)))
