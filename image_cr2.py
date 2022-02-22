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
cpus=16

def img(itr):
    j=int(itr//21)
    i=int(itr-j*21)
    f2=h5py.File('/db/user/4D/4D_%d.h5' % itr, 'r')
    inp=f2['data'][1205, 806:1414, 50:562]
    f2.close()
    f2=h5py.File('/db/user/4D/4D_%d_infer/Pred_%d_den.h5' % (itr, itr), 'r')
    inp2=f2['data'][1205, 806:1414, 50:562]
    f2.close()
    f2=h5py.File('/db/user/4D/original.h5', 'r')
    data0=f2['data'][itr, 1205, 806:1414, 50:562] #806+40:806+110, 50+190:50+310]
    f2.close()
    f2=h5py.File('/db/user/4D/hmm1_5.h5', 'r')
    data=f2['data'][itr, 1205, 806:1414, 50:562]
    f2.close()
    f2=h5py.File('/db/user/4D/hmm2_5.h5', 'r')
    data2=f2['data'][itr, 1205, 806:1414, 50:562]
    f2.close()
    # f2 = h5py.File('/db/user/4D/lq_annotations.h5', 'r')
    # ano = f2['data'][itr, 1190, 806:1414, 50:562]
    # f2.close()
    f, axarr = plt.subplots(1, 5, figsize=(20, 5), dpi=512)
    axarr[0].imshow(inp[:], vmin=np.nanmin(inp), vmax=np.nanmax(inp), cmap='gray')
    axarr[0].set_title('Tomogram reconsturction, time-step:%d' % itr, fontsize=10)
    axarr[1].imshow(inp2[:], vmin=np.nanmin(inp2), vmax=np.nanmax(inp2), cmap='gray')
    axarr[1].set_title('Denoised secondary output, time-step:%d' % itr, fontsize=10)
    axarr[2].imshow(data0[:], vmin=0, vmax=3, cmap='hot')
    axarr[2].set_title('Stacked-DenseUSeg, time-step:%d' % itr, fontsize=10)
    axarr[3].imshow(data[:], vmin=0, vmax=3, cmap='hot')
    axarr[3].set_title('Stacked-DenseUSeg+HMM-T, time-step:%d' % itr, fontsize=10)
    axarr[4].imshow(data2[:], vmin=0, vmax=3, cmap='hot')
    axarr[4].set_title('Stacked-DenseUSeg+HMM-TC, time-step, t:%d' % itr, fontsize=10)
    # axarr[4].imshow(ano[:], vmin=0, vmax=3, cmap='hot')
    # axarr[4].set_title('Manual 4D annotations, time-step, t:%d' % itr, fontsize=10)
    name='/home/user/Code_Folder/4D_results/Clear/ImageClear_%d' % itr
    f.savefig(name+'.png', dpi=512, bbox_inches='tight', pad_inches=0)
    return np.zeros((2,))
    
pool = mp.Pool(processes=cpus)
#stdimg=partial(img, outl)
ls = list(pool.imap(img, range(0, 21)))
