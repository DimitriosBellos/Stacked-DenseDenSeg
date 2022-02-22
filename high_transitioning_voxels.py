import numpy as np
import h5py
import sys
import time
import multiprocessing as mp
from functools import partial
from os.path import isfile
import tqdm
from hmmlearn import hmm
from utils.logger import LoggerLite
import matplotlib.pyplot as plt
import csv

write_filename='/home/user/Code_Folder/4D_results/High_trans/Voxels'
write_filename_im='/home/user/Code_Folder/4D_results/High_trans/Or/'
write_filename2='/db/user/4D/original.h5'
time_size=21
n_images=64
image_line_size=31
start = np.mgrid[1190:1191, 806:807, 50:51]  # 1702
start = np.rollaxis(start, 0, 4)
start = start.reshape(1, 3)
end = np.mgrid[1206:1207, 1414:1415, 562:563]  # 1703
end = np.rollaxis(end, 0, 4)
end = end.reshape(1, 3)
iterations=start.shape[0]
slices=end-start
cpus = 16
divisions = np.array([[2, 4, 4], ])
all_size = (slices[:, 0] * slices[:, 1] * slices[:, 2])
b = slices
b = b / divisions
b = b.astype(int)
n_classes = 4

def rot(iteration, itr):

    z = itr // (divisions[iteration, 1] * divisions[iteration, 2])
    tr = itr - z * (divisions[iteration, 1] * divisions[iteration, 2])
    y = tr // divisions[iteration, 2]
    x = tr - y * divisions[iteration, 2]

    f2 = h5py.File(write_filename2, 'r')
    data = f2.get(list(f2.keys())[0].encode('ascii', 'ignore'))
    while isinstance(data, h5py.Group):
        data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))
    Sam = data[:,
           ((z * b[iteration, 0]) + start[iteration, 0]):(((z + 1) * b[iteration, 0]) + start[iteration, 0]),
           ((y * b[iteration, 1]) + start[iteration, 1]):(((y + 1) * b[iteration, 1]) + start[iteration, 1]),
           ((x * b[iteration, 2]) + start[iteration, 2]):(((x + 1) * b[iteration, 2]) + start[iteration, 2])]
    f2.close()
    tmp = np.zeros(list(Sam.shape))
    for t in range(1, time_size):
        tmp[t,:]=(Sam[t-1,:]-Sam[t,:] != 0)
    tmp=np.sum(tmp, axis=0)
    Sz, Sy, Sx = np.indices(list(tmp.shape))
    Sz = Sz + start[iteration, 0]
    Sy = Sy + start[iteration, 1]
    Sx = Sx + start[iteration, 2]
    Samy = np.stack(list((Sz.flatten(), Sy.flatten(), Sx.flatten(), tmp.flatten())))
    Samy = np.transpose(Samy)
    return Samy

def img(idx, itr):
    f, axarr = plt.subplots(3, 3, figsize=(3.5, 5), dpi=512)
    add=(image_line_size-1)//2
    location = '/db/user/4D/4D_0.h5'
    f2 = h5py.File(location, 'r')
    data = f2.get(list(f2.keys())[0].encode('ascii', 'ignore'))
    while isinstance(data, h5py.Group):
        data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))
    zs=np.zeros((2,))
    ys=np.zeros((2,))
    xs=np.zeros((2,))
    if idx[itr, 0]+add+1>data.shape[0]:
        zs[1]=data.shape[0]
    else:
        zs[1]=idx[itr, 0]+add+1
    if idx[itr, 0]-add<0:
        zs[0]=0
    else:
        zs[0]=idx[itr, 0]-add
    if idx[itr, 1]+add+1>data.shape[1]:
        ys[1]=data.shape[1]
    else:
        ys[1]=idx[itr, 1]+add+1
    if idx[itr, 1]-add<0:
        ys[0]=0
    else:
        ys[0]=idx[itr, 1]-add
    if idx[itr, 2]+add+1>data.shape[2]:
        xs[1]=data.shape[2]
    else:
        xs[1]=idx[itr, 2]+add+1
    if idx[itr, 2]-add<0:
        xs[0]=0
    else:
        xs[0]=idx[itr, 2]-add
    zs=zs.astype(int)
    ys=ys.astype(int)
    xs=xs.astype(int)
    img_z=np.zeros((3, zs[1]-zs[0], time_size+4))
    img_y=np.zeros((3, ys[1]-ys[0], time_size+4))
    img_x=np.zeros((3, xs[1]-xs[0], time_size+4))

    mn=np.array([-1.28,-0.7,0])
    mx=np.array([4.62,5,3])
    for t in range(0, time_size):
        location = '/db/user/4D/4D_%d.h5' % t
        f2 = h5py.File(location, 'r')
        data = f2.get(list(f2.keys())[0].encode('ascii', 'ignore'))
        while isinstance(data, h5py.Group):
            data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))
        img_z[0,:,t+2]=data[zs[0]:zs[1], idx[itr, 1], idx[itr, 2]]
        img_y[0,:,t+2]=data[idx[itr, 0], ys[0]:ys[1], idx[itr, 2]]
        img_x[0,:,t+2]=data[idx[itr, 0], idx[itr, 1], xs[0]:xs[1]]
        f2.close()
        location = '/db/user/4D/4D_%d_infer/Pred_%d_den.h5' % (t, t)
        f2 = h5py.File(location, 'r')
        data = f2.get(list(f2.keys())[0].encode('ascii', 'ignore'))
        while isinstance(data, h5py.Group):
            data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))
        img_z[1,:,t+2]=data[zs[0]:zs[1], idx[itr, 1], idx[itr, 2]]
        img_y[1,:,t+2]=data[idx[itr, 0], ys[0]:ys[1], idx[itr, 2]]
        img_x[1,:,t+2]=data[idx[itr, 0], idx[itr, 1], xs[0]:xs[1]]
        f2.close()
        location = '/db/user/4D/4D_%d_infer/Pred_%d.h5' % (t, t)
        f2 = h5py.File(location, 'r')
        data = f2.get(list(f2.keys())[0].encode('ascii', 'ignore'))
        while isinstance(data, h5py.Group):
            data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))
        img_z[2,:,t+2]=data[zs[0]:zs[1], idx[itr, 1], idx[itr, 2]]
        img_y[2,:,t+2]=data[idx[itr, 0], ys[0]:ys[1], idx[itr, 2]]
        img_x[2,:,t+2]=data[idx[itr, 0], idx[itr, 1], xs[0]:xs[1]]
        f2.close()

    for i in range(0,3):
       if i<2:
           tp=np.concatenate((img_z[i,:,2:-2],img_y[i,:,2:-2],img_x[i,:,2:-2]))
           mx[i]=tp.max()
           mn[i]=tp.min()
       else:
           mn[i]=0
           mx[i]=n_classes-1
       img_z[i,:,:2]=mx[i]
       img_z[i,int(idx[itr, 0]-zs[0]),:2]=mn[i]
       img_z[i,:,int(time_size+2):]=mx[i]
       img_z[i,int(idx[itr, 0]-zs[0]),time_size+2:]=mn[i]
       img_y[i,:,:2]=mx[i]
       img_y[i,int(idx[itr, 1]-ys[0]),:2]=mn[i]
       img_y[i,:,int(time_size+2):]=mx[i]
       img_y[i,int(idx[itr, 1]-ys[0]),time_size+2:]=mn[i]
       img_x[i,:,:2]=mx[i]
       img_x[i,int(idx[itr, 2]-xs[0]),:2]=mn[i]
       img_x[i,:,int(time_size+2):]=mx[i]
       img_x[i,int(idx[itr, 2]-xs[0]),time_size+2:]=mn[i]
    axarr[0,0].imshow(img_z[0,:,:], vmin=mn[0], vmax=mx[0], cmap='gray')
    axarr[0,0].set_title('Input\nHeight x Time', fontsize=8)
    axarr[0,0].axis('off')
    axarr[0,1].imshow(img_y[0,:,:], vmin=mn[0], vmax=mx[0], cmap='gray')
    axarr[0,1].set_title('Input\nWidth x Time', fontsize=8)
    axarr[0,1].axis('off')
    axarr[0,2].imshow(img_x[0,:,:], vmin=mn[0], vmax=mx[0], cmap='gray')
    axarr[0,2].set_title('Input\nDepth x Time', fontsize=8)
    axarr[0,2].axis('off')
    axarr[1,0].imshow(img_z[1,:,:], vmin=mn[1], vmax=mx[1], cmap='gray')
    axarr[1,0].set_title('Denoised\nHeight x Time', fontsize=8)
    axarr[1,0].axis('off')
    axarr[1,1].imshow(img_y[1,:,:], vmin=mn[1], vmax=mx[1], cmap='gray')
    axarr[1,1].set_title('Denoised\nWidth x Time', fontsize=8)
    axarr[1,1].axis('off')
    axarr[1,2].imshow(img_x[1,:,:], vmin=mn[1], vmax=mx[1], cmap='gray')
    axarr[1,2].set_title('Denoised\nDepth x Time', fontsize=8)
    axarr[1,2].axis('off')
    axarr[2,0].imshow(img_z[2,:,:], vmin=mn[2], vmax=mx[2], cmap='hot')
    axarr[2,0].set_title('Annotations\nHeight x Time', fontsize=8)
    axarr[2,0].axis('off')
    axarr[2,1].imshow(img_y[2,:,:], vmin=mn[2], vmax=mx[2], cmap='hot')
    axarr[2,1].set_title('Annotations\nWidth x Time', fontsize=8)
    axarr[2,1].axis('off')
    axarr[2,2].imshow(img_x[2,:,:], vmin=mn[2], vmax=mx[2], cmap='hot')
    axarr[2,2].set_title('Annotations\nDepth x Time', fontsize=8)
    axarr[2,2].axis('off')
    #f.subplots_adjust(left=0, bottom=0, right=0, top=0, wspace=None, hspace=None)
    f.savefig(write_filename_im+'Toxel_%d.png' % itr, dpi=512, bbox_inches='tight', pad_inches=0)
    return np.zeros((2,))
    

pool = mp.Pool(processes=cpus)
outl=[]
'''
for k in range(0,iterations):
    stdfun=partial(rot, k)
    vr = list(pool.imap(stdfun, range(0, int(np.prod(divisions[k, :])))))
    vr=np.concatenate(vr)
    outll.append(vr)

outl=np.concatenate(outll)
outl = outl[np.argsort(outl[:,3])]
outl = outl[((-1)*n_images):,:]
outl=outl.astype(int)
worse=LoggerLite(write_filename, 'w')
t=('Index_0', 'Index_1', 'Index_2')
for i in range(0, time_size):
    r='Annotation_it_t=%d' % i
    t=t+(r,)
worse.setNames(t)

f2 = h5py.File(write_filename2, 'r')
data = f2.get(list(f2.keys())[0].encode('ascii', 'ignore'))
while isinstance(data, h5py.Group):
    data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))
for i in range(0, outl.shape[0]):
    st = data[:, outl[i, 0], outl[i, 1], outl[i, 2]]
    worse.add(list(outl[i, 0:3]) + list(st))
f2.close()
'''

with open(write_filename + '.csv', 'r') as csvfile:
    plots= csv.reader(csvfile, delimiter='\t')
    c = 0
    for row in plots:
        if c == 1:
            outl.append(np.array((row[0:3])).astype(int))
        else:
            c = 1
outl=np.stack(outl)

stdimg=partial(img, outl)
ls = list(pool.imap(stdimg, range(0, n_images)))

print('Done')


