import numpy as np
import h5py
import sys
import time
import multiprocessing as mp
from functools import partial
from os.path import isfile
import tqdm
from hmmlearn import hmm

#write_file = '/db/user/4D/hmm1_5.h5'
#write_file2 = '/db/user/4D/original.h5'
write_file2 = '/db/user/4D/Survos_low_quality_annotation2/annotations/annotations0.h5'
write_file = '/db/user/4D/lq_annotations.h5'
time_size = 21
start = np.mgrid[0:1, 0:1, 0:1]
#start = np.mgrid[1190:1191, 806:807, 50:51]  # 1702
start = np.rollaxis(start, 0, 4)
start = start.reshape(1, 3)
end = np.mgrid[1:2, 608:609, 512:513]
#end = np.mgrid[1191:1192, 1414:1415, 562:563]  # 1703
end = np.rollaxis(end, 0, 4)
end = end.reshape(1, 3)
iterations = start.shape[0]
slices = end - start
cpus = 64
all_size = (slices[:, 0] * slices[:, 1] * slices[:, 2])
b = all_size // cpus
n_classes = 4
startn = np.array([[1191, 806, 50],])
endn = np.array([[1192, 1414, 562],])

# transmat_ = np.array([[1, 0, 0, 0],  \
#                      [0, 0.97, 0, 0.03],  \
#                      [0, 0, 0.8, 0.2],  \
#                     [0, 0, 0, 1]])
transmat_ = np.array([[0.998, 0.001, 0.001, 0], \
                      [0.001, 0.996, 0.001, 0.002], \
                      [0.001, 0.001, 0.8982, 0.0998], \
                      [0, 0.001, 0.001, 0.998]])
emissionprob_ = np.array([[0.994575008802383, 0.00542041567891894, 0.00000420947720233167, 0.00000036604149585493], \
                          [0.00108663326090176, 0.981913807838567, 0.0168672732903087, 0.000132285610222696], \
                          [0.00000512616217419578, 0.255919838540538, 0.744041788474044, 0.0000332468232440698], \
                          [0.0000561955605507165, 0.257235178420905, 0.00889294745715089, 0.733815678561394]])
startprob_ = np.ones(n_classes, ) * (1 / n_classes)
# startprob_ = np.array([0.368421, 0.613238, 0.0181519, 0.000188779])

if not isfile(write_file):
    f = h5py.File(write_file, 'w')
    f.create_dataset('data', (time_size, 2160, 1500, 1500), chunks=True)
    f.close()
if not isfile(write_file2):
    f = h5py.File(write_file2, 'w')
    f.create_dataset('data', (time_size, 512, 512), chunks=True)
    f.close()


def rot(iteration, itr):
    model = hmm.MultinomialHMM(n_components=n_classes)
    model.transmat_ = transmat_
    model.emissionprob_ = emissionprob_
    model.startprob_ = startprob_
    l = []
    tomodel = []
    if itr != cpus - 1:
        Samy = np.zeros((b[iteration] * time_size,))
        f2 = h5py.File(write_file2, 'r')
        data = f2.get(list(f2.keys())[0].encode('ascii', 'ignore'))
        while isinstance(data, h5py.Group):
            data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))
        for i in range(itr * b[iteration], (itr + 1) * b[iteration]):
            z = i // (slices[iteration, 1] * slices[iteration, 2])
            tr = i - z * slices[iteration, 1] * slices[iteration, 2]
            y = tr // slices[iteration, 2]
            x = tr - y * slices[iteration, 2]
            #tmp = data[:, int(z) + start[iteration, 0], int(y) + start[iteration, 1], int(x) + start[iteration, 2]]
            tmp = data[:, int(y) + start[iteration, 1], int(x) + start[iteration, 2]]
            if np.sum(tmp - tmp.min()) == 0:
                Samy[(i - itr * b[iteration])*time_size:(i - itr * b[iteration]+1)*time_size] = tmp
            else:
                l.append(i)
                tomodel.append(tmp)
        f2.close()
        if len(l) > 0:
            lis = np.ones((len(l),))*time_size
            lis = lis.astype(int)
            tmp = model.predict(np.expand_dims(np.concatenate(tomodel), axis=1).astype(int), lengths=lis)
            for i in range(0, len(l)):
                Samy[(l[i] - itr * b[iteration]) * time_size:(l[i] - itr * b[iteration] + 1) * time_size] = tmp[i*time_size:(i+1)*time_size]
    else:
        sz = slices[iteration, 0] * slices[iteration, 1] * slices[iteration, 2]
        ss = sz - itr * b[iteration]
        Samy = np.zeros((ss * time_size,))
        f2 = h5py.File(write_file2, 'r')
        data = f2.get(list(f2.keys())[0].encode('ascii', 'ignore'))
        while isinstance(data, h5py.Group):
            data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))
        for i in range(itr * b[iteration], sz):
            z = i // (slices[iteration, 1] * slices[iteration, 2])
            tr = i - z * slices[iteration, 1] * slices[iteration, 2]
            y = tr // slices[iteration, 2]
            x = tr - y * slices[iteration, 2]
            #tmp = data[:, int(z) + start[iteration, 0], int(y) + start[iteration, 1], int(x) + start[iteration, 2]]
            tmp = data[:, int(y) + start[iteration, 1], int(x) + start[iteration, 2]]
            if np.sum(tmp - tmp.min()) == 0:
                Samy[(i - itr * b[iteration]) * time_size:(i - itr * b[iteration] + 1) * time_size] = tmp
            else:
                l.append(i)
                tomodel.append(tmp)
        f2.close()
        if len(l) > 0:
            lis = np.ones((len(l),)) * time_size
            lis = lis.astype(int)
            tmp = model.predict(np.expand_dims(np.concatenate(tomodel), axis=1).astype(int), lengths=lis)
            for i in range(0, len(l)):
                Samy[(l[i] - itr * b[iteration]) * time_size:(l[i] - itr * b[iteration] + 1) * time_size] = tmp[i * time_size:(i + 1) * time_size]

    return Samy  # np.array((Sam, Samy))


pool = mp.Pool(processes=cpus)
for k in range(0, iterations):
    stdfun = partial(rot, k)
    vr = list(pool.imap(stdfun, range(0, cpus)))
    first = np.concatenate(vr)
    f = h5py.File(write_file, 'a')
    for t in range(0, time_size):
        rst = first[t:(all_size[k] * time_size):time_size]
        #f['data'][t, start[k, 0]:end[k, 0], start[k, 1]:end[k, 1], start[k, 2]:end[k, 2]] = rst.reshape(slices[k, 0], slices[k, 1], slices[k, 2])
        f['data'][t, startn[k, 0]:endn[k, 0], startn[k, 1]:endn[k, 1], startn[k, 2]:endn[k, 2]] = rst.reshape(slices[k, 1], slices[k, 2])
    f.close()


    print('Done')
