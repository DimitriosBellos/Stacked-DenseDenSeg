import numpy as np
import h5py
import sys
import time
import multiprocessing as mp
from functools import partial
from os.path import isfile
import tqdm
from hmmlearn import hmm
# import pomegranate
# from pomegranate.distributions import DiscreteDistribution
# from pomegranate.hmm import HiddenMarkovModel

write_file = '/db/user/4D/hmm2_5.h5'
write_file2 = '/db/user/4D/original_prob.h5'
time_size = 21
start = np.mgrid[1660:1670, 806:807, 50:51]  # 1702
start = np.rollaxis(start, 0, 4)
start = start.reshape(10, 3)
end = np.mgrid[1661:1671, 1414:1415, 562:563]  # 1703
end = np.rollaxis(end, 0, 4)
end = end.reshape(10, 3)
iterations = start.shape[0]
slices = end - start
cpus = 64
binn = 3
all_size = (slices[:, 0] * slices[:, 1] * slices[:, 2])
b = all_size // cpus
n_classes = 4

# transmat_ = np.array([[1, 0, 0, 0],  \
#                      [0, 0.97, 0, 0.03],  \
#                      [0, 0, 0.8, 0.2],  \
#                     [0, 0, 0, 1]])
transmat_ = np.array([[0.998, 0.001, 0.001, 0], \
                      [0.001, 0.996, 0.001, 0.002], \
                      [0.001, 0.001, 0.8982, 0.0998], \
                      [0, 0.001, 0.001, 0.998]])
confusion_matrix = np.array([[615, 58203, 27112284, 650, 45731, 101700, 17, 55, 43, 2, 7, 1], \
                             [1055, 64192, 200083, 5503, 1001348, 238755654, 4407, 910667, 3203510, 134, 5614, 26553], \
                             [5, 23, 7, 998, 207744, 1538605, 1016, 269887, 4809208, 8, 74, 145], \
                             [2, 1, 1, 63, 2552, 15695, 26, 235, 372, 42, 2354, 49837]])
emissionprob_ = confusion_matrix.astype(float)

for i in range(0, n_classes):
    emissionprob_[i, :] = (emissionprob_[i, :]) / (np.sum(emissionprob_[i, :]))
emissionprob_[3, 0] = emissionprob_[3, 0] * 0.01
emissionprob_[3, 2] = emissionprob_[3, 2] * 0.01
emissionprob_[3, 3] = emissionprob_[3, 3] * 0.01
emissionprob_[3, 4] = emissionprob_[3, 4] * 0.01
emissionprob_[3, 4] = emissionprob_[3, 5] * 0.1
emissionprob_[3, 4] = emissionprob_[3, 6] * 0.002
emissionprob_[0, 3] = emissionprob_[0, 3] * 0.9
emissionprob_[2, 3] = emissionprob_[2, 3] * 0.15
emissionprob_[2, 4] = emissionprob_[2, 4] * 0.134

startprob_ = np.ones(n_classes, ) * (1 / n_classes)
# startprob_ = np.array([0.368421, 0.613238, 0.0181519, 0.000188779])
if not isfile(write_file):
    f = h5py.File(write_file, 'w')
    f.create_dataset('data', (time_size, 2160, 1500, 1500), chunks=True)
    f.close()
if not isfile(write_file2):
    f = h5py.File(write_file2, 'w')
    f.create_dataset('data', (time_size, 2160, 1500, 1500), chunks=True)
    f.close()


def rot(iteration, itr):
    # dists = [DiscreteDistribution(dict(enumerate(emissionprob_[0, :]))),
    #          DiscreteDistribution(dict(enumerate(emissionprob_[1, :]))),
    #          DiscreteDistribution(dict(enumerate(emissionprob_[2, :]))),
    #          DiscreteDistribution(dict(enumerate(emissionprob_[3, :])))]
    # model = HiddenMarkovModel.from_matrix(transmat_, dists, startprob_)
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
            tmr = data[:, :, int(z) + start[iteration, 0], int(y) + start[iteration, 1], int(x) + start[iteration, 2]]
            tm1 = np.max(tmr, 1)
            tm2 = np.argmax(tmr, 1)
            tm1 = tm1 * n_classes * binn / (n_classes - 1) - binn / (n_classes - 1) - 1
            tm1 = np.ceil(tm1)
            tm1 = binn * tm2 + tm1
            # Samy[(i - itr * b[iteration]) * time_size:(i - itr * b[iteration] + 1) * time_size] = tm1
            if np.sum(tm2 - tm2.min()) == 0:
                Samy[(i - itr * b[iteration]) * time_size:(i - itr * b[iteration] + 1) * time_size] = tm2
            else:
                # tmp = model.viterbi(tm1)[1]
                # Samy[(i - itr * b[iteration]) * time_size:(i - itr * b[iteration] + 1) * time_size] = [tmp[g][0] for g in range(1, time_size+1)]
                l.append(i)
                tomodel.append(tm1)
        f2.close()
        if len(l) > 0:
            lis = np.ones((len(l),)) * time_size
            lis = lis.astype(int)
            tmp = model.predict(np.expand_dims(np.concatenate(tomodel), axis=1).astype(int), lengths=lis)
            for i in range(0, len(l)):
                Samy[(l[i] - itr * b[iteration]) * time_size:(l[i] - itr * b[iteration] + 1) * time_size] = tmp[i * time_size:(i + 1) * time_size]

        # lis = np.ones((b[iteration],)) * time_size
        # lis = lis.astype(int)
        # Samy = model.predict(np.expand_dims(Samy, axis=1).astype(int), lengths=lis)
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
            tmr = data[:, :, int(z) + start[iteration, 0], int(y) + start[iteration, 1], int(x) + start[iteration, 2]]
            tm1 = np.max(tmr, 1)
            tm2 = np.argmax(tmr, 1)
            tm1 = tm1 * n_classes * binn / (n_classes - 1) - binn / (n_classes - 1) - 1
            tm1 = np.ceil(tm1)
            tm1 = binn * tm2 + tm1
            # Samy[(i - itr * b[iteration]) * time_size:(i - itr * b[iteration] + 1) * time_size] = tm1
            if np.sum(tm2 - tm2.min()) == 0:
                Samy[(i - itr * b[iteration]) * time_size:(i - itr * b[iteration] + 1) * time_size] = tm2
            else:
                # tmp = model.viterbi(tm1)[1]
                # Samy[(i - itr * b[iteration]) * time_size:(i - itr * b[iteration] + 1) * time_size] = [tmp[g][0] for g in range(1, time_size+1)]
                l.append(i)
                tomodel.append(tm1)
        f2.close()
        if len(l) > 0:
            lis = np.ones((len(l),)) * time_size
            lis = lis.astype(int)
            tmp = model.predict(np.expand_dims(np.concatenate(tomodel), axis=1).astype(int), lengths=lis)
            for i in range(0, len(l)):
                Samy[(l[i] - itr * b[iteration]) * time_size:(l[i] - itr * b[iteration] + 1) * time_size] = tmp[i * time_size:(i + 1) * time_size]

        # lis = np.ones((ss,)) * time_size
        # lis = lis.astype(int)
        # Samy = model.predict(np.expand_dims(Samy, axis=1).astype(int), lengths=lis)
    return Samy  # np.array((Sam, Samy))


def correct(k, itr):
    f = h5py.File(write_file, 'r')
    tm = f['data'][itr, start[k, 0]:end[k, 0], start[k, 1]:end[k, 1], start[k, 2]:end[k, 2]]
    f.close()
    for x in range(0, tm.shape[0]):
        for y in range(1, tm.shape[1] - 1):
            for z in range(1, tm.shape[2] - 1):
                if tm[x, y, z] == 2:
                    if tm[x, y - 1, z] == 0 or tm[x, y + 1, z] == 0 or tm[x, y, z - 1] == 0 or tm[x, y, z + 1] == 0 or tm[x, y - 1, z - 1] == 0 or tm[x, y + 1, z + 1] == 0 or tm[
                        x, y - 1, z + 1] == 0 or tm[x, y + 1, z - 1] == 0:
                        tm[x, y, z] = 1
    return tm


pool = mp.Pool(processes=cpus)
# if pomegranate.utils.is_gpu_enabled():
#     print('GPU is Enabled')
for k in range(0, iterations):
    stdfun = partial(rot, k)
    vr = list(pool.imap(stdfun, range(0, cpus)))
    first = np.concatenate(vr)
    f = h5py.File(write_file, 'a')
    for t in range(0, time_size):
        rst = first[t:(all_size[k] * time_size):time_size]
        f['data'][t, start[k, 0]:end[k, 0], start[k, 1]:end[k, 1], start[k, 2]:end[k, 2]] = rst.reshape(slices[k, 0], slices[k, 1], slices[k, 2])
    f.close()
    print('Done')
