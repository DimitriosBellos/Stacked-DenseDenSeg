import numpy as np
import h5py
import sys
import time
import multiprocessing as mp
from functools import partial
from os.path import isfile
import tqdm
from hmmlearn import hmm
import pomegranate
from pomegranate.distributions import DiscreteDistribution
from pomegranate.hmm import HiddenMarkovModel

write_file = '/db/user/4D/hmm2_5.h5'
write_file2 = '/db/user/4D/original_prob.h5'
time_size = 21
start = np.mgrid[1192:1193, 806:807, 50:51]  # 1702
start = np.rollaxis(start, 0, 4)
start = start.reshape(1, 3)
end = np.mgrid[1660:1661, 1414:1415, 562:563]  # 1703
end = np.rollaxis(end, 0, 4)
end = end.reshape(1, 3)
iterations = start.shape[0]
slices = end - start
cpus = 32
binn = 3
divisions = np.array([[4, 4, 4], ])
all_size = (slices[:, 0] * slices[:, 1] * slices[:, 2])
b = slices
b = b / divisions
b = b.astype(int)
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
    dists = [DiscreteDistribution(dict(enumerate(emissionprob_[0, :]))),
             DiscreteDistribution(dict(enumerate(emissionprob_[1, :]))),
             DiscreteDistribution(dict(enumerate(emissionprob_[2, :]))),
             DiscreteDistribution(dict(enumerate(emissionprob_[3, :])))]
    model = HiddenMarkovModel.from_matrix(transmat_, dists, startprob_)
    # model = hmm.MultinomialHMM(n_components=n_classes)
    # model.transmat_ = transmat_
    # model.emissionprob_ = emissionprob_
    # model.startprob_ = startprob_

    z = itr // (divisions[iteration, 1] * divisions[iteration, 2])
    tr = itr - z * (divisions[iteration, 1] * divisions[iteration, 2])
    y = tr // divisions[iteration, 2]
    x = tr - y * divisions[iteration, 2]

    f2 = h5py.File(write_file2, 'r')
    data = f2.get(list(f2.keys())[0].encode('ascii', 'ignore'))
    while isinstance(data, h5py.Group):
        data = data.get(list(data.keys())[0].encode('ascii', 'ignore'))

    tmr = data[:,
               :,
               ((z * b[iteration, 0]) + start[iteration, 0]):(((z + 1) * b[iteration, 0]) + start[iteration, 0]),
               ((y * b[iteration, 1]) + start[iteration, 1]):(((y + 1) * b[iteration, 1]) + start[iteration, 1]),
               ((x * b[iteration, 2]) + start[iteration, 2]):(((x + 1) * b[iteration, 2]) + start[iteration, 2])]
    tm1 = np.max(tmr, 1)
    tm2 = np.argmax(tmr, 1)
    del tmr
    tm1 = tm1 * n_classes * binn / (n_classes - 1) - binn / (n_classes - 1) - 1
    tm1 = np.ceil(tm1)
    tm1 = binn * tm2 + tm1
    del tm2
    Samy = tm1.copy()
    repl = np.argmax(emissionprob_, 0) + 1000
    for i in range(0, repl.shape[0]):
        Samy[Samy == i] = repl[i]
    Samy = Samy - 1000
    tm1_5 = np.sum(tm1, 0) - (np.min(tm1, 0)*int(tm1.shape[0]))
    nzr = np.stack(np.nonzero(tm1_5))
    for i in range(0, nzr.shape[1]):
        tmp = model.viterbi(tm1[:, nzr[0, i], nzr[1, i], nzr[2, i]])[1]
        Samy[:, nzr[0, i], nzr[1, i], nzr[2, i]] = [tmp[g][0] for g in range(1, time_size+1)]
        # Samy[:, nzr[0, i], nzr[1, i], nzr[2, i]] = model.predict(np.expand_dims(tm1[:, nzr[0, i], nzr[1, i], nzr[2, i]], axis=1).astype(int))
    del tm1
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
if pomegranate.utils.is_gpu_enabled():
    print('GPU is Enabled')
for k in range(0, iterations):
    stdfun = partial(rot, k)
    vr = np.stack(pool.imap(stdfun, range(0, int(np.prod(divisions[k, :])))))
    f = h5py.File(write_file, 'a')
    for t in range(0, int(vr.shape[0])):
        z = t // (divisions[k, 1] * divisions[k, 2])
        tr = t - z * (divisions[k, 1] * divisions[k, 2])
        y = tr // divisions[k, 2]
        x = tr - y * divisions[k, 2]
        f['data'][:,
                  ((z * b[k, 0]) + start[k, 0]):(((z + 1) * b[k, 0]) + start[k, 0]),
                  ((y * b[k, 1]) + start[k, 1]):(((y + 1) * b[k, 1]) + start[k, 1]),
                  ((x * b[k, 2]) + start[k, 2]):(((x + 1) * b[k, 2]) + start[k, 2])] = vr[t, :]
    f.close()
    print('Done')
