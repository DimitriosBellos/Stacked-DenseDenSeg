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

#write_file = '/db/user/4D/hmm1_5.h5'
#write_file2 = '/db/user/4D/original.h5'
write_file = '/mnt/Wolverine/4D/hmm1_5.h5'
write_file2 = '/mnt/Wolverine/4D/original.h5'
#write_file2 = '/db/user/4D/Survos_low_quality_annotation2/annotations/annotations0.h5'
#write_file = '/db/user/4D/lq_annotations.h5'
time_size = 21
start = np.mgrid[1190:1191, 806:807, 50:51]  # 1702
start = np.rollaxis(start, 0, 4)
start = start.reshape(1, 3)
end = np.mgrid[1191:1192, 1414:1415, 562:563]  # 1703
end = np.rollaxis(end, 0, 4)
end = end.reshape(1, 3)
iterations = start.shape[0]
slices = end - start
cpus = 8
divisions = np.array([[1, 4, 4], ])
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
    Samy = data[:,
                ((z * b[iteration, 0]) + start[iteration, 0]):(((z + 1) * b[iteration, 0]) + start[iteration, 0]),
                ((y * b[iteration, 1]) + start[iteration, 1]):(((y + 1) * b[iteration, 1]) + start[iteration, 1]),
                ((x * b[iteration, 2]) + start[iteration, 2]):(((x + 1) * b[iteration, 2]) + start[iteration, 2])]
    tm1_5 = np.sum(Samy, 0) - (np.min(Samy, 0)*int(Samy.shape[0]))
    nzr = np.stack(np.nonzero(tm1_5))
    for i in range(0, nzr.shape[1]):
        tmp = model.viterbi(Samy[:, nzr[0, i], nzr[1, i], nzr[2, i]])[1]
        Samy[:, nzr[0, i], nzr[1, i], nzr[2, i]] = [tmp[g][0] for g in range(1, time_size+1)]
        # Samy[:, nzr[0, i], nzr[1, i], nzr[2, i]] = model.predict(np.expand_dims(tm1[:, nzr[0, i], nzr[1, i], nzr[2, i]], axis=1).astype(int))
    return Samy  # np.array((Sam, Samy))


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
