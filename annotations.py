import torch
import h5py

name='annotations3.1'
classes=3
f=h5py.File((('%s.h5') % name),'r')
data=f.get(f.keys()[0].encode('ascii','ignore'))
f2=h5py.File((('%s_sep.h5') % name),'w')
dset = f2.create_dataset(f.keys()[0].encode('ascii','ignore'), (classes, data.shape[0], data.shape[1], data.shape[2]), chunks=True)

for i in range(0,data.shape[0]):
    torch_data=torch.from_numpy(data[i,:,:])
    for j in range(0,classes):
        #d=1-torch.abs(torch.sign(torch_data-j))
        d=(torch_data==j)
        dset[j,i,:,:]=d
