import os
import numpy as np
import netCDF4 as nc

from PIL import Image

import sys
sys.path.append("")

from model.params import *

#resolve for pwat/cwat/rh/uwind/vwind

#(1870-2014)
pwat = np.load(f"{params.reanalysis_dir}/meta_data/final/pwat.npz")['pwat'].reshape((-1, 91, 180))[228:, 25:65, 80:135]
cwat = np.load(f"{params.reanalysis_dir}/meta_data/final/cwat.npz")['cwat'].reshape((-1, 91, 180))[228:, 25:65, 80:135]
rh = np.load(f"{params.reanalysis_dir}/meta_data/final/rh.npz")['rh'].reshape((-1, 91, 180))[228:, 25:65, 80:135]
uwind = np.load(f"{params.reanalysis_dir}/meta_data/final/uwind.npz")['uwind'].reshape((-1, 91, 180))[228:, 25:65, 80:135]
vwind = np.load(f"{params.reanalysis_dir}/meta_data/final/vwind.npz")['vwind'].reshape((-1, 91, 180))[228:, 25:65, 80:135]
sst_l = nc.Dataset(f"{params.reanalysis_dir}/meta_data/final/HadISST_sst.nc").variables['sst'][0:1740, 50:130, 340:360]
sst_r = nc.Dataset(f"{params.reanalysis_dir}/meta_data/final/HadISST_sst.nc").variables['sst'][0:1740, 50:130, 0:90]
sst = np.concatenate((sst_l, sst_r), axis=2).filled()
sst[sst == -1.0e+30] = 0

print(pwat.shape, cwat.shape, rh.shape, uwind.shape, vwind.shape,sst.shape)

expand=2

pwat_resolve = []
for i in range(pwat.shape[0]):
    data = pwat[i]
    size = tuple((data.shape[1] * expand, data.shape[0] * expand))
    data = np.array(Image.fromarray(data).resize(size, Image.BICUBIC))#扩张成320*440
    pwat_resolve.append(data)

cwat_resolve = []
for i in range(cwat.shape[0]):
    data = cwat[i]
    size = tuple((data.shape[1] * expand, data.shape[0] * expand))
    data = np.array(Image.fromarray(data).resize(size, Image.BICUBIC))
    cwat_resolve.append(data)

rh_resolve = []
for i in range(rh.shape[0]):
    data = rh[i]
    size = tuple((data.shape[1] * expand, data.shape[0] * expand))
    data = np.array(Image.fromarray(data).resize(size, Image.BICUBIC))
    rh_resolve.append(data)

uwind_resolve = []
for i in range(uwind.shape[0]):
    data = uwind[i]
    size = tuple((data.shape[1] * expand, data.shape[0] * expand))
    data = np.array(Image.fromarray(data).resize(size, Image.BICUBIC))
    uwind_resolve.append(data)

vwind_resolve = []
for i in range(vwind.shape[0]):
    data = vwind[i]
    size = tuple((data.shape[1] * expand, data.shape[0] * expand))
    data = np.array(Image.fromarray(data).resize(size, Image.BICUBIC))
    vwind_resolve.append(data)

#取nino34区域 35-45；30-80

pwat_resolve=np.array(pwat_resolve)[:,35:45,30:80]
cwat_resolve=np.array(cwat_resolve)[:,35:45,30:80]
rh_resolve=np.array(rh_resolve)[:,35:45,30:80]
uwind_resolve=np.array(uwind_resolve)[:,35:45,30:80]
vwind_resolve=np.array(vwind_resolve)[:,35:45,30:80]
sst=np.array(sst)[:,35:45,30:80]

print(pwat_resolve.shape, cwat_resolve.shape, rh_resolve.shape, uwind_resolve.shape, vwind_resolve.shape,sst.shape)

#save
data = {'pwat': pwat_resolve}
np.savez(f'{params.reanalysis_npz_dir}/pwat-resolve.npz', **data)
data = {'cwat': cwat_resolve}
np.savez(f'{params.reanalysis_npz_dir}/cwat-resolve.npz', **data)
data = {'rh': rh_resolve}
np.savez(f'{params.reanalysis_npz_dir}/rh-resolve.npz', **data)
data = {'uwind': uwind_resolve}
np.savez(f'{params.reanalysis_npz_dir}/uwind-resolve.npz', **data)
data = {'vwind': vwind_resolve}
np.savez(f'{params.reanalysis_npz_dir}/vwind-resolve.npz', **data)
data = {'sst': sst}
np.savez(f'{params.reanalysis_npz_dir}/sst-resolve.npz', **data)
