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

data = {'pwat': np.array(pwat)}
np.savez(f'{params.reanalysis_npz_dir}/pwat-resolve.npz', **data)
data = {'cwat': np.array(cwat)}
np.savez(f'{params.reanalysis_npz_dir}/cwat-resolve.npz', **data)
data = {'rh': np.array(rh)}
np.savez(f'{params.reanalysis_npz_dir}/rh-resolve.npz', **data)
data = {'uwind': np.array(uwind)}
np.savez(f'{params.reanalysis_npz_dir}/uwind-resolve.npz', **data)
data = {'vwind': np.array(vwind)}
np.savez(f'{params.reanalysis_npz_dir}/vwind-resolve.npz', **data)
data = {'sst': np.array(sst)}
np.savez(f'{params.reanalysis_npz_dir}/sst-resolve.npz', **data)
