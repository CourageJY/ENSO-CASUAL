import os
import numpy as np
import netCDF4 as nc

from PIL import Image

import sys
sys.path.append("")

from model.params import *

#resolve for pwat/cwat/rh/uwind/vwind

# 40*55
pwat = np.load(f"{params.reanalysis_dir}/meta_data/final/pwat.npz")['pwat'].reshape((-1, 91, 180))[0:1740, 25:65, 80:135]
cwat = np.load(f"{params.reanalysis_dir}/meta_data/final/cwat.npz")['cwat'].reshape((-1, 91, 180))[0:1740, 25:65, 80:135]
rh = np.load(f"{params.reanalysis_dir}/meta_data/final/rh.npz")['rh'].reshape((-1, 91, 180))[0:1740, 25:65, 80:135]
uwind = np.load(f"{params.reanalysis_dir}/meta_data/final/uwind.npz")['uwind'].reshape((-1, 91, 180))[0:1740, 25:65, 80:135]
vwind = np.load(f"{params.reanalysis_dir}/meta_data/final/vwind.npz")['vwind'].reshape((-1, 91, 180))[0:1740, 25:65, 80:135]

print(pwat.shape, cwat.shape, rh.shape, uwind.shape, vwind.shape)

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
