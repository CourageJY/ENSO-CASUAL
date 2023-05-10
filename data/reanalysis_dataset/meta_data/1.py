import numpy as np
import sys

sys.path.append("")
from model.params import *
sst=np.load(f"{params.reanalysis_npz_dir}/sst-resolve.npz")['sst']
uwind=np.load(f"{params.reanalysis_npz_dir}/uwind-resolve.npz")['uwind']

print(sst.shape)
print(uwind.shape)

print("end")

# a=np.array([[1,2,3],
#             [2,3,4,]])

# b=a[0:,1:2]

