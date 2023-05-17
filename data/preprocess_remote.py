import os
import sys
import json
import random
import numpy as np
import tensorflow as tf
import netCDF4 as nc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from progress.bar import PixelBar

sys.path.append("")
from model.params import *

def get_mean(data,j,k,h):
    all=0
    for u in np.arange(j,j+h):
        for v in np.arange(k,k+h):
            all+=data[u][v]
    return all/(h*h)

# ---------- Prepare Data ----------
def parse_npz_data():
    #-----REMSS------
    # (279,320,440)
    sst = np.load(f"{params.remote_sensing_npz_dir}/sst-no-nan.npz")['sst']
    uwind = np.load(f"{params.reanalysis_npz_dir}/uwind-resolve.npz")['uwind']
    vwind = np.load(f"{params.reanalysis_npz_dir}/vwind-resolve.npz")['vwind']
    vapor = np.load(f"{params.remote_sensing_npz_dir}/vapor-no-nan.npz")['vapor']
    cloud = np.load(f"{params.remote_sensing_npz_dir}/cloud-no-nan.npz")['cloud']
    rain = np.load(f"{params.remote_sensing_npz_dir}/rain-no-nan.npz")['rain']

    print(sst.shape)

    #将过小数据记为0
    sst[abs(sst) < 8e-17] = 0
    vapor[abs(vapor) < 5e-16] = 0

    # (279,320,440) =>(279,320/shrink_size,440/shrink_size) 通过取平均值来放缩数据，降低分辨率
    bar = PixelBar(r'Generating', max=sst.shape[0], suffix='%(percent)d%%')#进度条显示
    sst_, uwind_,vwind_,vapor_,cloud_,rain_=[],[],[],[],[],[]
    shrink_size=8
    for i in range(sst.shape[0]):
        sst_.append([])
        uwind_.append([])
        vwind_.append([])
        vapor_.append([])
        cloud_.append([])
        rain_.append([])
        for j in range(int(sst.shape[1]/shrink_size)):
            sst_[i].append([])
            uwind_[i].append([])
            vwind_[i].append([])
            vapor_[i].append([])
            cloud_[i].append([])
            rain_[i].append([])
            for k in range(int(sst.shape[2]/shrink_size)):
                h=shrink_size
                sst_[i][j].append(sst[i:i+1,j*h:j*h+h,k*h:k*h+h].mean())
                uwind_[i][j].append(uwind[i:i+1,j*h:j*h+h,k*h:k*h+h].mean())
                vwind_[i][j].append(vwind[i:i+1,j*h:j*h+h,k*h:k*h+h].mean())
                vapor_[i][j].append(vapor[i:i+1,j*h:j*h+h,k*h:k*h+h].mean())
                cloud_[i][j].append(cloud[i:i+1,j*h:j*h+h,k*h:k*h+h].mean())
                rain_[i][j].append(rain[i:i+1,j*h:j*h+h,k*h:k*h+h].mean())
        bar.next()
    
    bar.finish()
    sst=np.array(sst_)
    uwind=np.array(uwind_)
    vwind=np.array(vwind_)
    vapor=np.array(vapor_)
    cloud=np.array(cloud_)
    rain=np.array(rain_)
    print(sst.shape)
    
    #对数据进行归一化
    sst=(sst-sst.min())/(sst.max()-sst.min())
    uwind=(uwind-uwind.min())/(uwind.max()-uwind.min())
    vwind=(vwind-vwind.min())/(vwind.max()-vwind.min())
    vapor=(vapor-vapor.min())/(vapor.max()-vapor.min())
    vapor=(vapor-vapor.min())/(vapor.max()-vapor.min())
    rain=(rain-rain.min())/(rain.max()-rain.min())

    return sst,uwind,vwind,vapor,cloud,rain

# ---------- Go! ----------
if __name__ == "__main__":
    print("Start!")

    sst,uwind,vwind,vapor,cloud,rain=parse_npz_data()
    np.savez(f'./data/final/remote/sst-final.npz', **{'sst': sst})
    np.savez(f'./data/final/remote/uwind-final.npz', **{'uwind': uwind})
    np.savez(f'./data/final/remote/vwind-final.npz', **{'vwind': vwind})
    np.savez(f'./data/final/remote/vapor-final.npz', **{'vapor': vapor})
    np.savez(f'./data/final/remote/cloud-final.npz', **{'cloud': cloud})
    np.savez(f'./data/final/remote/rain-final.npz', **{'rain': rain})

    print("Done!")
