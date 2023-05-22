#[1851-1995]
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
    #-----Reanlysis-----
    # (1740,40,55) or (1740,80,110)
    sst = np.load(f"{params.reanalysis_npz_dir}/sst-resolve.npz")['sst']
    uwind = np.load(f"{params.reanalysis_npz_dir}/uwind-resolve.npz")['uwind']
    vwind = np.load(f"{params.reanalysis_npz_dir}/vwind-resolve.npz")['vwind']
    vapor = np.load(f"{params.reanalysis_npz_dir}/rh-resolve.npz")['rh']
    cloud = np.load(f"{params.reanalysis_npz_dir}/cwat-resolve.npz")['cwat']
    rain = np.load(f"{params.reanalysis_npz_dir}/pwat-resolve.npz")['pwat']
    
    print(sst.shape)

    #将过小数据记为0
    sst[abs(sst) < 8e-17] = 0
    vapor[abs(vapor) < 5e-16] = 0

    # (1740,80,110) =>(1740,40,55) 通过取平均值来放缩数据，降低sst分辨率
    bar = PixelBar(r'Generating', max=sst.shape[0], suffix='%(percent)d%%')#进度条显示
    sst_=[]
    shink_size=2
    for i in range(sst.shape[0]):
        sst_.append([])
        for j in range(int(sst.shape[1]/shink_size)):
            sst_[i].append([])
            for k in range(int(sst.shape[2]/shink_size)):
                h=shink_size
                sst_[i][j].append(sst[i:i+1,j*h:j*h+h,k*h:k*h+h].mean())
        bar.next()
    
    bar.finish()
    sst=np.array(sst_)
    print(sst.shape)

    #save
    np.savez(f"{params.reanalysis_npz_dir}/sst-resolve-shink.npz",**{'sst':sst})
    
    #对数据进行归一化
    scaler = MinMaxScaler()

    sst=np.reshape(scaler.fit_transform(np.reshape(sst, (-1, 40*55))), (-1, 40, 55))
    uwind=np.reshape(scaler.fit_transform(np.reshape(uwind, (-1, 40*55))), (-1, 40, 55))
    vwind=np.reshape(scaler.fit_transform(np.reshape(vwind, (-1, 40*55))), (-1, 40, 55))
    cloud=np.reshape(scaler.fit_transform(np.reshape(cloud, (-1, 40*55))), (-1, 40, 55))
    vapor=np.reshape(scaler.fit_transform(np.reshape(vapor, (-1, 40*55))), (-1, 40, 55))
    rain=np.reshape(scaler.fit_transform(np.reshape(rain, (-1, 40*55))), (-1, 40, 55))
    # sst=(sst-sst.min())/(sst.max()-sst.min())
    # uwind=(uwind-uwind.min())/(uwind.max()-uwind.min())
    # vwind=(vwind-vwind.min())/(vwind.max()-vwind.min())
    # cloud=(vapor-vapor.min())/(cloud.max()-vapor.min())
    # vapor=(vapor-vapor.min())/(vapor.max()-vapor.min())
    # rain=(rain-rain.min())/(rain.max()-rain.min())

    return sst,uwind,vwind,vapor,cloud,rain

# ---------- Go! ----------
if __name__ == "__main__":
    print("Start!")

    sst,uwind,vwind,vapor,cloud,rain=parse_npz_data()
    np.savez(f'./data/final/reanalysis/sst-final.npz', **{'sst': sst})
    np.savez(f'./data/final/reanalysis/uwind-final.npz', **{'uwind': uwind})
    np.savez(f'./data/final/reanalysis/vwind-final.npz', **{'vwind': vwind})
    np.savez(f'./data/final/reanalysis/vapor-final.npz', **{'vapor': vapor})
    np.savez(f'./data/final/reanalysis/cloud-final.npz', **{'cloud': cloud})
    np.savez(f'./data/final/reanalysis/rain-final.npz', **{'rain': rain})

    print("Done!")
