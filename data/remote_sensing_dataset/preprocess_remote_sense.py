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


# ---------- Helpers ----------
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# ---------- Prepare Data ----------
def parse_npz_data():
    # (123,320,440)
    sst = np.load(f"{params.remote_sensing_npz_dir}/sst-no-nan.npz")['sst']
    uwind = np.load(f"{params.remote_sensing_npz_dir}/uwind-no-nan.npz")['uwind']
    vwind = np.load(f"{params.remote_sensing_npz_dir}/vwind-no-nan.npz")['vwind']
    vapor = np.load(f"{params.remote_sensing_npz_dir}/vapor-no-nan.npz")['vapor']
    cloud = np.load(f"{params.remote_sensing_npz_dir}/cloud-no-nan.npz")['cloud']
    rain = np.load(f"{params.remote_sensing_npz_dir}/rain-no-nan.npz")['rain']

    #将过小数据记为0
    sst[abs(sst) < 8e-17] = 0
    vapor[abs(vapor) < 5e-16] = 0

    # (123,320,440) =>(123,32,44) 通过取平均值来放缩数据，降低分辨率
    bar = PixelBar(r'Generating', max=sst.shape[0], suffix='%(percent)d%%')#进度条显示
    sst_, uwind_,vwind_,vapor_,cloud_,rain_=[],[],[],[],[],[]
    for i in range(sst.shape[0]):
        sst_.append([])
        uwind_.append([])
        vwind_.append([])
        vapor_.append([])
        cloud_.append([])
        rain_.append([])
        for j in range(int(sst.shape[1]/params.shrink_size)):
            sst_[i].append([])
            uwind_[i].append([])
            vwind_[i].append([])
            vapor_[i].append([])
            cloud_[i].append([])
            rain_[i].append([])
            for k in range(int(sst.shape[2]/params.shrink_size)):
                sst_[i][j].append(sst[i:i+1].reshape(320,440)[j:j+10].transpose()[k:k+10].transpose().mean())
                uwind_[i][j].append(uwind[i:i+1].reshape(320,440)[j:j+10].transpose()[k:k+10].transpose().mean())
                vwind_[i][j].append(vwind[i:i+1].reshape(320,440)[j:j+10].transpose()[k:k+10].transpose().mean())
                vapor_[i][j].append(vapor[i:i+1].reshape(320,440)[j:j+10].transpose()[k:k+10].transpose().mean())
                cloud_[i][j].append(cloud[i:i+1].reshape(320,440)[j:j+10].transpose()[k:k+10].transpose().mean())
                rain_[i][j].append(rain[i:i+1].reshape(320,440)[j:j+10].transpose()[k:k+10].transpose().mean())
        bar.next()
    
    bar.finish()
    sst=np.array(sst_)
    uwind=np.array(uwind_)
    vwind=np.array(vwind_)
    vapor=np.array(vapor_)
    cloud=np.array(cloud_)
    rain=np.array(rain_)
    print(sst.shape)
    
    #采用minmax进行数据归一化
    scaler = MinMaxScaler() #StandardScaler() Normalizer()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst, (-1, 32*44))), (-1, 32, 44))
    uwind = np.reshape(scaler.fit_transform(np.reshape(uwind, (-1, 32*44))), (-1, 32, 44))
    vwind = np.reshape(scaler.fit_transform(np.reshape(vwind, (-1, 32*44))), (-1, 32, 44))
    vapor = np.reshape(scaler.fit_transform(np.reshape(vapor, (-1, 32*44))), (-1, 32, 44))
    cloud = np.reshape(scaler.fit_transform(np.reshape(cloud, (-1, 32*44))), (-1, 32, 44))
    rain = np.reshape(scaler.fit_transform(np.reshape(rain, (-1, 32*44))), (-1, 32, 44))

    return sst,uwind,vwind,vapor,cloud,rain

# ---------- Go! ----------
if __name__ == "__main__":
    print("Start!")

    sst,uwind,vwind,vapor,cloud,rain=parse_npz_data()
    np.savez(f'./data/remote_sensing_dataset/final/sst-no-nan-final.npz', **{'sst': sst})
    np.savez(f'./data/remote_sensing_dataset/final/uwind-no-nan-final.npz', **{'uwind': uwind})
    np.savez(f'./data/remote_sensing_dataset/final/vwind-no-nan-final.npz', **{'vwind': vwind})
    np.savez(f'./data/remote_sensing_dataset/final/vapor-no-nan-final.npz', **{'vapor': vapor})
    np.savez(f'./data/remote_sensing_dataset/final/cloud-no-nan-final.npz', **{'cloud': cloud})
    np.savez(f'./data/remote_sensing_dataset/final/rain-no-nan-final.npz', **{'rain': rain})

    print("Done!")
