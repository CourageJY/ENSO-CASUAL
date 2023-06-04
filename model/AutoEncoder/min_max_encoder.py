import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import sys

sys.path.append("")
from model.params import *

data_type='reanalysis'

#load the encoder data
data_e=[]
for var in params.variables:
    e=np.load(f'{params.encoder_save_dir}/{data_type}/{var}-encoder.npz')[var]
    data_e.append(e)

#min-max归一化
scaler = MinMaxScaler()
data_m=[]
for i in range(len(data_e)):
    m=scaler.fit_transform(data_e[i])
    data_m.append(m)

#save
for i in range(len(params.variables)):
    np.savez(f'{params.encoder_save_dir}/{data_type}/{params.remote_sensing_variables[i]}-min-max-encoder.npz',**{params.variables[i]:data_m[i]})

print("end")