import numpy as np
import sys
from sklearn.model_selection import train_test_split

sys.path.append("")
from model.AutoEncoder.auto_encoder import *
from model.LSTM.lstm_model import *
from model.params import *

def get_enocder_info(data_type):
    #load the encodered data
    data_e=[]
    for var in params.variables:
        data_e.append(np.load(f'{params.encoder_save_dir}/{data_type}/{var}-encoder.npz')[var])
    #--------get the num of no_zero in data----------
    data_names,data_nums=[],[]
    for i in range(len(data_e)):
        names,nums=[],[]
        for j in range(data_e[i].shape[1]):
            if(data_e[i][0:,j:j+1].sum()>0):#not zero
                names.append(params.variables[i]+str(j))
                nums.append(j)
        data_names.append(names)
        data_nums.append(nums)
    
    return data_names,data_nums

def get_encoder_model(data_type):
    #load model
    encoder_models=[]
    for var in params.remote_sensing_variables:
        model=Autoencoder(params.latent_dim)
        model.load_weights(f'{params.encoder_save_dir}/{data_type}/{var}-model')
        encoder_models.append(model)
    return encoder_models

def get_num_from_name(list,name):
    for i in range(len(list)):
        if(list[i]==name):
            return i
    return -1

def get_lstm_model(data_type,casual_type,sst_nums):
    lstm_models=[]
    for num in sst_nums:
        model=LSTM_model()
        model.load_weights(f'./model/LSTM/model_storage/{data_type}/{casual_type}/sst-{num}-model')
        lstm_models.append(model)
    
    return lstm_models
