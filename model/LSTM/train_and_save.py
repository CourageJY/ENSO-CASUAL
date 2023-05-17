import numpy as np
import sys
from sklearn.model_selection import train_test_split

sys.path.append("")
from model.AutoEncoder.auto_encoder import *
from model.params import *
from model.LSTM.tools import *
from model.LSTM.lstm_model import *

if __name__=='__main__':
    data_type='reanalysis'
    casual_type='external'
    sst_casual,sst_nums=get_sst_casual_data(data_type)
    for i in range(len(sst_casual)):
        print("---------------------------")
        print("-------Step{}--------------".format(i+1))
        print("---------------------------")
        (train_data,train_label),(test_data,test_label)=split_series_data(sst_casual[i],params.sequence_length,0)
        with tf.device('/CPU:0'):
            lstm_model=LSTM_model()
            lstm_model.compile_and_fit(train_data,train_label,test_data,test_label)

            # Save the weights
            lstm_model.save_weights(f'./model/LSTM/model_storage/{data_type}/{casual_type}/sst-{sst_nums[i]}-model')

    print("end")
