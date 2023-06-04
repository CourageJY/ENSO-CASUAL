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
    casual_type='internal_and_external'
    casual_algorithm='ganger'
    for u in range(len(params.variables)):
        #if u==0 or u==1:continue
        var_casual,var_nums=get_var_casual_data(data_type,casual_type,casual_algorithm,u,month_weight=True)
        for i in range(len(var_casual)):
            print("---------------------------")
            print("-------Step{}--------------".format(i+1))
            print("---------------------------")
            (train_data,train_label),(test_data,test_label)=split_series_data(var_casual[i],params.sequence_length,0)
            with tf.device('/CPU:0'):
                lstm_model=LSTM_model()
                history=lstm_model.compile_and_fit(train_data,train_label,test_data,test_label)
                #save history
                np.savez(f'./model/LSTM/history/{params.variables[u]}-{i}-{params.sequence_length}-{casual_type}.npz',**(history.history))
                # Save the weights
                lstm_model.save_weights(f'./model/LSTM/model_storage/{data_type}/{casual_type}-{casual_algorithm}/{params.variables[u]}-{var_nums[i]}-model')

    print("end")
