import numpy as np
import tensorflow as tf
import sys
sys.path.append("")

from model.params import *
from forecast.tools import *

def work_flow(orgin_data):
    data_type='reanalysis'
    casual_type='external'

    #get encoder model and info
    data_names,data_nums=get_enocder_info(data_type)
    encoder_models=get_encoder_model(data_type)

    #dispose the orginal data to encodered data
    data_es=[]#(batch,features,sequence_length,40,55)
    for data in orgin_data:#data:(features,sequence_length,40,55)
        data_e=[]
        for j in len(data):
            data_e.append(encoder_models[j].encoder(data[j]))#(sequence_length,40,55))
        data_es.append(data_e)
    
    #get casual data
    sst_casual_merge_all=[]#(batch,……)
    for data_e in data_es:
        sst_casual_merge=[]
        for i in range(len(data_nums[0])):
            sst_casual_single=data_e[0][:,data_nums[0][i]].reshape(-1,1)#self
            #internal casual
            internal_casual=np.load(f'./model/CasualDiscovery/graph_storage/{data_type}/sst-internal.npz')[data_names[0][i]]
            for name in internal_casual:
                k=get_num_from_name(data_names[0],name)
                sst_casual_single=np.concatenate((sst_casual_single,data_e[0][:,data_nums[0][k]].reshape(-1,1)),axis=1)
            #external casual      
            nodes=set()
            nodes.add(data_names[0][i])
            for j in range(1,6):
                external_casual=np.load(f'./model/CasualDiscovery/graph_storage/'
                                        +f'{data_type}/{params.variables[j]}-sst-casual-pc.npz')[data_names[0][i]]
                for name in external_casual:
                    if name[0] not in nodes:#去除本身以及其它重复元素
                        k=get_num_from_name(data_names[j],name[0])#获得编号
                        sst_casual_single=np.concatenate((sst_casual_single,data_e[j][:,data_nums[j][k]].reshape(-1,1)),axis=1)
            sst_casual_merge.append(sst_casual_single)
        sst_casual_merge_all.append(sst_casual_merge)
    
    #get lstm models
    lstm_models=get_lstm_model(data_type,casual_type,data_nums[0])

    #dispose the encodered data by lstm models
    #todo:
    # datas_by_lstm=[]
    # for sst_casual_merge in sst_casual_merge_all:
    #     data_by_lstm=[]
    #     for model in lstm_models:



    #decoder data

                
