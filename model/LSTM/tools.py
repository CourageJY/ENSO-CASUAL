import numpy as np
import sys
from sklearn.model_selection import train_test_split

sys.path.append("")
from model.AutoEncoder.auto_encoder import *
from model.params import *

#train_data:(batch_size,sequence_length,features) train_label:(batch_size,1,1)
def split_series_data(data,sequence_length,target_id):
    train,test=train_test_split(data,test_size=params.train_eval_split,shuffle=False)
    print(train.shape,test.shape)
    #train
    train_data,train_label=[],[]
    for i in range(train.shape[0]-sequence_length):
        train_data.append(train[i:i+sequence_length])
        train_label.append(train[i+sequence_length][target_id])
    train_data=np.array(train_data)
    train_label=np.array(train_label).reshape(-1,1,1)
    print(train_data.shape,train_label.shape)
    #test
    test_data,test_label=[],[]
    for i in range(test.shape[0]-sequence_length):
        test_data.append(test[i:i+sequence_length])
        test_label.append(test[i+sequence_length][target_id])
    test_data=np.array(test_data)
    test_label=np.array(test_label).reshape(-1,1,1)
    print(test_data.shape,test_label.shape)

    return (train_data,train_label),(test_data,test_label)

def get_sst_casual_data(data_type):
    #load the encodered data
    data_e=[]
    for var in params.variables:
        data_e.append(np.load(f'{params.encoder_save_dir}/{data_type}/{var}-encoder.npz')[var])
    
    #load the external casual
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
    # get res
    sst_casual_merge=[]
    for i in range(len(data_nums[0])):
        sst_casual_single=data_e[0][:,data_nums[0][i]].reshape(-1,1)#self
        #internal casual
        # internal_casual=np.load(f'./model/CasualDiscovery/graph_storage/{data_type}/sst-internal.npz')[data_names[0][i]]
        # for name in internal_casual:
        #     k=get_num_from_name(data_names[0],name)
        #     sst_casual_single=np.concatenate((sst_casual_single,data_e[0][:,data_nums[0][k]].reshape(-1,1)),axis=1)
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

        print(sst_casual_single.shape)
    
    return sst_casual_merge,data_nums[0]
            
def get_num_from_name(list,name):
    for i in range(len(list)):
        if(list[i]==name):
            return i
    return -1

if __name__=='__main__':
    # data_type='reanalysis'
    # sst_e=np.load(f'{params.encoder_save_dir}/{data_type}/sst-encoder.npz')['sst']
    # no_zero_nums=[]
    # for i in range(sst_e.shape[1]):
    #     if(sst_e[:,i:i+1].sum()>0):
    #         no_zero_nums.append(i)
    
    # #test
    # data=sst_e[:,no_zero_nums[0]].reshape(-1,1)
    # print(data.shape)
    # (train_data,train_label),(test_data,test_label)=split_series_data(data,6,0)
    data=get_sst_casual_data('reanalysis')
    print("end")