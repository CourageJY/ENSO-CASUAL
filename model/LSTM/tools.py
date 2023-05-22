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

def get_var_casual_data(data_type,casual_type,casual_algorithm,var_num,month_weight=True):
    #load the encodered data
    data_e=[]
    for var in params.variables:
        data_e.append(np.load(f'{params.encoder_save_dir}/{data_type}/{var}-encoder.npz')[var])

    #get the month weight
    m_w=get_month_weight(data_e[0].shape[0])

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
    casual_merge=[]
    for i in range(len(data_nums[var_num])):
        casual_single=data_e[var_num][:,data_nums[var_num][i]].reshape(-1,1)#self
        
        if('internal' in casual_type):
            #internal casual
            internal_casual=np.load(f'./model/CasualDiscovery/graph_storage/{data_type}/{params.variables[var_num]}-internal.npz')[data_names[var_num][i]]
            for name in internal_casual:
                k=get_num_from_name(data_names[var_num],name)
                casual_single=np.concatenate((casual_single,data_e[var_num][:,data_nums[var_num][k]].reshape(-1,1)),axis=1)
        if('external' in casual_type):
            #external casual      
            nodes=set()
            nodes.add(data_names[var_num][i])
            for j in range(0,6):
                if(j==var_num):continue
                external_casual=np.load(f'./model/CasualDiscovery/graph_storage/'
                                        +f'{data_type}/{casual_algorithm}-external/{params.variables[j]}-{params.variables[var_num]}-casual-{casual_algorithm}.npz')[data_names[var_num][i]]
                for name in external_casual:
                    if name[0] not in nodes and int(name[1]) >= -3:#去除本身以及其它重复元素,并保证只取在三个月内因果相关的数据
                        k=get_num_from_name(data_names[j],name[0])#获得编号
                        casual_single=np.concatenate((casual_single,data_e[j][:,data_nums[j][k]].reshape(-1,1)),axis=1)
        if month_weight:#添加月份权重
            casual_single=np.concatenate((casual_single,m_w),axis=1)
        casual_merge.append(casual_single)

        print(casual_single.shape)
    
    return casual_merge,data_nums[var_num]
            
def get_num_from_name(list,name):
    for i in range(len(list)):
        if(list[i]==name):
            return i
    return -1


def get_month_weight(lens):
    m_w=[]
    for i in range(lens):
        m_w.append((i%12+1)/10)
    return np.array(m_w).reshape(lens,1)

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
    data=get_var_casual_data('reanalysis')
    print("end")