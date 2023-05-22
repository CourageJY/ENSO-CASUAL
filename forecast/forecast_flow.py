import numpy as np
import tensorflow as tf
import sys
sys.path.append("")

from model.params import *
from forecast.tools import *
from model.Tools.draw_heat_map import *
from sklearn.metrics import mean_squared_error, r2_score
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
import cmaps
from progress.bar import PixelBar

def predict_single(orgin_data,var_num):#return list of numpy which shape is (40,55),the last of orgin_data is month weight
    data_type='reanalysis'
    casual_type='internal_and_external'
    casual_algorithm='ganger'

    #get encoder model and info
    data_names,data_nums=get_enocder_info(data_type)
    encoder_models=get_encoder_model(data_type)

    #dispose the orginal data to encodered data
    data_es=[]#orgin_data:(batch,features,sequence_length,40,55)
    for data in orgin_data:#data:(features,sequence_length,40,55)
        data_e=[]
        for j in range(len(data)-1):
            data_e.append(encoder_models[j].encoder(data[j]).numpy())#(sequence_length,40,55))
        data_e.append(data[-1])
        data_es.append(data_e) #data_es:(features,sequence_length,64)
    
    #get casual data
    casual_merge_all=[]#(batch,……)
    for data_e in data_es:
        casual_merge=[]
        for i in range(len(data_nums[var_num])):
            casual_single=data_e[var_num][:,data_nums[var_num][i]].reshape(-1,1)#self
            #internal casual
            internal_casual=np.load(f'./model/CasualDiscovery/graph_storage/{data_type}/{params.variables[var_num]}-internal.npz')[data_names[var_num][i]]
            for name in internal_casual:
                k=get_num_from_name(data_names[var_num],name)
                casual_single=np.concatenate((casual_single,data_e[var_num][:,data_nums[var_num][k]].reshape(-1,1)),axis=1)
            #external casual      
            nodes=set()
            nodes.add(data_names[var_num][i])
            for j in range(0,6):
                if(j==var_num):continue
                external_casual=np.load(f'./model/CasualDiscovery/graph_storage/{data_type}/{casual_algorithm}-external/{params.variables[j]}-{params.variables[var_num]}-casual-{casual_algorithm}.npz')[data_names[var_num][i]]
                for name in external_casual:
                    if name[0] not in nodes and int(name[1]) >= -3:#去除本身以及其它重复元素
                        k=get_num_from_name(data_names[j],name[0])#获得编号
                        casual_single=np.concatenate((casual_single,data_e[j][:,data_nums[j][k]].reshape(-1,1)),axis=1)
            casual_single=np.concatenate((casual_single,data_e[-1]),axis=1)
            casual_merge.append(casual_single)
        casual_merge_all.append(casual_merge)
    
    #get lstm models
    lstm_models=get_lstm_model(data_type,casual_type,casual_algorithm,data_nums[var_num],var_num)

    #dispose the encodered data by lstm models
    datas_by_lstm=[]
    with tf.device('/CPU:0'):
        for casual_merge in casual_merge_all:
            data_by_lstm=[]
            for i in range(params.latent_dim):
                data_by_lstm.append(0)
            for i in range(len(lstm_models)):
                shape=casual_merge[i].shape
                val=lstm_models[i](casual_merge[i].reshape(1,shape[0],shape[1])).numpy()
                data_by_lstm[data_nums[var_num][i]]=val[0][-1][0]
            datas_by_lstm.append(data_by_lstm)

    #decoder data
    datas_decoder=[]
    for data_by_lstm in datas_by_lstm:
        data_decoder=encoder_models[var_num].decoder(tf.reshape(data_by_lstm,[1,64])).numpy().reshape(40,55)
        datas_decoder.append(data_decoder)
    
    return datas_decoder

def predict_mutil(start,predict_lens):
    #cmap
    cmap=cmaps.GMT_panoply
    #序列长度
    sequence_len=6
    #load month weight
    m_w=get_month_weight(1740)
    #从npz文件中加载numpy数据
    data_type='reanalysis'
    data_all=[]
    for var in params.variables:
        data_all.append(np.load(f"{params.final_data_dir}/{data_type}/{var}-final.npz")[var])
    #data for save
    save_data={}
    mse,psnr,ssim=[],[],[]
    #initial data
    orgin_data=[]
    
    for i in range(6):
        orgin_data.append(data_all[i][start:start+sequence_len])
    orgin_data.append(m_w[start:start+sequence_len])
    #recursion
    for i in range(predict_lens):
        orgin_datas=[]
        orgin_datas.append(orgin_data)
        predict_res=[]
        bar = PixelBar(r'Generating', max=6, suffix='%(percent)d%%')#进度条显示
        for j in range(6):
            single_res=predict_single(orgin_datas,j)[0]
            predict_res.append(single_res)
            bar.next()
        bar.finish()
        #save
        save_data[str(i)+'m']=predict_res
        #performance
        data1,data2=data_all[0][start+sequence_len+i],predict_res[0]
        MSE=mean_squared_error(data1, data2)
        mse.append(MSE)
        PSNR = peak_signal_noise_ratio(data1, data2,data_range=1.0)
        psnr.append(PSNR)
        SSIM = structural_similarity(data1, data2, multichannel=False,data_range=1.0)
        ssim.append(SSIM)
        print('MSE: ', MSE)
        print('PSNR: ', PSNR)
        print('SSIM: ', SSIM)
        #draw pictures
        plot_heatmap(data1,llcrnrlon=160,urcrnrlon=270,llcrnrlat=-40,
            urcrnrlat=40,xgrid=55, ygrid=40,xbar_d=10,ybar_d=10,
            save=True,title=f'{str(1850+int((start+sequence_len+i)/12))}year {(start+sequence_len+i)%12+1}month orginal',
            file_name=f'./forecast/data_storage/{str(start)}-orgin-{str(i+1)}.png',
            cmap=cmap,level=np.arange(0,1.05,1.05/100)) 
        plot_heatmap(data2,llcrnrlon=160,urcrnrlon=270,llcrnrlat=-40,
            urcrnrlat=40,xgrid=55, ygrid=40,xbar_d=10,ybar_d=10,
            save=True,title=f'{str(1850+int((start+sequence_len+i)/12))}year {(start+sequence_len+i)%12+1}month predicted',
            file_name=f'./forecast/data_storage/{str(start)}-predict-{str(i+1)}.png',
            cmap=cmap,level=np.arange(0,1.05,1.05/100)) 
        #update orginal data
        for j in range(6):
            orgin_data[j]=np.concatenate((orgin_data[j],predict_res[j].reshape(1,40,55)),axis=0)
            orgin_data[j]=orgin_data[j][1:,:,:]
        orgin_data[6]=m_w[start+i+1:start+sequence_len+i+1]
    save_data['mse']=mse
    save_data['psnr']=psnr
    save_data['ssim']=ssim

    np.savez(f'./forecast/data_storage/{str(start)}-predict-{str(predict_lens)}-res.npz',**save_data)
    return save_data

if __name__=='__main__':
    #
    year=2004
    start=(year-1870)*12-6
    predict_lens=18

    res=predict_mutil(start,predict_lens)

    print("end")

                
