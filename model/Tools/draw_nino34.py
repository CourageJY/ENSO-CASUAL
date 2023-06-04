import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
sys.path.append("")

from model.params import *

#get min max scaler
sst_orgin = np.load(f"{params.reanalysis_npz_dir}/sst-resolve.npz")['sst']#(1740,40,55)
scaler = MinMaxScaler()
scaler.fit(np.reshape(sst_orgin, (-1, 10*50)))

#get mean data of sst
data_type='reanalysis'
sst = np.load(f"{params.final_data_dir}/{data_type}/sst-final.npz")['sst']
sst=np.reshape(scaler.inverse_transform(np.reshape(sst,(-1,10*50))),(-1,10,50))
len_=len(sst)
years=30

sst_mean_month=[]#(12,40,55)
for i in range(12):
    sst_mean_month.append(0)
    for j in range(len_-30*12,len_):
        if(j%12==i):
            sst_mean_month[i]+=sst[j]
    sst_mean_month[i]=sst_mean_month[i]/years

#load the predicted data
#start=1345
year=1982
start=(year-1870)*12#-params.sequence_length
sequence_len=params.sequence_length
predict_len=15
#file_name='1345-predict-15-res.npz'
file_name='1982-predict-15-res.npz'
sst_predict=[]
h=np.load(f'./forecast/data_storage/{file_name}')
#h1=np.load(f'./forecast/data_storage/ago/{file_name}')
for i in range(predict_len):
    p_=np.load(f'./forecast/data_storage/{file_name}')[str(i)+'m'][0]#(40,55)
    p_=np.reshape(scaler.inverse_transform(np.reshape(p_,(-1,10*50))),(10,50))
    sst_predict.append(p_)

#get anormal data of the predicted
sst_predict_anormal=[]#(predict_len,40,55)
for i in range(predict_len):
    m=(start+sequence_len+i)%12
    anormal_=sst_predict[i]-sst_mean_month[m]
    sst_predict_anormal.append(anormal_)

#get anormal data of the real
sst_real_anormal=[]
for i in range(predict_len):
    m=(start+sequence_len+i)%12
    anormal_=sst[start+sequence_len+i]-sst_mean_month[m]
    sst_real_anormal.append(anormal_)

#get the mean anormal of the nino3.4 region
sst_predict_anormal_nino34=[]
sst_real_anormal_nino34=[]
for i in range(predict_len):
    #predict_anormal_mean=(sst_predict_anormal[i][18:22,15:39].sum()+sst_predict_anormal[i][17,15:40].sum()/2+sst_predict_anormal[i][18:22,40].sum()/2)/(125+31/2)
    predict_anormal_mean=sst_predict_anormal[i].sum()/(10*50)
    sst_predict_anormal_nino34.append(predict_anormal_mean)
    #real_anormal_mean=(sst_real_anormal[i][18:22,15:39].sum()+sst_real_anormal[i][17,15:40].sum()/2+sst_real_anormal[i][18:22,40].sum()/2)/(125+31/2)
    real_anormal_mean=sst_real_anormal[i].sum()/(10*50)
    sst_real_anormal_nino34.append(real_anormal_mean)

#draw the picture
fig = plt.figure(figsize=(12, 6))

x=range(1,predict_len+1)

plt.plot(x,sst_predict_anormal_nino34,'s-',color = 'r',label="nino3.4 predict")#s-:方形
plt.plot(x,sst_real_anormal_nino34,'o-',color = 'g',label="nino3.4 real")#o-:圆形
#plt.plot(x,[0.93,1.39,2.00,2.07,2.33,2.43,2.22,1.69,1.12,1.12,0.62,-0.11,-0.13,-0.50,-1.03],'o-',color = 'y',label="nino3.4 others")#o-:圆形
#plt.plot(x,[0.70,0.82,1.16,1.41,1.41,0.98,0.64,0.48,-0.03,-0.52,-0.19,0.14,0.05,0.15,0.46],'o-',color = 'y',label="nino3.4 others")#o-:圆形
#plt.plot(x,[-0.50,-1.03,-1.13,-0.95,-0.80,-0.53,-0.48,-0.60,-0.64,-0.90,-0.40,-0.40,-0.35,-0.83,-1.18],'o-',color = 'y',label="nino3.4 others")#o-:圆形
#plt.plot(x,[0.93,1.39,2.00,2.07,2.33,2.43],'o-',color = 'y',label="nino3.4 others")
plt.xlabel("month")#横坐标名字
plt.xticks(range(1,predict_len+1))#横坐标刻度
plt.ylabel("anormal temperature(C)")#纵坐标名字
plt.legend(loc = "best")#图例

plt.axhline(0, color='black', lw=0.5)
plt.axhline(0.5, color='black', linewidth=0.5, linestyle='dotted')
plt.axhline(-0.5, color='black', linewidth=0.5, linestyle='dotted')
plt.title('1982 Niño 3.4 Index')

plt.show()

print("end")