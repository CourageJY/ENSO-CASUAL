import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
sys.path.append("")

from model.params import *
from model.AutoEncoder.auto_encoder import *

from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio

import seaborn as sns

#get min max scaler
sst_orgin = np.load(f"{params.reanalysis_npz_dir}/sst-resolve-shink.npz")['sst']#(1740,40,55)
scaler = MinMaxScaler()
scaler.fit(np.reshape(sst_orgin, (-1, 40*55)))

#get mean data of sst
data_type='reanalysis'
sst_unfit = np.load(f"{params.final_data_dir}/{data_type}/sst-final.npz")['sst']
sst=np.reshape(scaler.inverse_transform(np.reshape(sst_unfit,(-1,40*55))),(-1,40,55))
len_=len(sst)
years=30

sst_mean_month=[]#(12,40,55)
for i in range(12):
    sst_mean_month.append(0)
    for j in range(len_-30*12,len_):
        if(j%12==i):
            sst_mean_month[i]+=sst[j]
    sst_mean_month[i]=sst_mean_month[i]/years

#load the encodered data
sst_encoder=np.load(f'./model/AutoEncoder/model_storage/reanalysis/sst-encoder.npz')['sst']

#load the decoder model
data_type='reanalysis'
model=Autoencoder(params.latent_dim)
model.load_weights(f'{params.encoder_save_dir}/{data_type}/sst-model').expect_partial()

year=2004
start=(year-1870)*12
predict_len=36

#get the decodered data(predict)
t=tf.convert_to_tensor(sst_encoder, tf.float32)
t=tf.reshape(t,[-1,64])
sst_predict=model.decoder(t).numpy().reshape(-1,40,55)
sst_predict_fit=np.reshape(scaler.inverse_transform(np.reshape(sst_predict,(-1,40*55))),(-1,40,55))

#get anormal data of the predicted
sst_predict_anormal=[]#(predict_len,40,55)
for i in range(predict_len):
    m=(start+i)%12
    anormal_=sst_predict_fit[start+i]-sst_mean_month[m]
    sst_predict_anormal.append(anormal_)

#get anormal data of the real
sst_real_anormal=[]
for i in range(predict_len):
    m=(start+i)%12
    anormal_=sst[start+i]-sst_mean_month[m]
    sst_real_anormal.append(anormal_)

#get the mean anormal of the nino3.4 region
sst_predict_anormal_nino34=[]
sst_real_anormal_nino34=[]
for i in range(predict_len):
    predict_anormal_mean=(sst_predict_anormal[i][18:22,15:39].sum()+sst_predict_anormal[i][17,15:40].sum()/2+sst_predict_anormal[i][18:22,40].sum()/2)/(125+31/2)
    sst_predict_anormal_nino34.append(predict_anormal_mean)
    real_anormal_mean=(sst_real_anormal[i][18:22,15:39].sum()+sst_real_anormal[i][17,15:40].sum()/2+sst_real_anormal[i][18:22,40].sum()/2)/(125+31/2)
    sst_real_anormal_nino34.append(real_anormal_mean)

#draw nino3.4 index
fig = plt.figure(figsize=(12, 6))

x=range(1,predict_len+1)

plt.plot(x,sst_predict_anormal_nino34,'s-',color = 'r',label="nino3.4 predict")#s-:方形
plt.plot(x,sst_real_anormal_nino34,'o-',color = 'g',label="nino3.4 real")#o-:圆形
plt.xlabel("month")#横坐标名字
plt.xticks(range(1,predict_len+1))#横坐标刻度
plt.ylabel("anormal temperature(C)")#纵坐标名字
plt.legend(loc = "best")#图例

plt.axhline(0, color='black', lw=0.5)
plt.axhline(0.5, color='black', linewidth=0.5, linestyle='dotted')
plt.axhline(-0.5, color='black', linewidth=0.5, linestyle='dotted')
plt.title(f'{year} year Niño 3.4 Index')

plt.show()

#get psnr and ssim of difference
psnr=[]
ssim=[]
for i in range(len(sst_unfit)):
    PSNR = peak_signal_noise_ratio(sst_unfit[i], sst_predict[i],data_range=1.0)
    psnr.append(PSNR)
    SSIM = structural_similarity(sst_unfit[i], sst_predict[i], multichannel=False,data_range=1.0)
    ssim.append(SSIM)

#draw picture for psnr and ssim between no-fit data
fig, ax1 = plt.subplots(figsize=(24, 6))
ax2=ax1.twinx()

x = np.arange(1,predict_len+1) #group number
x = x - 0.2
b1= ax1.bar(x, ssim[start:start+predict_len],width=0.4,color = sns.xkcd_rgb["denim blue"],label='ssim')
b2= ax2.bar(x+0.4, psnr[start:start+predict_len],width=0.4,color = sns.xkcd_rgb["pale red"],label='psnr')

# 坐标轴标签设置
ax1.set_title(f'{year} year: difference for psnr and ssim')
ax1.set_xlabel('month')
ax2.set_ylabel('psnr')
ax1.set_ylabel('ssim')
ax2.set_xticks(np.arange(1,predict_len+1))
ax2.set_yticks(np.arange(20,31,1))
ax1.set_yticks(np.arange(0.5,1.01,0.05))
ax2.set_ylim(20)
ax1.set_ylim(0.5)
# 图例设置
plt.legend(handles = [b1,b2])
# 网格设置
plt.grid('off')

plt.show()

print("end")

