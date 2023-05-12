import numpy as np
import sys
from sklearn.model_selection import train_test_split

sys.path.append("")
from model.AutoEncoder.auto_encoder import *
from model.params import *
from model.Tools.draw_heat_map import *
import cmaps

sst = np.load(f"{params.final_data_dir}/sst-final.npz")['sst']

cmap=cmaps.GMT_panoply
#figure1
plot_heatmap(sst[0],llcrnrlon=160,urcrnrlon=270,llcrnrlat=-40,
            urcrnrlat=40,xgrid=55, ygrid=40,xbar_d=10,ybar_d=10,
            show=True,cmap=cmap,level=np.arange(0,1.06,1.05/100)) 

#load model
model=Autoencoder(params.latent_dim)
model.load_weights(f'{params.encoder_save_dir}/sst-model')

sst_encoder=np.load(f'{params.encoder_save_dir}/sst-encoder.npz')['sst']
#print(sst_encoder[0].reshape(64).shape)
t = tf.convert_to_tensor(sst_encoder, tf.float32)

print(tf.reshape(t[0],[1,64]).shape)
print(t.shape)
res=model.decoder((tf.reshape(t[0],[1,64])))
#res=model.encoder(sst)

#figure2
plot_heatmap(res[0],llcrnrlon=160,urcrnrlon=270,llcrnrlat=-40,
            urcrnrlat=40,xgrid=55, ygrid=40,xbar_d=10,ybar_d=10,
            show=True,cmap=cmap,level=np.arange(0,1.06,1.05/100)) 

print("end")