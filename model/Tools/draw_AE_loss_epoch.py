import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append("")

from model.params import *
import seaborn as sns

#load the history
var='sst'
history=np.load(f'./model/AutoEncoder/model_storage/reanalysis/{var}-AE-history.npz')
train_loss=history['loss']
val_loss=history['val_loss']

x=np.arange(0,200,1)

#draw loss-epoch figure
fig = plt.figure(figsize=(18, 12))

plt.plot(x,train_loss,color = sns.xkcd_rgb["denim blue"],label="train_loss")#s-:方形
plt.plot(x,val_loss,color = sns.xkcd_rgb["pale red"],label="val_loss")#o-:圆形

plt.xlabel("epoch")#横坐标名字
plt.ylabel("loss")#纵坐标名字
plt.legend(loc = "best")#图例
plt.ylim(10)
plt.xticks(np.arange(0,201,10))

plt.show()

print("end")