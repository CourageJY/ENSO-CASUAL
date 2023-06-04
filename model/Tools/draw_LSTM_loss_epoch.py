import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append("")

from model.params import *
import seaborn as sns

#load the history
var='sst'
history=np.load(f'./model/LSTM/history/{var}-1-3.npz')
train_loss=history['loss']
val_loss=history['val_loss']
train_metric=history['mean_absolute_error']
val_metric=history['val_mean_absolute_error']

x=np.arange(0,len(train_loss),1)

#draw loss-epoch figure
fig = plt.figure(figsize=(8, 6))

plt.plot(x,train_metric,color = sns.xkcd_rgb["denim blue"],label="mean_absolute_error")#s-:方形
plt.plot(x,val_metric,color = sns.xkcd_rgb["pale red"],label="val_mean_absolute_error")#o-:圆形

plt.xlabel("epoch")#横坐标名字
plt.ylabel("loss")#纵坐标名字
plt.legend(loc = "best")#图例
#plt.ylim(0.05)
plt.xticks(np.arange(0,len(train_loss),5))
plt.yticks(np.arange(0,0.25,0.03))
plt.ylim(0.05)

plt.title(f'{var} loss-epoch', fontsize=6, pad=20)

plt.savefig(f'./forecast/analysis/lstm_{var}_loss_epoch.png')

plt.show()

plt.close(0)

print("end")