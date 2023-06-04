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


psnr=[23.2,20.1,22.4,22.0,20.2,17.1,20.3,19.8,17.6,15.7,18.3,17.9]
ssim=[0.84,0.76,0.82,0.81,0.76,0.63,0.75,0.74,0.68,0.57,0.71,0.70]
ysr=['enso-casual','convl2d','convl3d','AE-lstm','enso-casual','convl2d','convl3d','AE-lstm','enso-casual','convl2d','convl3d','AE-lstm']
#ysr2=['0-5 month','6-10 month','11-15 month']
y3=[0,0,0,0,0,0,0,0,0,0,0,0]
#draw picture for psnr and ssim among defferent casuality
fig, ax1 = plt.subplots(figsize=(14, 6))
ax2=ax1.twinx()

x = np.arange(1,13) #group number
x = x - 0.2
b1= ax1.bar(x, ssim,width=0.4,color = sns.xkcd_rgb["denim blue"],label='ssim')
b2= ax2.bar(x+0.4, psnr,width=0.4,color = sns.xkcd_rgb["pale red"],label='psnr')
b3= ax1.bar(x+0.2, y3,width=0.4,color = sns.xkcd_rgb["pale red"],label=' ',tick_label=ysr)
#b4= ax1.bar([2,5,8], [0,0,0],width=0.4,color = sns.xkcd_rgb["pale red"],label=' ',tick_label=ysr2)

# 坐标轴标签设置
#ax1.set_title(f'{year} year: difference for psnr and ssim')
#ax1.set_xlabel('month')
#ax2.set_xlabel('month2')
ax2.set_ylabel('psnr')
ax1.set_ylabel('ssim')
#ax2.set_xticks(['0-5 month','6-10 month','11-15 month'])
ax2.set_yticks(np.arange(12,31,1))
ax1.set_yticks(np.arange(0.5,1.01,0.05))
ax2.set_ylim(12)
ax1.set_ylim(0.5)
# 图例设置
plt.legend(handles = [b1,b2])
# 网格设置
plt.grid('off')

plt.show()

print("end")