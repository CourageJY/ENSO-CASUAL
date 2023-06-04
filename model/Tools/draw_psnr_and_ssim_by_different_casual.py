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


psnr=[23.2,21.6,20.1,20.2,18.1,17.3,17.6,16.7,16.2]
ssim=[0.84,0.80,0.79,0.80,0.75,0.70,0.71,0.68,0.63]
ysr=['| all-casual',' just-internal',' no-casual |','| all-casual',' just-internal',' no-casual |','| all-casual',' just-internal',' no-casual |']
ysr2=['0-5 month','6-10 month','11-15 month']
y3=[0,0,0,0,0,0,0,0,0]
#draw picture for psnr and ssim among defferent casuality
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2=ax1.twinx()

x = np.arange(1,10) #group number
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