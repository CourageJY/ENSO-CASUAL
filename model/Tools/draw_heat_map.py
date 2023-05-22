#display the map for sst/vwind/wwind...
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.basemap import Basemap
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

sys.path.append("")
from model.params import *

import cmaps

def plot_heatmap(data, llcrnrlon=150, urcrnrlon=230, llcrnrlat=-40, urcrnrlat=40, xgrid=0, ygrid=0,xbar_d=0,ybar_d=0
                 ,show=False,save=False,title=None,file_name=None,cmap=None,level=[]):
    plt.figure()
    m = Basemap(projection="cyl", llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, resolution='i')

    m.fillcontinents(color='grey')
    m.drawcoastlines(linewidth=0.2)
    m.drawcountries()

    m.drawparallels(np.arange(llcrnrlat, urcrnrlat+1, ybar_d), labels=[True, False, True, False], linewidth=0.2, color='k', fontsize=6)
    m.drawmeridians(np.arange(llcrnrlon, urcrnrlon+1, xbar_d), labels=[True, False, False, True], linewidth=0.2, color='k', fontsize=6)
    # m.shadedrelief()

    xx = np.linspace(llcrnrlon, urcrnrlon, xgrid)
    yy = np.linspace(llcrnrlat, urcrnrlat, ygrid)
    Lon, Lat = np.meshgrid(xx, yy)
    x, y = m(Lon, Lat)
    if len(level)!=0:
        cs = m.contourf(x, y, data, level, cmap=cmap)
    else:
        cs = m.contourf(x, y, data, cmap=cmap)
    cbar = m.colorbar(cs)
    cbar.outline.set_linewidth(1)
    if save:
        plt.title(title, fontsize=6, pad=20)
        plt.savefig(file_name)
        plt.close(0)
    if show:
        plt.show()

if __name__=='__main__':
    # x=np.arange(0,110,10)
    # y=np.arange(0,80,10)
    # xx, yy = np.meshgrid(x, y)
    # data=np.array(xx)
    # plot_heatmap(data,llcrnrlon=160,urcrnrlon=270,llcrnrlat=-40,
    #               urcrnrlat=40,xgrid=11, ygrid=8,xbar_d=10,ybar_d=10,show=True)
    cmap=cmaps.GMT_panoply
    sst=np.load("./data/final/reanalysis/sst-final.npz")['sst']
    #sst_=np.load(f"{params.remote_sensing_npz_dir}/sst-no-nan-final.npz")['sst']

    # #figure1
    plot_heatmap(sst[0],llcrnrlon=160,urcrnrlon=270,llcrnrlat=-40,
                 urcrnrlat=40,xgrid=55, ygrid=40,xbar_d=10,ybar_d=10,show=True,cmap=cmap,level=np.arange(0,1.06,1.05/100))
    
    plot_heatmap(sst[3],llcrnrlon=160,urcrnrlon=270,llcrnrlat=-40,
                urcrnrlat=40,xgrid=55, ygrid=40,xbar_d=10,ybar_d=10,show=True,cmap=cmap,level=np.arange(0,1.06,1.05/100))
    
    plot_heatmap(sst[6],llcrnrlon=160,urcrnrlon=270,llcrnrlat=-40,
            urcrnrlat=40,xgrid=55, ygrid=40,xbar_d=10,ybar_d=10,show=True,cmap=cmap,level=np.arange(0,1.06,1.05/100))
    plot_heatmap(sst[9],llcrnrlon=160,urcrnrlon=270,llcrnrlat=-40,
            urcrnrlat=40,xgrid=55, ygrid=40,xbar_d=10,ybar_d=10,show=True,cmap=cmap,level=np.arange(0,1.06,1.05/100))
    # #figure2
    # plot_heatmap(sst_[0],llcrnrlon=160,urcrnrlon=270,llcrnrlat=-40,
    #              urcrnrlat=40,xgrid=55, ygrid=40,xbar_d=10,ybar_d=10,show=True,cmap=cmap,level=np.arange(0,1.06,1.05/100)) 

    # #figure3
    # sst=(sst-sst.min())/(sst.max()-sst.min())#数据归一化

    # plot_heatmap(sst[0],llcrnrlon=160,urcrnrlon=270,llcrnrlat=-40,
    #              urcrnrlat=40,xgrid=440, ygrid=320,xbar_d=10,ybar_d=10,show=True,cmap=cmap,level=np.arange(0,1.06,1.05/11))

    print("end")