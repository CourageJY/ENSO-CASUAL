import os
import json
import random
import numpy as np
import tensorflow as tf
import netCDF4 as nc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from progress.bar import PixelBar
# from netCDF4 import Dataset
# import numpy as np
# import sys
# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.basemap import Basemap
# from pandas import DataFrame

# #数据读入
# nc=Dataset('RSS_AMSR2_ocean_L3_3day_2023-02-28_v08.2.nc')

# print(nc.variables.keys())

# #取出各variable的数据看看,数据格式为numpy数组
# for var in nc.variables.keys():
#     data=nc.variables[var][:].data
#     print(var,data.shape)

# print(nc.variables["lon"][:].data)

print("end")