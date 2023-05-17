import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle as pkl
import time

from causalai.models.time_series.pc import PCSingle, PC
from causalai.models.common.CI_tests.partial_correlation import PartialCorrelation
from causalai.models.common.CI_tests.kci import KCI
from causalai.data.data_generator import DataGenerator, GenerateRandomTimeseriesSEM
from causalai.models.common.CI_tests.discrete_ci_tests import DiscreteCI_tests


# also importing data object, data transform object, and prior knowledge object, and the graph plotting function
from causalai.data.time_series import TimeSeriesData
from causalai.data.transforms.time_series import StandardizeTransform
from causalai.models.common.prior_knowledge import PriorKnowledge
from causalai.misc.misc import plot_graph, get_precision_recall

from progress.bar import PixelBar

import sys
sys.path.append("")
from model.params import *

#load the encodered data
data_e=[]
data_type='reanalysis'
for var in params.variables:
    data_e.append(np.load(f'{params.encoder_save_dir}/{data_type}/{var}-encoder.npz')[var])

# sst_e=np.load(f'{params.encoder_save_dir}/sst-encoder.npz')['sst']
# uwind_e=np.load(f'{params.encoder_save_dir}/uwind-encoder.npz')['uwind']
# vwind_e=np.load(f'{params.encoder_save_dir}/vwind-encoder.npz')['vwind']
# rain_e=np.load(f'{params.encoder_save_dir}/rain-encoder.npz')['rain']

#--------get the num of no_zero in data----------
data_names,data_nums=[],[]
for i in range(len(data_e)):
    names,nums=[],[]
    for j in range(data_e[i].shape[1]):
        if(data_e[i][0:,j:j+1].sum()>0):#not zero
            names.append(params.variables[i]+str(j))
            nums.append(j)
    data_names.append(names)
    data_nums.append(nums)

# sst_names,sst_nums=[],[]
# for i in range(sst_e.shape[1]):
#     if(sst_e[0:,i:i+1].sum()>0):
#         sst_names.append("sst"+str(i))
#         sst_nums.append(i)

# uwind_names,uwind_nums=[],[]
# for i in range(uwind_e.shape[1]):
#     if(uwind_e[0:,i:i+1].sum()>0):
#        uwind_names.append("uwind"+str(i)) 
#        uwind_nums.append(i)

#------pc casual discovey for single sst latent features----------
for i in range(1,len(data_e)): # not include 0, 0 is sst
    print("---------------------------")
    print("-------Step{}--------------".format(i))
    print("---------------------------")
    res={}
    bar = PixelBar(r'Generating', max=len(data_nums[0]), suffix='%(percent)d%%')#进度条显示
    for j in range(len(data_nums[0])):
        var_names=[]
        var_names.append(data_names[0][j])
        var_names=var_names+data_names[i]

        #print(var_names)

        data=data_e[0][0:,data_nums[0][j]:data_nums[0][j]+1]
        for k in data_nums[i]:
            data=np.concatenate((data,data_e[i][0:,k:k+1]),axis=1)

        #print(data.shape)

        StandardizeTransform_ = StandardizeTransform()
        StandardizeTransform_.fit(data)

        data_trans = StandardizeTransform_.transform(data)

        data_obj=TimeSeriesData(data_trans, var_names=var_names)

        #--------PC Discovery-----------
        prior_knowledge = None 

        target_var = var_names[0]
        max_lag = 5
        pvalue_thres = 0.1
        #print(f'Target Variable: {target_var}, using max_lag {max_lag}, pvalue_thres {pvalue_thres}')

        CI_test = PartialCorrelation() # use KCI() if the causal relationship is expected to be non-linear
        #CI_test = KCI()
        pc_single = PCSingle(
            data=data_obj,
            prior_knowledge=prior_knowledge,
            CI_test=CI_test,
            use_multiprocessing=False,
            )

        tic = time.time()
        result = pc_single.run(target_var=target_var, pvalue_thres=pvalue_thres, 
                                max_lag=max_lag, max_condition_set_size=4)

        toc = time.time()
        #print(f'Time taken: {toc-tic:.2f}s\n')

        #print(f'Predicted parents:')
        parents = result['parents']
        #print(parents)

        res[var_names[0]]=parents

        bar.next()

    bar.finish()

    #save
    np.savez(f'./model/CasualDiscovery/graph_storage/{data_type}/{params.variables[i]}-sst-casual-pc.npz',**res)

    #print(res)

print("end")
