import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle as pkl
import time

from causalai.models.time_series.pc import PCSingle, PC
from causalai.models.time_series.granger import GrangerSingle, Granger
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
    data_e.append(np.load(f'{params.encoder_save_dir}/{data_type}/{var}-min-max-encoder.npz')[var])

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

#------ganger casual discovey for single varibles latent features----------
for k in range(len(data_e)):
    #if k<3:continue
    for i in range(len(data_e)): # not include 0, 0 is sst
        print("---------------------------")
        print("-------Step{}--------------".format(i))
        print("---------------------------")
        if(i==k):continue

        res={}
        bar = PixelBar(r'Generating', max=len(data_nums[k]), suffix='%(percent)d%%')#进度条显示
        for j in range(len(data_nums[k])):
            var_names=[]
            var_names.append(data_names[k][j])
            var_names=var_names+data_names[i]

            #print(var_names)

            data=data_e[k][0:,data_nums[k][j]:data_nums[k][j]+1]
            for u in data_nums[i]:
                data=np.concatenate((data,data_e[i][0:,u:u+1]),axis=1)

            #print(data.shape)

            StandardizeTransform_ = StandardizeTransform()
            StandardizeTransform_.fit(data)

            data_trans = StandardizeTransform_.transform(data)

            data_obj=TimeSeriesData(data_trans, var_names=var_names)

            #--------Ganger Discovery-----------
            prior_knowledge = None 

            target_var = var_names[0]
            max_lag = params.external_casual_times
            pvalue_thres = 0.005
            #print(f'Target Variable: {target_var}, using max_lag {max_lag}, pvalue_thres {pvalue_thres}')

            #CI_test = PartialCorrelation() # use KCI() if the causal relationship is expected to be non-linear
            #CI_test = KCI()
            granger_single = GrangerSingle(
                data=data_obj,
                prior_knowledge=prior_knowledge,
                max_iter=20000, # number of optimization iterations for model fitting (default value is 1000)
                use_multiprocessing=False
                )
            tic = time.time()
            result = granger_single.run(target_var=target_var, 
                                        pvalue_thres=pvalue_thres, 
                                        max_lag=max_lag)

            toc = time.time()
            #print(f'Time taken: {toc-tic:.2f}s\n')

            #print(f'Predicted parents:')
            parents = result['parents']
            #print(parents)

            res[var_names[0]]=parents

            bar.next()

        bar.finish()

        #save
        np.savez(f'./model/CasualDiscovery/graph_storage/{data_type}/ganger-external/{params.variables[i]}-{params.variables[k]}-casual-ganger.npz',**res)

        #print(res)

print("end")
