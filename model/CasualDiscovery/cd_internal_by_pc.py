from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils

import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz, d_separation
from causallearn.graph.SHD import SHD
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph

import sys
sys.path.append("")
from model.params import *
import time

#load the encodered data
data_e=[]
data_type='reanalysis'
for var in params.variables:
    data_e.append(np.load(f'{params.encoder_save_dir}/{data_type}/{var}-encoder.npz')[var])

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


# data_type='reanalysis'
# sst_e=np.load(f'{params.encoder_save_dir}/{data_type}/sst-encoder.npz')['sst']

# names,nums=[],[]
# for j in range(sst_e.shape[1]):
#     if(sst_e[0:,j:j+1].sum()>0):#not zero
#         names.append('sst'+str(j))
#         nums.append(j)

datas_u=[]
for i in range(len(data_e)):
    data_u=data_e[i][0:,data_nums[i][0]:data_nums[i][0]+1]
    for j in range(1,len(data_nums[i])):
        data_u=np.concatenate((data_u,data_e[i][0:,data_nums[i][j]:data_nums[i][j]+1]),axis=1)
    datas_u.append(data_u)
    print(data_u.shape)

for i in range(len(data_e)):
    tic = time.time()

    cg = pc(datas_u[i], 0.05, fisherz, node_names=data_names[i])

    toc = time.time()

    print(f'Time taken: {toc-tic:.2f}s\n')

    # visualization using pydot
    cg.draw_pydot_graph()

    # save the graph
    pyd = GraphUtils.to_pydot(cg.G)
    pyd.write_png(f'./model/CasualDiscovery/graph_storage/{data_type}/{params.variables[i]}_internal.png')

    # save the graph as npz
    graph=cg.G.graph
    data_interal={}
    pa=[]
    for j in range(len(data_nums[i])):
        pa.append([])
    for j in range(len(data_nums[i])):
        for k in range(len(graph[i])):  
            if(graph[j][k]==-1):#1 or -1
                pa[k].append(data_nums[i][j])
    for j in range(len(data_nums[i])):
        data_interal[str(data_names[i][j])]=pa[j]
    print(data_interal)

    np.savez(f'./model/CasualDiscovery/graph_storage/{data_type}/{params.variables[i]}-internal.npz',**data_interal)

print("end")





# visualization using networkx
# cg.to_nx_graph()
# cg.draw_nx_graph(skel=False)