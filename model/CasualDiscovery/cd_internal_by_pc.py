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


sst_e=np.load(f'{params.encoder_save_dir}/sst-encoder.npz')['sst']

names,nums=[],[]
for j in range(sst_e.shape[1]):
    if(sst_e[0:,j:j+1].sum()>0):#not zero
        names.append('sst'+str(j))
        nums.append(j)

sst_u=sst_e[0:,nums[0]:nums[0]+1]
for i in range(1,len(nums)):
    sst_u=np.concatenate((sst_u,sst_e[0:,nums[i]:nums[i]+1]),axis=1)

print(sst_u.shape)

tic = time.time()

cg = pc(sst_u, 0.05, fisherz,node_names=names)

toc = time.time()

print(f'Time taken: {toc-tic:.2f}s\n')

# visualization using pydot
cg.draw_pydot_graph()

# save the graph
pyd = GraphUtils.to_pydot(cg.G)
pyd.write_png('./model/CasualDiscovery/graph_storage/sst_internal.png')

# save the graph as npz
graph=cg.G.graph
sst_interal={}
pa=[]
for i in range(len(nums)):
    pa.append([])
for i in range(len(nums)):
    for j in range(len(graph[i])):  
        if(graph[i][j]==-1):#1 or -1
            pa[j].append(names[i])
for i in range(len(nums)):
    sst_interal[str(names[i])]=pa[i]
print(sst_interal)

np.savez('./model/CasualDiscovery/graph_storage/sst-internal.npz',**sst_interal)

print("end")



# visualization using networkx
# cg.to_nx_graph()
# cg.draw_nx_graph(skel=False)