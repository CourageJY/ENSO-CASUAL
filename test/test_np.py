import numpy as np
 
# A = np.array([[1,-1,2],
#               [3, 2,0]])

# v = np.transpose(np.array([[2,1,3]]))

# print(A[1,2])

# col = A[:,1:2]

# w = np.dot(A,v)

# print(col)

# print(w)

# (x,y),(m,n)=((np.array([1,3,3]),np.array([0,1,2])),(np.array([1,2,3]),np.array([2,3,4])))

# print(x,n)

# print(np.__version__)

# A = np.array([[[1,-1,2],
#               [3, 2,0],
#               [4, 5,6]],
#               [[1,-1,2],
#               [3, 2,0],
#               [4, 5,6]]])
# #B=A[0:2].transpose()[0:2].transpose()
# #C=A[0:1].reshape(3,3)
# B1=A[0:1].reshape(3,3)[0:2].transpose()[0:2].transpose().reshape(1,2,2)
# B2=A[1:2].reshape(3,3)[0:2].transpose()[0:2].transpose().reshape(1,2,2)
# print(np.concatenate((B1,B2),axis=0).mean().reshape(1,1,1).shape)


# for i in range(10):
#     print(i)

sst_, uwind_=[],[]

print(type(sst_))