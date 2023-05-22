import matplotlib.pyplot as plt

#折线图
x = [5,7,11,17,19,25]#点的横坐标
k1 = [0.8222,0.918,0.9344,0.9262,0.9371,0.9353]#线1的纵坐标
k2 = [0.8988,0.9334,0.9435,0.9407,0.9453,0.9453]#线2的纵坐标
plt.plot(x,k1,'s-',color = 'r',label="ATT-RLSTM")#s-:方形
plt.plot(x,k2,'o-',color = 'g',label="CNN-RLSTM")#o-:圆形
plt.xlabel("region length")#横坐标名字
plt.ylabel("accuracy")#纵坐标名字
plt.legend(loc = "best")#图例
plt.show()