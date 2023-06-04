#bar chat
import numpy as np
import matplotlib.pyplot as plt

valid = [0.8892,0.761,0.7243,0.6954,0.6732,0.6621,0.6650,0.6712,0.6754,0.6751,0.6778,0.6796]
train = [0.8766,0.7556,0.7116,0.6874,0.6632,0.6541,0.6572,0.6642,0.6674,0.6701,0.6723,0.6736]

x = np.arange(1,13) #group number
total_width, n = 0.8, 2
width = total_width / n

fig = plt.figure(figsize=(10, 6))

plt.bar(x+0.2, valid, color = "g",width=width,label='valid loss')
plt.bar(x-0.2 , train, color = "y",width=width,label='train loss')

plt.xlabel("sequence length")
plt.ylabel("mean_absolute_error(normalized data)")
plt.xticks(range(1,13),range(1,13))
plt.yticks(np.arange(0,1, 0.05))
plt.legend(loc = "best")
plt.ylim(0.4)
plt.grid('off')

plt.show()
