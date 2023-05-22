#bar chat
import numpy as np
import matplotlib.pyplot as plt

validation = [0.8892,0.761,0.7243,0.6954,0.6732,0.6321,0.6420,0.6532,0.6544,0.6631,0.6638,0.6636]
Test = [0.8966,0.7556,0.7316,0.6974,0.6832,0.6341,0.6432,0.6542,0.6574,0.6671,0.6698,0.6696]

x = np.arange(1,13) #group number
total_width, n = 0.4, 2
width = total_width / n
x = x - (total_width-width)
plt.bar(x, validation, color = "r",width=width,label='validation')
plt.bar(x + width, Test, color = "y",width=width,label='test')

plt.xlabel("sequence length")
plt.ylabel("mean_absolute_error")
plt.xticks(range(1,13),range(1,13))
plt.yticks(np.arange(0,1, 0.1))
plt.legend(loc = "best")

plt.show()
