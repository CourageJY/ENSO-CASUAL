# print(0)

# print(1)

# print(2)

# print(3)

# print(4)

import tensorflow as tf

# from typing_extensions import Required

# # Helper libraries
# import numpy as np
# import matplotlib.pyplot as plt

# from tensorflow.keras import datasets, layers, models

# fashion_mnist = datasets.fashion_mnist

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# print(train_images.shape)

# print(tf.__version__)

# print(tf.config.list_physical_devices('GPU'))
import sys
sys.path.append("")
from model.AutoEncoder.auto_encoder import *

model=Autoencoder(64)

model.save_weights('./test/my_checkpoint')

newmodel=Autoencoder(64)
newmodel.save_weights('./test/my_checkpoint')

print("end")