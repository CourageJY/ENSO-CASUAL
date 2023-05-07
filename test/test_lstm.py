import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# model = keras.Sequential()
# # Add an Embedding layer expecting input vocab of size 1000, and
# # output embedding dimension of size 64.
# model.add(layers.Embedding(input_dim=1000, output_dim=64))

# # Add a LSTM layer with 128 internal units.
# model.add(layers.LSTM(128))

# # Add a Dense layer with 10 units.
# model.add(layers.Dense(10))

# model.summary()

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# tf.config.experimental.set_memory_growth(physical_devices[1], enable=True)
with tf.device('/CPU:0'):
 paragraph1 = np.random.random((20, 10, 50)).astype(np.float32)
 paragraph2 = np.random.random((20, 10, 50)).astype(np.float32)
 paragraph3 = np.random.random((20, 10, 50)).astype(np.float32)

 lstm_layer = tf.keras.layers.LSTM(64, stateful=True)
 output = lstm_layer(paragraph1)
 output = lstm_layer(paragraph2)
 output = lstm_layer(paragraph3)

 print(np.shape(output))

# reset_states() will reset the cached state to the original initial_state.
# If no initial_state was provided, zero-states will be used by default.
 #lstm_layer.reset_states()

# import tensorflow as tf
# print("Num GPUs Available: ", tf.config.list_physical_devices('CPU'))
