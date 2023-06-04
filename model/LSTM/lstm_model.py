import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

import sys
sys.path.append("")
from model.params import *

#采用三层lstm的结构
class LSTM_model(Model):
 def __init__(self):
    super(LSTM_model, self).__init__() 
    self.lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, 32]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape [batch, time, 32] => [batch, time, 24]
    #tf.keras.layers.LSTM(24, return_sequences=True),
    # Shape [batch, time, 32] => [batch, time, 16]
    #tf.keras.layers.LSTM(16, return_sequences=True),
    # Shape => [batch, time, 1]
    tf.keras.layers.Dense(units=1)
    ])

 def call(self, x):
    return self.lstm_model(x)#(batch,time,1)
 
 def compile_and_fit(self, train_data,train_label,test_data,test_label, patience=8):#过拟合则提前终止
   early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

   self.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

   history = self.fit(train_data,train_label, 
                      epochs=params.num_epochs,
                      validation_data=(test_data,test_label),
                      batch_size=params.batch_size,
                      shuffle=True,
                      callbacks=[early_stopping])
   return history