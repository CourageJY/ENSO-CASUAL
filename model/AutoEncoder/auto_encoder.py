import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

import sys
sys.path.append("")

from model.params import *
from model.AutoEncoder.tools import *

tf.keras.models
class Autoencoder(tf.keras.models.Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),# (batch_size, 10, 50) => (batch_size, 10*50)
      layers.Dense(latent_dim, activation='relu'),# (batch_size, 10*50) => (batch_size, 32)
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(10*50, activation='sigmoid'),# (batch_size, 32) => (batch_size, 10*50)
      layers.Reshape((10, 50))# (batch_size, 10*50) => (batch_size, 10, 50)
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  
  def compile_and_fit(self, train_data,train_label,test_data,test_label, patience=8):#过拟合则提前终止
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    self.compile(loss=loss_func,
                 #loss=tf.keras.losses.MeanSquaredError(),
                 optimizer=tf.keras.optimizers.Adam(params.learning_rate),
                 metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = self.fit(train_data,train_label, 
                      epochs=params.num_epochs,
                      validation_data=(test_data,test_label),
                      batch_size=params.batch_size,
                      shuffle=True,
                      callbacks=[early_stopping])
    return history
  