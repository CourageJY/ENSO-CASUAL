import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

tf.keras.models
class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),# (batch_size, 32, 44) => (batch_size, 32*44)
      layers.Dense(latent_dim, activation='relu'),# (batch_size, 32*44) => (batch_size, 64)
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(32*44, activation='sigmoid'),# (batch_size, 64) => (batch_size, 32, 44)
      layers.Reshape((32, 44))# (batch_size, 32*44) => (batch_size, 32, 44)
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  