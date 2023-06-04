from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio

import numpy as np
import tensorflow as tf

def loss_func(actual,pred):

    #return tf.math.reduce_mean(tf.square(actual - pred))

    # pred=pred.numpy()

    #PSNR = peak_signal_noise_ratio(actual, pred,data_range=1.0)
    PSNR = tf.image.psnr(actual, pred, max_val=1.0)

    #SSIM = structural_similarity(actual, pred, multichannel=False,data_range=1.0)
    SSIM = tf.image.ssim(actual, pred, max_val=1.0,filter_size=10)

    return 40-PSNR*SSIM
