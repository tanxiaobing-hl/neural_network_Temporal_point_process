import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

################################### for generating synthetic data
from scipy.stats import lognorm,gamma
from scipy.optimize import brentq

################################### for neural network modeling
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

######################################################
### RNN
######################################################
class RNN_PP():

    def __init__(self, size_rnn, time_step, log_mode=True):
        self.size_rnn = size_rnn
        self.time_step = time_step
        self.log_mode = log_mode

    def __call__(self, inputs):
        x = inputs

        if self.log_mode:
            x_nmlz = layers.Lambda(lambda x: (K.log(x) - self.mu_x) / self.sigma_x)(x)
        else:
            x_nmlz = layers.Lambda(lambda x: (x - self.mu_x) / self.sigma_x)(x)

        rnn = layers.SimpleRNN(self.size_rnn, input_shape=(self.time_step, 1), activation='tanh')(x_nmlz)

        return rnn

    def normalize_input(self, x):

        if self.log_mode:
            self.mu_x = np.log(x).mean()
            self.sigma_x = np.log(x).std()
        else:
            self.mu_x = x.mean()
            self.sigma_x = x.std()

        return self
