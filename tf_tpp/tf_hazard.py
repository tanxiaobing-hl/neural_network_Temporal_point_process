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
### constant hazard function
######################################################
class HAZARD_const():
    class Layer_LL(layers.Layer):

        def __init__(self, **kwargs):
            super(HAZARD_const.Layer_LL, self).__init__(**kwargs)

        def build(self, input_shape):
            self.build = True

        def call(self, inputs):
            x = inputs[0];
            p = inputs[1];
            log_l = p
            Int_l = K.exp(p) * x
            return [log_l, Int_l]

        def compute_output_shape(self, input_shape):
            return [input_shape[0], input_shape[0]]

    def __call__(self, inputs):
        x = inputs[0];
        rnn = inputs[1];
        p = layers.Dense(1)(rnn)
        [log_l, Int_l] = HAZARD_const.Layer_LL()([x, p])
        LL = layers.Subtract()([log_l, Int_l])
        return [LL, log_l, Int_l]


######################################################
### exponential hazard function
######################################################
class HAZARD_exp():
    class Layer_LL(layers.Layer):

        def __init__(self, **kwargs):
            super(HAZARD_exp.Layer_LL, self).__init__(**kwargs)

        def build(self, input_shape):
            self.a = self.add_weight(name='a', initializer=keras.initializers.Constant(value=1.0), shape=(),
                                     trainable=True)
            self.build = True

        def call(self, inputs):
            x = inputs[0];
            p = inputs[1];
            a = self.a;
            log_l = p - a * x
            Int_l = K.exp(p) * (1 - K.exp(-a * x)) / a
            return [log_l, Int_l]

        def compute_output_shape(self, input_shape):
            return [input_shape[0], input_shape[0]]

    def __call__(self, inputs):
        x = inputs[0];
        rnn = inputs[1];
        p = layers.Dense(1)(rnn)
        [log_l, Int_l] = HAZARD_exp.Layer_LL()([x, p])
        LL = layers.Subtract()([log_l, Int_l])
        return [LL, log_l, Int_l]


######################################################
### piecewise constant hazard function
######################################################
class HAZARD_pc():
    class Layer_LL(layers.Layer):

        def __init__(self, size_div, t_max, **kwargs):
            self.size_div = size_div
            self.t_max = t_max
            self.bin_l = K.constant(np.linspace(0, t_max, size_div + 1)[:-1].reshape(1, -1))
            self.bin_r = K.constant(np.linspace(0, t_max, size_div + 1)[1:].reshape(1, -1))
            self.width = K.constant(t_max / size_div)
            self.ones = K.constant(np.ones([size_div, 1]))
            super(HAZARD_pc.Layer_LL, self).__init__(**kwargs)

        def build(self, input_shape):
            self.build = True

        def call(self, inputs):
            x = inputs[0];
            p = inputs[1];

            r_le = K.cast(K.greater_equal(x, self.bin_l), dtype=K.floatx())
            r_er = K.cast(K.less(x, self.bin_r), dtype=K.floatx())
            r_e = r_er * r_le
            r_l = 1 - r_er

            log_l = K.log(K.dot(p * r_e, self.ones))
            Int_l = K.dot(p * r_l * self.width, self.ones) + K.dot(p * (x - self.bin_l) * r_e, self.ones)

            return [log_l, Int_l]

        def compute_output(self, input_shape):
            return [input_shape[0], input_shape[0], input_shape[0]]

    def __init__(self, size_div, t_max):
        self.size_div = size_div
        self.t_max = t_max

    def __call__(self, inputs):
        x = inputs[0];
        rnn = inputs[1];
        p = layers.Dense(self.size_div, activation='softplus')(rnn)
        [log_l, Int_l] = HAZARD_pc.Layer_LL(self.size_div, self.t_max)([x, p])
        LL = layers.Subtract()([log_l, Int_l])
        return [LL, log_l, Int_l]


######################################################
### neural network based hazard function
######################################################
class HAZARD_NN():

    def __init__(self, size_layer, size_nn, log_mode=True):
        self.size_layer = size_layer
        self.size_nn = size_nn
        self.log_mode = log_mode

    def __call__(self, inputs):
        x = inputs[0];
        rnn = inputs[1];

        if self.log_mode:
            x_nmlz = layers.Lambda(lambda x: (K.log(x) - self.mu_x) / self.sigma_x)(x)
        else:
            x_nmlz = layers.Lambda(lambda x: (x - self.mu_x) / self.sigma_x)(x)

        def abs_glorot_uniform(shape, dtype=None, partition_info=None):
            return K.abs(keras.initializers.glorot_uniform(seed=None)(shape, dtype=dtype))

        hidden_x = layers.Dense(self.size_nn, kernel_initializer=abs_glorot_uniform,
                                kernel_constraint=keras.constraints.NonNeg(), use_bias=False)(x_nmlz)
        hidden_p = layers.Dense(self.size_nn)(rnn)
        hidden = layers.Add()([hidden_x, hidden_p])
        hidden = layers.Activation('tanh')(hidden)

        for i in range(self.size_layer - 1):
            hidden = layers.Dense(self.size_nn, activation='tanh', kernel_initializer=abs_glorot_uniform,
                                  kernel_constraint=keras.constraints.NonNeg())(hidden)

        Int_l = layers.Dense(1, activation='softplus', kernel_initializer=abs_glorot_uniform,
                             kernel_constraint=keras.constraints.NonNeg())(hidden)
        log_l = layers.Lambda(lambda inputs: K.log(1e-10 + K.gradients(inputs[0], inputs[1])[0]))([Int_l, x])
        LL = layers.Subtract()([log_l, Int_l])

        return [LL, log_l, Int_l]

    def normalize_input(self, x):

        if self.log_mode:
            self.mu_x = np.log(x).mean()
            self.sigma_x = np.log(x).std()
        else:
            self.mu_x = x.mean()
            self.sigma_x = x.std()

        return self
