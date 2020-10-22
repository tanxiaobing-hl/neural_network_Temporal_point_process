import numpy as np

################################### for neural network modeling
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model


######################################################################################
### a class for a recurrent neural network based model for temporal point processes
######################################################################################
from tf_tpp.tf_hazard import HAZARD_const, HAZARD_exp, HAZARD_pc, HAZARD_NN
from tf_tpp.tf_rnn import RNN_PP


class CustomEarlyStopping(keras.callbacks.Callback):

    def __init__(self):
        super(CustomEarlyStopping, self).__init__()
        self.best_val_loss = 100000
        self.history_val_loss = []
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):

        val_loss = logs['val_loss']
        self.history_val_loss = np.append(self.history_val_loss, val_loss)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_weights = self.model.get_weights()

        if self.best_val_loss + 0.05 < val_loss:
            self.model.stop_training = True

        if (epoch + 1) % 5 == 0:

            # print('epoch: %d, current_val_loss: %f, min_val_loss: %f' % (epoch+1,val_loss,self.best_val_loss) )

            if (epoch + 1) >= 15:
                if self.best_val_loss > self.history_val_loss[:-5].min() - 0.001:
                    self.model.stop_training = True

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)
        # print('set optimal weights')


class NPP():

    def __init__(self, time_step, size_rnn, type_hazard, size_layer=2, size_nn=64, size_div=128, log_mode=True):
        self.time_step = time_step
        self.size_rnn = size_rnn
        self.type_hazard = type_hazard
        self.size_layer = size_layer
        self.size_nn = size_nn
        self.log_mode = log_mode
        self.size_div = size_div

    def set_data(self, T):

        def rolling_matrix(x, window_size):
            x = x.flatten()
            n = x.shape[0]
            stride = x.strides[0]
            return np.lib.stride_tricks.as_strided(x, shape=(n - window_size + 1, window_size),
                                                   strides=(stride, stride)).copy()

        def transform_data(T, n_train, n_test, time_step):
            np.random.seed(0)

            index_shuffle = np.random.permutation(n_train - time_step - 1)

            dT_train = np.ediff1d(T[:n_train])
            r_dT_train = rolling_matrix(dT_train, time_step + 1)[index_shuffle]

            dT_test = np.ediff1d(T[n_train - time_step - 1:n_train + n_test])
            r_dT_test = rolling_matrix(dT_test, time_step + 1)

            dT_train_input = r_dT_train[:, :-1].reshape(-1, time_step, 1)
            dT_train_target = r_dT_train[:, [-1]]
            dT_test_input = r_dT_test[:, :-1].reshape(-1, time_step, 1)
            dT_test_target = r_dT_test[:, [-1]]

            return [dT_train_input, dT_train_target, dT_test_input, dT_test_target]

        n = T.shape[0]
        [dT_train_input, dT_train_target, dT_test_input, dT_test_target] = transform_data(T, int(n * 0.8),
                                                                                          n - int(n * 0.8),
                                                                                          self.time_step)  # A sequence is divided into training and test data.

        self.dT_train_input = dT_train_input
        self.dT_train_target = dT_train_target
        self.dT_test_input = dT_test_input
        self.dT_test_target = dT_test_target

        self.n_train = dT_train_target.shape[0]
        self.n_test = dT_test_target.shape[0]

        self.t_max = np.max(np.vstack([self.dT_train_target, self.dT_test_target])) * 1.001

        # print("n_train: %d, n_test: %d, t_max: %f" % (self.n_train,self.n_test,self.t_max) )

        return self

    def set_model(self):

        if self.type_hazard == 'const':
            self.layer_hazard = HAZARD_const()
        elif self.type_hazard == 'exp':
            self.layer_hazard = HAZARD_exp()
        elif self.type_hazard == 'pc':
            self.layer_hazard = HAZARD_pc(size_div=self.size_div, t_max=self.t_max)
        elif self.type_hazard == 'NN':
            self.layer_hazard = HAZARD_NN(size_layer=self.size_layer, size_nn=self.size_nn,
                                          log_mode=self.log_mode).normalize_input(self.dT_train_target)

        self.rnn = RNN_PP(size_rnn=self.size_rnn, time_step=self.time_step, log_mode=self.log_mode).normalize_input(
            self.dT_train_target)

        input_dT = layers.Input(shape=(self.time_step, 1))
        input_x = layers.Input(shape=(1,))
        output_rnn = self.rnn(input_dT)
        [LL, log_l, Int_l] = self.layer_hazard([input_x, output_rnn])
        self.model = Model(inputs=[input_dT, input_x], outputs=[LL, log_l, Int_l])
        self.model.add_loss(-K.mean(LL))
        # model.summary()

        return self

    def compile(self, lr=1e-3):
        self.model.compile(keras.optimizers.Adam(lr=lr))
        return self

    def scores(self):
        scores = - self.model.predict([self.dT_test_input, self.dT_test_target], batch_size=self.n_test)[0].flatten()
        return scores



    def fit_eval(self, epochs=100, batch_size=256):

        es = CustomEarlyStopping()
        history = self.model.fit([self.dT_train_input, self.dT_train_target], epochs=epochs, batch_size=batch_size,
                                 validation_split=0.2, verbose=2, callbacks=[es])
        scores = self.scores()

        self.history = {'loss': np.array(history.history['loss']), 'val_loss': np.array(history.history['val_loss'])}
        self.val_loss = es.best_val_loss
        self.mnll = scores.mean()

        self.mae = self.mean_absolute_error()

        # print('score: %f' % scores.mean() )
        # print()

        return self

    def bisect_target(self, taus):
        return self.model.predict([self.dT_test_input, taus], batch_size=taus.shape[0])[2] - np.log(2)

    def median_prediction(self, l, r):

        for i in range(13):
            c = (l + r) / 2
            v = self.bisect_target(c)
            l = np.where(v < 0, c, l)
            r = np.where(v >= 0, c, r)

        return (l + r) / 2

    def mean_absolute_error(self):
        l = np.mean(self.dT_train_target) * 0.0001 * np.ones_like(self.dT_test_target)
        r = np.mean(self.dT_train_target) * 100.0 * np.ones_like(self.dT_test_target)
        tau_pred = self.median_prediction(l, r)
        return np.mean(np.abs(tau_pred - self.dT_test_target))
