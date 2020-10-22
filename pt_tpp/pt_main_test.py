import torch
import numpy as np
import tpp_simulation as smlMgmt
import torch.optim as optim

from pt_tpp.early_stopping import EarlyStopping
from pt_tpp.load_data import TPP_Dataset, transform_data, create_data_loader, eval_normalize_param

from pt_tpp.pt_npp import PT_NPP
from pt_tpp.pt_loss import eval_loss_pt, mean_absolute_error

########## Hyper-parameters
time_step = 20 # truncation depth of a RNN
size_rnn = 64  # the number of units in a RNN
size_div = 128 # the number of sub-intervals of the piecewise constant function (piecewise constant model)
size_nn = 64   # the number of units in each hidden layer of the cumulative hazard function network (neural network based model)
size_layer = 2 # the number of hidden layers of the cumulative hazard function network (neural network based model)
batch_size = 256
log_mode = True
epoch_num = 100

########## Generate data
# [T, score_ref] = smlMgmt.generate_stationary_poisson(True)     # generate synthetic data: stationary Poisson process.
[T,score_ref] = smlMgmt.generate_nonstationary_poisson(True)     # generate synthetic data: stationary Poisson process.
n = T.shape[0]
[dT_train_input, dT_train_target, dT_test_input, dT_test_target] = transform_data(T, int(n * 0.8),
                                                                                  n - int(n * 0.8),
                                                                                  time_step)  # A sequence is divided into training and test data.

train_dataset = TPP_Dataset(dT_train_input, dT_train_target)
test_dataset = TPP_Dataset(dT_test_input, dT_test_target)

train_loader, test_loader, valid_loader = create_data_loader(train_dataset, test_dataset, batch_size, valid_split=0.2)

mu_y_train = eval_normalize_param(dT_train_target)

npp = PT_NPP(rnn_hidden_dim=size_rnn, hazard_hidden_dim=size_nn, layers_num=size_layer, log_mode=log_mode)
npp.eval_normalize_params(dT_train_input, dT_train_target)

#########--train model--######################
patience = 8
# to track the training loss as the model trains
train_losses = []
# to track the validation loss as the model trains
valid_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
avg_valid_losses = []

# load the last checkpoint with the best model
npp.load_state_dict(torch.load('checkpoint.pt'))

# 模型测试
x, y = test_dataset[:]
tau_pred = npp.predict(x, mu_y_train)
abs_error = mean_absolute_error(tau_pred, y.numpy())
print("abs_error:", abs_error)
