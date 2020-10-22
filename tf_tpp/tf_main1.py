import tpp_simulation as smlMgmt

from tf_tpp.tf_npp import NPP

########## Hyper-parameters
time_step = 20 # truncation depth of a RNN
size_rnn = 64  # the number of units in a RNN
size_div = 128 # the number of sub-intervals of the piecewise constant function (piecewise constant model)
size_nn = 64   # the number of units in each hidden layer of the cumulative hazard function network (neural network based model)
size_layer = 2 # the number of hidden layers of the cumulative hazard function network (neural network based model)

########## Generate data
# [T,score_ref] = smlMgmt.generate_stationary_poisson(True)
# [T,score_ref] = smlMgmt.generate_nonstationary_poisson(True)
# [T,score_ref] = smlMgmt.generate_stationary_renewal(True)
[T,score_ref] = smlMgmt.generate_nonstationary_renewal(True)


########## Train and evaluate the model (The following code will raise warnings, but please ignore them.)

# constant model
# npp1 = NPP(time_step=time_step,size_rnn=size_rnn,type_hazard='const').set_data(T).set_model().compile(lr=0.001).fit_eval(batch_size=256)

# exponential model
# npp2 = NPP(time_step=time_step,size_rnn=size_rnn,type_hazard='exp').set_data(T).set_model().compile(lr=0.001).fit_eval(batch_size=256)

# piecewise constant model
# npp3 = NPP(time_step=time_step,size_rnn=size_rnn,type_hazard='pc',size_div=size_div).set_data(T).set_model().compile(lr=0.001).fit_eval(batch_size=256)

# neural network based model
npp4 = NPP(time_step=time_step,size_rnn=size_rnn,type_hazard='NN',size_layer=size_layer,size_nn=size_nn).set_data(T).set_model().compile(lr=0.001).fit_eval(batch_size=256)

print()
print('#######################################')
print('## Performance for test data (MNLL: mean negative log-likelihood, MAE: mean absolute error)')
print('#######################################')
# print('## constand model             \n     MNLL: %.3f (standardized score: %.3f), MAE: %.3f' % (npp1.mnll,npp1.mnll-score_ref,npp1.mae) )
# print('## exponential model          \n     MNLL: %.3f (standardized score: %.3f), MAE: %.3f' % (npp2.mnll,npp2.mnll-score_ref,npp2.mae) )
# print('## piecewise constant model   \n     MNLL: %.3f (standardized score: %.3f), MAE: %.3f' % (npp3.mnll,npp3.mnll-score_ref,npp3.mae) )
print('## neural network based model \n     MNLL: %.3f (standardized score: %.3f), MAE: %.3f' % (npp4.mnll,npp4.mnll-score_ref,npp4.mae) )
