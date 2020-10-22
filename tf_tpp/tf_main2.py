import tpp_simulation as smlMgmt

from tf_tpp.tf_npp import NPP


class training():

    def training(self, type_hazard, T):
        min_val_loss = 100000
        for time_step in [5, 10, 20, 40]:
            npp = NPP(time_step=time_step, type_hazard=type_hazard, size_rnn=64, size_layer=2, size_nn=64, size_div=128,
                      log_mode=True).set_data(T).set_model().compile(lr=0.001).fit_eval(batch_size=256)
            if npp.val_loss < min_val_loss:
                min_val_loss = npp.val_loss
                self.mnll = npp.mnll
                self.mae = npp.mae

        return self

########## Generate data
[T,score_ref] = smlMgmt.generate_stationary_poisson()     # generate synthetic data: stationary Poisson process.
print('#######################################')
print('## Stationary Poisson process')
print('#######################################')
print()

########## Train and evaluate the model (The following code will raise warnings, but please ignore them.)

# constant model
npp1 = training().training('const',T)

# exponential model
npp2 = training().training('exp',T)

# piecewise constant model
npp3 = training().training('pc',T)

# neural network based model
npp4 = training().training('NN',T)

print()
print('#######################################')
print('## Performance for test data (MNLL: mean negative log-likelihood, MAE: mean absolute error)')
print('#######################################')
print('## constand model             \n     MNLL: %.3f (standardized score: %.3f), MAE: %.3f' % (npp1.mnll,npp1.mnll-score_ref,npp1.mae) )
print('## exponential model          \n     MNLL: %.3f (standardized score: %.3f), MAE: %.3f' % (npp2.mnll,npp2.mnll-score_ref,npp2.mae) )
print('## piecewise constant model   \n     MNLL: %.3f (standardized score: %.3f), MAE: %.3f' % (npp3.mnll,npp3.mnll-score_ref,npp3.mae) )
print('## neural network based model \n     MNLL: %.3f (standardized score: %.3f), MAE: %.3f' % (npp4.mnll,npp4.mnll-score_ref,npp4.mae) )
