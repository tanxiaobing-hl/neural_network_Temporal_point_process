import torch
import numpy as np

def eval_loss_pt(int_l, y_grad):
    ll = torch.log(1e-10 + y_grad) - int_l
    loss = -torch.mean(ll)

    return loss

def mean_absolute_error(tau_pred, y):
    return np.mean(np.abs(tau_pred - y))
