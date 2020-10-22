import torch
import numpy as np



class PT_RNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, log_mode=True):
        super(PT_RNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.log_mode = log_mode

        self.rnn = torch.nn.RNN(input_dim, hidden_dim, num_layers=1, batch_first=True)
        # torch.nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        # torch.nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        # torch.nn.init.zeros_(self.rnn.bias_ih_l0)
        # torch.nn.init.zeros_(self.rnn.bias_hh_l0)

    def forward(self, x, mu_x, sigma_x):
        if self.log_mode:
            x_nmlz = (torch.log(x) - mu_x) / sigma_x
        else:
            x_nmlz = (x - mu_x)/sigma_x

        output, h_n = self.rnn(x_nmlz)
        h_n = h_n.view(-1, self.hidden_dim)
        return h_n
