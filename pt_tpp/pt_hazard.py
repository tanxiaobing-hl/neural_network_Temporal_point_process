import torch

def linear_init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

class PT_HAZARD(torch.nn.Module):
    def __init__(self, rnn_output_dim, hidden_dim, layers_num, log_mode):
        super(PT_HAZARD, self).__init__()

        self.rnn_output_dim = rnn_output_dim
        self.hidden_dim = hidden_dim
        self.layers_num = layers_num
        self.log_mode = log_mode

        self.linear_y = torch.nn.Linear(in_features=1, out_features=hidden_dim, bias=False)
        # torch.nn.init.xavier_uniform_(self.linear_y.weight)

        self.linear_rnn = torch.nn.Linear(in_features=rnn_output_dim, out_features=hidden_dim)
        # torch.nn.init.xavier_uniform_(self.linear_rnn.weight)
        # torch.nn.init.zeros_(self.linear_rnn.bias)

        self.linear_list = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Tanh())
            for _ in range(layers_num-1)])
        # for dense in self.linear_list:
        #     dense.apply(linear_init_weights)

        self.softplus = torch.nn.Sequential(torch.nn.Linear(hidden_dim, 1), torch.nn.Softplus())
        # self.softplus.apply(linear_init_weights)

    def forward(self, x, y, mu_y, sigma_y):
        if self.log_mode:
            y_nmlz = (torch.log(y)- mu_y)/sigma_y
        else:
            y_nmlz = (y - mu_y)/sigma_y

        hidden_x = self.linear_rnn(x)
        hidden_y = self.linear_y(y_nmlz)

        hidden = torch.nn.Tanh()(hidden_x + hidden_y)

        for i, dense in enumerate(self.linear_list):
            hidden = dense(hidden)

        int_l = self.softplus(hidden)

        return int_l

    def positive_parameters(self):
        for p in self.linear_y.parameters():
            p.data.abs_()

        for p in self.linear_list.parameters():
            p.data.abs_()

        for p in self.softplus.parameters():
            p.data.abs_()
