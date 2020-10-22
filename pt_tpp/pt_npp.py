import torch
import numpy as np

from pt_tpp.pt_rnn import PT_RNN
from pt_tpp.pt_hazard import PT_HAZARD

class PT_NPP(torch.nn.Module):
    def __init__(self, rnn_hidden_dim, hazard_hidden_dim, layers_num, log_mode=True):
        super(PT_NPP, self).__init__()

        self.rnn_hidden_dim = rnn_hidden_dim
        self.hazard_hidden_dim = hazard_hidden_dim
        self.log_mode = log_mode

        self.mu_x = None
        self.sigma_x = None
        self.mu_y = None
        self.sigma_y = None

        self.rnn = PT_RNN(input_dim=1, hidden_dim=rnn_hidden_dim, log_mode=log_mode)
        self.hazard = PT_HAZARD(rnn_output_dim=rnn_hidden_dim, hidden_dim=hazard_hidden_dim,
                                layers_num=layers_num, log_mode=log_mode)

    def forward(self, x, y):
        rnn_output = self.rnn(x, self.mu_x, self.sigma_x)
        int_l = self.hazard(rnn_output, y, self.mu_y, self.sigma_y)

        return int_l


    def positive_parameters(self):
        self.hazard.positive_parameters()

    # pred_y_ref: 预测结果 y 的参考值；
    # 由 参考值 得到预测结果的搜索范围: [万分之一参考值,  100倍参考值]
    def predict(self, x, pred_y_ref):
        sample_num = x.size()[0]
        l = pred_y_ref * 0.0001 * np.ones((sample_num, 1), dtype=np.float32)
        r = pred_y_ref * 100.0 * np.ones((sample_num, 1), dtype=np.float32)

        tau_pred = self.median_prediction(x, l, r)

        return tau_pred

    def bisect_target(self, x, taus):
        taus = torch.from_numpy(taus)
        return self.forward(x, taus) - np.log(2)

    def median_prediction(self, x, l, r):

        for i in range(13):
            c = (l + r) / 2
            v = self.bisect_target(x, c)
            l = np.where(v < 0, c, l)
            r = np.where(v >= 0, c, r)

        return (l + r) / 2

    # 设置数据规范化所需要的参数，x和y的均值、方差；
    # 这些参数由训练数据计算得到，不仅在训练时要用，在推理时也要用，
    # 所以把它们作为模型的参数
    def eval_normalize_params(self, x_train, y_train):
        if self.log_mode:
            self.mu_x = np.log(x_train).mean()
            self.sigma_x = np.log(x_train).std()

            self.mu_y = np.log(y_train).mean()
            self.sigma_y = np.log(y_train).std()
        else:
            self.mu_x = x_train.mean()
            self.sigma_x = x_train.std()

            self.mu_y = y_train.mean()
            self.sigma_y = y_train.std()
