import torch
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

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

class TPP_Dataset(Dataset):
    def __init__(self, x, y):
        super(TPP_Dataset, self).__init__()

        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)

def eval_normalize_param(x):
    mu_x_test = np.mean(x)
    return mu_x_test

def create_data_loader(train_dataset, test_dataset, batch_size, valid_split=0.2):
    # obtain training indices that will be used for validation
    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_split * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # load training data in batches
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=0)

    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=0)

    # load test data in batches
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              num_workers=0)

    return train_loader, test_loader, valid_loader


############################################################
class StrDataSet(Dataset):
    def __init__(self, x):
        super().__init__()
        self.x = x

    def __getitem__(self, item):
        return self.x[item]

    def __len__(self):
        return len(self.x)

if __name__ == '__main__':
    x = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    train_dataset = StrDataSet(list(x))
    test_dataset = StrDataSet(list(x))
    batch_size = 2

    train_loader, test_loader, valid_loader = create_data_loader(train_dataset, test_dataset, batch_size, valid_split=0.2)

    for batch, data in enumerate(train_loader, 1):
        print("--batch:", batch, "--data:", data)
    for data in valid_loader:
        print("--test data:", data)
