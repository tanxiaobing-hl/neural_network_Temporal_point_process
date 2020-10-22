import numpy as np
import os

from scipy.stats import lognorm,gamma
from scipy.optimize import brentq

def use_reserved_data(file_name):
    dir_name = os.path.dirname(__file__)
    rst = np.load('%s/data/%s.npy' % (dir_name, file_name), allow_pickle=True).tolist()
    return rst

def save_tpp_data(file_name, rst):
    dir_name = os.path.dirname(__file__)
    np.save('%s/data/%s.npy' % (dir_name, file_name), rst)

######################################################
### stationary poisson process
######################################################
def generate_stationary_poisson(use_reserved=False):
    file_name = 'stationary_poisson'
    if use_reserved:
        return use_reserved_data(file_name)

    tau = np.random.exponential(size=100000)
    T = tau.cumsum()
    score = 1

    save_tpp_data(file_name, [T, score])
    return [T, score]


######################################################
### non-stationary poisson process
######################################################
def generate_nonstationary_poisson(use_reserved=False):
    file_name = 'nonstationary_poisson'
    if use_reserved:
        return use_reserved_data(file_name)

    L = 20000
    amp = 0.99
    l_t = lambda t: np.sin(2 * np.pi * t / L) * amp + 1
    l_int = lambda t1, t2: - L / (2 * np.pi) * (np.cos(2 * np.pi * t2 / L) - np.cos(2 * np.pi * t1 / L)) * amp + (
                t2 - t1)

    while 1:
        T = np.random.exponential(size=210000).cumsum() * 0.5
        r = np.random.rand(210000)
        index = r < l_t(T) / 2.0

        if index.sum() > 100000:
            T = T[index][:100000]
            score = - (np.log(l_t(T[80000:])).sum() - l_int(T[80000 - 1], T[-1])) / 20000
            break

    save_tpp_data(file_name, [T, score])
    return [T, score]


######################################################
### stationary renewal process
######################################################
def generate_stationary_renewal(use_reserved=False):
    file_name = 'stationary_renewal'
    if use_reserved:
        return use_reserved_data(file_name)

    s = np.sqrt(np.log(6 * 6 + 1))
    mu = -s * s / 2
    tau = lognorm.rvs(s=s, scale=np.exp(mu), size=100000)
    lpdf = lognorm.logpdf(tau, s=s, scale=np.exp(mu))
    T = tau.cumsum()
    score = - np.mean(lpdf[80000:])

    save_tpp_data(file_name, [T, score])
    return [T, score]


######################################################
### non-stationary renewal process
######################################################
def generate_nonstationary_renewal(use_reserved=False):
    file_name = 'nonstationary_renewal'
    if use_reserved:
        return use_reserved_data(file_name)

    L = 20000
    amp = 0.99
    l_t = lambda t: np.sin(2 * np.pi * t / L) * amp + 1
    l_int = lambda t1, t2: - L / (2 * np.pi) * (np.cos(2 * np.pi * t2 / L) - np.cos(2 * np.pi * t1 / L)) * amp + (
                t2 - t1)

    T = []
    lpdf = []
    x = 0

    k = 4
    rs = gamma.rvs(k, size=100000)
    lpdfs = gamma.logpdf(rs, k)
    rs = rs / k
    lpdfs = lpdfs + np.log(k)

    for i in range(100000):
        x_next = brentq(lambda t: l_int(x, t) - rs[i], x, x + 1000)
        l = l_t(x_next)
        T.append(x_next)
        lpdf.append(lpdfs[i] + np.log(l))
        x = x_next

    T = np.array(T)
    lpdf = np.array(lpdf)
    score = - lpdf[80000:].mean()

    save_tpp_data(file_name, [T, score])
    return [T, score]


######################################################
### self-correcting process
######################################################
def generate_self_correcting(use_reserved=False):
    def self_correcting_process(mu, alpha, n):
        t = 0;
        x = 0;
        T = [];
        log_l = [];
        Int_l = [];

        for i in range(n):
            e = np.random.exponential()
            tau = np.log(e * mu / np.exp(x) + 1) / mu  # e = ( np.exp(mu*tau)- 1 )*np.exp(x) /mu
            t = t + tau
            T.append(t)
            x = x + mu * tau
            log_l.append(x)
            Int_l.append(e)
            x = x - alpha

        return [np.array(T), np.array(log_l), np.array(Int_l)]

    file_name = 'self_correcting'
    if use_reserved:
        return use_reserved_data(file_name)

    [T, log_l, Int_l] = self_correcting_process(1, 1, 100000)
    score = - (log_l[80000:] - Int_l[80000:]).sum() / 20000

    save_tpp_data(file_name, [T, score])
    return [T, score]


######################################################
### Hawkes process
######################################################
def generate_hawkes1(use_reserved=False):
    file_name = 'hawkes1'
    if use_reserved:
        return use_reserved_data(file_name)

    [T, LL] = simulate_hawkes(100000, 0.2, [0.8, 0.0], [1.0, 20.0])
    score = - LL[80000:].mean()

    save_tpp_data(file_name, [T, score])
    return [T, score]


def generate_hawkes2(use_reserved=False):
    file_name = 'hawkes2'
    if use_reserved:
        return use_reserved_data(file_name)

    [T, LL] = simulate_hawkes(100000, 0.2, [0.4, 0.4], [1.0, 20.0])
    score = - LL[80000:].mean()

    save_tpp_data(file_name, [T, score])
    return [T, score]


def simulate_hawkes(n, mu, alpha, beta, use_reserved=False):
    file_name = 'hawkes'
    if use_reserved:
        return use_reserved_data(file_name)

    T = []
    LL = []

    x = 0
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0

    while 1:
        l = mu + l_trg1 + l_trg2
        step = np.random.exponential() / l
        x = x + step

        l_trg_Int1 += l_trg1 * (1 - np.exp(-beta[0] * step)) / beta[0]
        l_trg_Int2 += l_trg2 * (1 - np.exp(-beta[1] * step)) / beta[1]
        mu_Int += mu * step
        l_trg1 *= np.exp(-beta[0] * step)
        l_trg2 *= np.exp(-beta[1] * step)
        l_next = mu + l_trg1 + l_trg2

        if np.random.rand() < l_next / l:  # accept
            T.append(x)
            LL.append(np.log(l_next) - l_trg_Int1 - l_trg_Int2 - mu_Int)
            l_trg1 += alpha[0] * beta[0]
            l_trg2 += alpha[1] * beta[1]
            l_trg_Int1 = 0
            l_trg_Int2 = 0
            mu_Int = 0
            count += 1

            if count == n:
                break

    save_tpp_data(file_name, [np.array(T), np.array(LL)])
    return [np.array(T), np.array(LL)]
