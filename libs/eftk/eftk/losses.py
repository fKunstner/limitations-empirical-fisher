import numpy as np
from . import helpers
import scipy as sp


def MSE(a, b):
    assert(len(a) == len(b))
    return np.linalg.norm(a.reshape((-1, 1)) - b.reshape((-1, 1)))**2 / (2 * len(a))


def gaussian_negloglik(y, yh, var):
    assert(y.size == yh.size)

    y_ = y.reshape((-1,))
    yh_ = yh.reshape((-1,))

    if yh.size**2 == var.size:
        assert(len(var.shape) == 2 and var.shape[0] == var.shape[1])
        return -sp.stats.multivariate_normal(mean=yh_, cov=var).logpdf(y_) / len(y_)
    elif yh.size == var.size:
        var_ = np.diag(var.reshape((-1,)))
        return -sp.stats.multivariate_normal(mean=yh_, cov=var_).logpdf(y_) / len(y_)
    else:
        raise NotImplementedError


def binary_negloglik(p, y):
    return - (np.log(p) * y + np.log(1 - p) * (1 - y)).mean()


def accuracy(yh, y):
    return (yh == y).mean()
