"""
"""

import numpy as np
import scipy as sp


def clip(p, threshold=10**-8):
    return np.maximum(np.minimum(p, (1.0 - threshold)), threshold)


def sigmoid(x):
    return clip(sp.special.expit(x))


def plogdet(x):
    sign, logdet = np.linalg.slogdet(x)
    assert(sign > 0)
    return logdet
