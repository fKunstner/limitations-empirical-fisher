"""
Utils to interface with scipy opt
"""

import numpy as np
import scipy as sp


def cg(problem):
    theta = np.zeros((problem.D, 1))

    for i in range(2):
        H = problem.hess(theta)
        g = problem.g(theta)
        dif, info = sp.sparse.linalg.cg(H, g)
        assert(info == 0)
        theta -= dif.reshape((problem.D, 1))

    return theta, info


def lbfgs(problem):
    theta = np.zeros(problem.D)
    res = sp.optimize.minimize(fun=problem.loss, x0=theta, jac=problem.g)
    return res.x, res.message
