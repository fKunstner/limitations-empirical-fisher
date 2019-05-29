"""
Basic optimizers with logging
"""

import numpy as np


class Optimizer():
    def __init__(self, problem, theta=None):
        self.problem = problem
        if theta is None:
            self.theta = np.zeros((problem.D, ))
        else:
            self.theta = theta

    def run(self, n_iter=100, loggingFunc=lambda opt: opt.problem.loss(opt.theta)):
        for t in range(n_iter + 1):
            if t > 0:
                self.iteration()
            yield loggingFunc(self)

    def iteration(self):
        raise NotImplementedError


class GD(Optimizer):
    def __init__(self, problem, theta=None, lr=0.01, damping=None):
        super().__init__(problem, theta)
        self.lr = lr

    def iteration(self):
        self.theta -= self.lr * self.problem.g(self.theta)


class NGD(Optimizer):
    def __init__(self, problem, theta=None, lr=1, damping=10**-6):
        super().__init__(problem, theta)
        self.lr = lr
        self.damping = np.eye(problem.D) * damping

    def iteration(self):
        self.theta -= self.lr * np.linalg.solve(self.problem.fisher(self.theta) + self.problem.hess_prior() + self.damping, self.problem.g(self.theta))


class EFGD(Optimizer):
    def __init__(self, problem, theta=None, lr=1, damping=10**-6):
        super().__init__(problem, theta)
        self.lr = lr
        self.damping = np.eye(problem.D) * damping

    def iteration(self):
        self.theta -= self.lr * np.linalg.solve(self.problem.ef(self.theta) + self.problem.hess_prior() + self.damping, self.problem.g(self.theta))
