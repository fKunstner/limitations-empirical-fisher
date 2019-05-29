from . import problem_defs
from . import solvers
from . import losses
from . import helpers
import numpy as np


class Model():
    def evaluate(self, X, y):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class Linear(Model):
    def __init__(self, noise=1, prior_var=10**-8):
        self.noise = noise
        self.prior_var = prior_var

    def fit(self, X, y):
        placeholder_noise = 1 if self.noise is None else self.noise

        self.problem = problem_defs.LinearRegression(X, y, self.prior_var, placeholder_noise)
        self.theta, _ = solvers.cg(self.problem)
        self.theta = self.theta.reshape((-1,))

        if self.noise is None:
            self.noise = np.var(self.predict(X) - y)
            self.problem.noise = self.noise

    def evaluate(self, X, y):
        return losses.MSE(self.predict(X), y), losses.gaussian_negloglik(y, *self.predict_var(X))

    def predict(self, X):
        return X @ self.theta

    def predict_var(self, X):
        return self.predict(X), np.ones((X.shape[0], 1)) * self.noise


class Logistic(Model):
    def __init__(self, prior_var=10**-8):
        self.prior_var = prior_var

    def fit(self, X, y):
        self.problem = problem_defs.LogisticRegression(X, y, self.prior_var)
        self.theta, _ = solvers.lbfgs(self.problem)
        self.theta = self.theta.reshape((-1,))

    def evaluate(self, X, y):
        return losses.binary_negloglik(self.predict(X), y), losses.accuracy(self.predict(X).round(), y)

    def predict(self, X):
        return helpers.sigmoid(X @ self.theta)
