"""
"""

import numpy as np


def missspecified_model_linearRegression2(N):
    X = np.concatenate([np.random.randn(N, 1), np.ones((N, 1))], axis=1)
    thetaT = np.array([[1], [1]])
    y = np.matmul(X, thetaT) + 0.5 * X[:, 0].reshape((-1, 1))**2 + np.random.randn(N, 1)
    return X, y,


def missspecified_noise_linearRegression2(N):
    X = np.concatenate([np.random.randn(N, 1), np.ones((N, 1))], axis=1)
    thetaT = np.array([[1], [1]])
    y = np.matmul(X, thetaT) + 2 * np.random.randn(N, 1)
    return X, y,


def correct_linearRegression2(N):
    X = np.concatenate([np.random.randn(N, 1), np.ones((N, 1))], axis=1)
    thetaT = np.array([[1], [1]])
    y = np.matmul(X, thetaT) + np.random.randn(N, 1)
    return X, y,


def gradient_field_problem(N):
    X = np.hstack([np.ones((N, 1)), np.random.lognormal(0, 0.75, (N, 1))])
    thetaT = np.array([[2], [2]])
    y = np.matmul(X, thetaT) + np.random.randn(N, 1)
    return X, y,


def wellspecified_logreg(N):
    mean1 = np.array([1, 1]).reshape((-1, 1))
    mean2 = np.array([-1, -1]).reshape((-1, 1))
    cov1 = np.identity(2) * 2
    cov2 = np.identity(2) * 2

    X1 = mean1.T + np.matmul(np.random.randn(int(N / 2), 2), cov1)
    X2 = mean2.T + np.matmul(np.random.randn(int(N / 2), 2), cov2)
    y1 = np.zeros((int(N / 2), 1))
    y2 = np.ones((int(N / 2), 1))

    X = np.vstack([X1, X2])
    y = np.vstack([y1, y2])
    return X, y,


def missnoise_logreg(N):
    mean1 = np.array([1.5, 1.5]).reshape((-1, 1))
    mean2 = np.array([-1.5, -1.5]).reshape((-1, 1))
    cov1 = np.identity(2) * 3
    cov2 = np.identity(2)

    X1 = mean1.T + np.matmul(np.random.randn(int(N / 2), 2), cov1)
    X2 = mean2.T + np.matmul(np.random.randn(int(N / 2), 2), cov2)
    y1 = np.zeros((int(N / 2), 1))
    y2 = np.ones((int(N / 2), 1))

    X = np.vstack([X1, X2])
    y = np.vstack([y1, y2])
    return X, y,


def misspecified_logreg(N):
    mean1 = np.array([1, 1]).reshape((-1, 1))
    mean2 = np.array([-1, -1]).reshape((-1, 1))
    cov1 = np.array([[1, -0.6], [-0.6, 1]]) * 1.5
    cov2 = np.array([[1, 0.6], [0.6, 1]]) * 1.5

    X1 = mean1.T + np.matmul(np.random.randn(int(N / 2), 2), cov1)
    X2 = mean2.T + np.matmul(np.random.randn(int(N / 2), 2), cov2)
    y1 = np.zeros((int(N / 2), 1))
    y2 = np.ones((int(N / 2), 1))

    X = np.vstack([X1, X2])
    y = np.vstack([y1, y2])
    return X, y,
