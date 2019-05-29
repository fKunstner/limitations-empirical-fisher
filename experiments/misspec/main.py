import eftk
import numpy as np

from . import plotter


N = 1000


def load_problem_appendix():
    datasetFuncs = [
        eftk.toydata.correct_linearRegression2,
        eftk.toydata.missspecified_noise_linearRegression2,
        eftk.toydata.missspecified_model_linearRegression2
    ]

    Xs = []
    ys = []
    ps = []

    for datasetFunc in datasetFuncs:
        np.random.seed(0)
        X, y = datasetFunc(N)
        p = eftk.problem_defs.LinearRegression(X, y)
        p.thetaStar, _ = eftk.solvers.cg(p)
        Xs.append(X)
        ys.append(y)
        ps.append(p)

    return Xs, ys, ps


def load_problem():
    datasetFuncs = [
        eftk.toydata.wellspecified_logreg,
        eftk.toydata.missnoise_logreg,
        eftk.toydata.misspecified_logreg
    ]

    Xs = []
    ys = []
    ps = []

    for datasetFunc in datasetFuncs:
        np.random.seed(0)
        X, y = datasetFunc(N)
        p = eftk.problem_defs.LogisticRegression(X, y)
        p.thetaStar, _ = eftk.solvers.lbfgs(p)
        Xs.append(X)
        ys.append(y)
        ps.append(p)

    return Xs, ys, ps


def run():
    print("This experiment has no pre-computation to run.")


def plot():
    Xs, ys, ps = load_problem()
    fig = plotter.plot(Xs, ys, ps)
    return [fig]


def run_appendix():
    print("This experiment's appendix has no pre-computation to run.")


def plot_appendix():
    Xs, ys, ps = load_problem_appendix()
    fig = plotter.plot_appendix(Xs, ys, ps)
    return [fig]
