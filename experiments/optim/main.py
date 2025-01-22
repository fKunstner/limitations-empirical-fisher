import pickle as pk
import eftk
import numpy as np

from . import runner
from . import plotter


def save_results(res):
    with open("optim.pk", "wb") as fh:
        pk.dump(res, fh)


def load_results():
    with open("optim.pk", "rb") as fh:
        return pk.load(fh)


def save_results_appendix(res):
    with open("optim-apx.pk", "wb") as fh:
        pk.dump(res, fh)


def load_results_appendix():
    with open("optim-apx.pk", "rb") as fh:
        return pk.load(fh)


def load_problem():
    OPTIMIZERS = {
        "GD": eftk.optimizers.GD,
        "NGD": eftk.optimizers.NGD,
        "EFGD": eftk.optimizers.EFGD,
    }
    DATASET_TO_PROBLEMS = {
        "BreastCancer": eftk.problem_defs.LogisticRegression,
        "Wine": eftk.problem_defs.LinearRegression,
        "a1a": eftk.problem_defs.LogisticRegression,
    }
    LRS = np.logspace(-20, 10, 241)
    DAMPINGS = np.logspace(-10, 10, 41)
    print(LRS)
    print(DAMPINGS)
    N_ITERS = 100

    def startingPointFunc(D):
        return np.zeros((D,))

    return OPTIMIZERS, DATASET_TO_PROBLEMS, LRS, DAMPINGS, N_ITERS, startingPointFunc


def load_problem_appendix():
    OPTIMIZERS = {
        "GD": eftk.optimizers.GD,
        "NGD": eftk.optimizers.NGD,
        "EFGD": eftk.optimizers.EFGD,
    }
    DATASET_TO_PROBLEMS = {
        "Wine": eftk.problem_defs.LinearRegression,
        "Energy": eftk.problem_defs.LinearRegression,
        "Powerplant": eftk.problem_defs.LinearRegression,
        "Yacht": eftk.problem_defs.LinearRegression,
    }
    LRS = np.logspace(-20, 10, 241)
    DAMPINGS = np.logspace(-10, 10, 41)
    print(LRS)
    print(DAMPINGS)
    N_ITERS = 100

    def startingPointFunc(D):
        return np.zeros((D,))

    return OPTIMIZERS, DATASET_TO_PROBLEMS, LRS, DAMPINGS, N_ITERS, startingPointFunc


def run():
    save_results(runner.run(*load_problem()))


def plot():
    return plotter.plot(*load_problem(), load_results())


def run_appendix():
    save_results_appendix(runner.run(*load_problem_appendix()))


def plot_appendix():
    return plotter.plot_appendix(*load_problem_appendix(), load_results_appendix())
