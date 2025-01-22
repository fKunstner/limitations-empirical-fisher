import pickle as pk
import eftk
import numpy as np

from . import runner
from . import plotter


def save_results(res):
    with open("vecfield.pk", "wb") as fh:
        pk.dump(res, fh)


def load_results():
    with open("vecfield.pk", "rb") as fh:
        return pk.load(fh)


def load_problem():
    np.random.seed(0)
    N = 1000
    X, y = eftk.toydata.gradient_field_problem(N)
    problem = eftk.problem_defs.LinearRegression(X, y)
    problem.thetaStar, _ = eftk.solvers.cg(problem)
    gammas = [1 / 3, 1, 3]

    def vectorFunctions(gammas, problem):
        return [
            lambda t: - gammas[0] * problem.g(t),
            lambda t: - gammas[1] * np.linalg.solve(problem.hess(t) + (1e-8) * np.eye(2), problem.g(t)),
            lambda t: - gammas[2] * np.linalg.solve(problem.ef(t) + (1e-8) * np.eye(2), problem.g(t)),
        ]

    startingPoints = [
        np.array([2, 4.5]).reshape((-1, 1)),
        np.array([1, 0]).reshape((-1, 1)),
        np.array([4.5, 3]).reshape((-1, 1)),
        np.array([-0.5, 3]).reshape((-1, 1)),
    ]

    return vectorFunctions(gammas, problem), startingPoints, X, y, problem


def run():
    vectorFuncs, startingPoints, X, y, problem = load_problem()
    results = runner.run(vectorFuncs, startingPoints)
    save_results(results)


def plot():
    vectorFuncs, startingPoints, X, y, problem = load_problem()
    results = load_results()
    fig = plotter.plot(X, y, problem, vectorFuncs, startingPoints, results)
    return [fig]


def run_appendix():
    print("This experiment has no appendix.")


def plot_appendix():
    print("This experiment has no appendix.")
