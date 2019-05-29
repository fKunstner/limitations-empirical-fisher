import numpy as np
from tqdm import tqdm

"""
Configuration and magic strings
"""

N_ITER = 50000
STEP_SIZE = 0.001


def run(vectorFuncs, startingPoints):

    def gd(vecFunc, startingPoint):
        xs = np.zeros((N_ITER, 2))
        xs[0, :] = startingPoint.copy().reshape((-1,))

        for t in tqdm(range(1, N_ITER), leave=False):
            xs[t] = (xs[t - 1] + STEP_SIZE * vecFunc(xs[t - 1])).reshape((-1,))

        return xs

    results = []
    for i, startingpoint in tqdm(enumerate(startingPoints), total=len(startingPoints), leave=False):
        results.append([])
        for j, vecfunc in tqdm(enumerate(vectorFuncs), total=len(vectorFuncs), leave=False):
            results[i].append(gd(vecfunc, startingpoint))

    return results
