import eftk
import numpy as np
from tqdm import tqdm as tqdm


def run(optimizers, dataset_to_problems, lrs, dampings, N_ITERS, startingPointFunc):

    def run_optimizer(problem, opt, lr, damping, theta0):
        optimizer = opt(problem, lr=lr, theta=theta0.copy(), damping=damping)

        steps = []
        for step in tqdm(optimizer.run(n_iter=N_ITERS), total=N_ITERS):
            steps.append(step)
            diverging = steps[-1] > steps[0] * 100
            if diverging:
                break
        return steps

    results = {}
    for i, dname in tqdm(enumerate(dataset_to_problems), total=len(dataset_to_problems), leave=False):
        results[dname] = {}

        dataset = eftk.datasets.load(dname)
        dataset.add_bias()
        problem = dataset_to_problems[dname](*dataset.get_train(), prior_var=np.inf)

        theta0 = startingPointFunc(problem.D)

        for j, opt_name in tqdm(enumerate(optimizers), total=len(optimizers), leave=False):
            results[dname][opt_name] = []

            for k, lr in tqdm(enumerate(lrs), total=len(lrs), leave=False):
                if opt_name is "GD":
                    run = {
                        "lr": lr,
                        "losses": run_optimizer(problem, optimizers[opt_name], lr, None, theta0),
                    }
                    results[dname][opt_name].append(run)
                else:
                    for l, damping in tqdm(enumerate(dampings), total=len(dampings), leave=False):
                        run = {
                            "lr": lr,
                            "damping": damping,
                            "losses": run_optimizer(problem, optimizers[opt_name], lr, damping, theta0),
                        }
                        results[dname][opt_name].append(run)
    return results
