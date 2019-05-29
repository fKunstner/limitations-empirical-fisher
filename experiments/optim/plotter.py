import numpy as np
import eftk
from efplt import plt
from efplt import matplotlib
import efplt

LINEWIDTH = 4
LAST_ITER = 100

###
# Helper functions
#


def collect_cosines(dname, problemFunc, lr, damping, N_ITERS, startingPointFunc):
    def loggingFunction(optimizer):
        theta = optimizer.theta
        problem = optimizer.problem

        g = problem.g(theta)
        EF = problem.ef(theta)
        ng = np.linalg.solve(problem.fisher(theta) + problem.hess_prior() + optimizer.damping, g)
        efg = np.linalg.solve(EF + problem.hess_prior() + optimizer.damping, g)

        def cos(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        return cos(ng, efg)

    dataset = eftk.datasets.load(dname).add_bias()
    problem = problemFunc(*dataset.get_train(), prior_var=np.inf)

    theta0 = startingPointFunc(problem.D)

    optimizer = eftk.optimizers.EFGD(problem, theta=theta0, lr=lr, damping=damping)
    cosines = list(optimizer.run(n_iter=N_ITERS, loggingFunc=loggingFunction))
    return cosines


def find_best_run(runs):
    return runs[np.argmin([run["losses"][-1] for run in runs])]


def make_angle_single_figure_grid():
    fig = plt.figure(figsize=(8, 3.25))
    gs1 = matplotlib.gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    gs2 = matplotlib.gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    gs1.update(left=0.15, right=0.9, bottom=0.475, top=0.88, hspace=0.15, wspace=0.15)
    gs2.update(left=0.15, right=0.9, bottom=0.18, top=0.425, hspace=0.15, wspace=0.15)

    return fig, fig.add_subplot(gs1[0]), fig.add_subplot(gs2[0])


def make_angle_figure_grid():
    fig = plt.figure(figsize=(12, 3))
    gs1 = matplotlib.gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
    gs2 = matplotlib.gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
    gs1.update(left=0.08, right=0.98, bottom=0.475, top=0.88, hspace=0.15, wspace=0.15)
    gs2.update(left=0.08, right=0.98, bottom=0.18, top=0.425, hspace=0.15, wspace=0.15)

    axes1 = [fig.add_subplot(gs1[0]), fig.add_subplot(gs1[1]), fig.add_subplot(gs1[2])]
    axes2 = [fig.add_subplot(gs2[0]), fig.add_subplot(gs2[1]), fig.add_subplot(gs2[2])]

    return fig, axes1, axes2


def find_best_hyperparameters(results):
    hyperparameters = {}
    for opt_name, opt_results in results.items():
        best_run = find_best_run(opt_results)
        hyperparameters[opt_name] = {
            "lr": best_run["lr"],
            "damping": best_run["damping"] if "damping" in best_run else None,
        }
    return hyperparameters


def run_different_startingpoints(dname, problemFunc, optimizers, hyperparameters, factors):
    dataset = eftk.datasets.load(dname)
    dataset.add_bias()
    problem = problemFunc(*dataset.get_train(), prior_var=np.inf)
    theta0 = eftk.solvers.lbfgs(problem)[0]

    def run_for_factor(factor, dataset_name, problemFunc, hyperparameters):

        results = {}
        for opt_name, opt_hyperparams in hyperparameters.items():
            optimizer = optimizers[opt_name](
                problem, theta=factor * theta0.copy(),
                lr=opt_hyperparams["lr"], damping=opt_hyperparams["damping"]
            )
            results[opt_name] = list(optimizer.run(n_iter=20))

        return results
    return list([run_for_factor(f, dataset_name=dname, problemFunc=problemFunc, hyperparameters=hyperparameters) for f in factors])


def make_starting_point_plot(dname, problemFunc, optimizers, results):
    fig = plt.figure(figsize=(6, 2.5))
    gs = matplotlib.gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    gs.update(left=0.1, right=0.95, bottom=0.15, top=0.85, hspace=0.2, wspace=0.1)

    def plot_1d(ax):
        LR = 0.5
        N_ITERS = 10
        thetas = [0.5, 1.0, 2.0, 4.0]

        ds = eftk.datasets.load("1D-Quadratic")
        problem = eftk.problem_defs.LinearRegression(*ds.get_train(), prior_var=np.inf)

        for opt_name, opt in optimizers.items():
            if opt_name is "GD":
                continue

            for i, theta0 in enumerate(thetas):
                res = list(opt(problem, lr=LR, theta=np.array([theta0]), damping=10**-6).run(n_iter=N_ITERS))
                if i == 0:
                    ax.plot(res, color=efplt.colors[opt_name], linewidth=LINEWIDTH, label=r"\bf{" + opt_name + "}", linestyle=efplt.linestyles[opt_name])
                else:
                    ax.plot(res, color=efplt.colors[opt_name], linewidth=LINEWIDTH, linestyle=efplt.linestyles[opt_name])

        ax.set_title(r"\LARGE $\mathbf{(x-0.5)^2 + (x+0.5)^2}$")
        ax.set_ylabel(r"\bf{Loss}")
        ax.set_xlabel("Iteration", labelpad=-15)
        ax.set_yscale("log")
        ax.set_xticks([0, 10])

        efplt.hide_frame([ax])

    def plot_boston(ax):

        factors = 1.0 - np.array([0.125, 0.25, 0.5, 1])
        hyperparameters = find_best_hyperparameters(results)

        different_startingpoints = run_different_startingpoints(dname, problemFunc, optimizers, hyperparameters, factors)

        for i, startingPoint in enumerate(different_startingpoints):
            ax.plot(startingPoint["NGD"], color=efplt.colors["NGD"], linewidth=LINEWIDTH, linestyle=efplt.linestyles["NGD"], label=r"\bf{NGD}" if i == 0 else None)
        for i, startingPoint in enumerate(different_startingpoints):
            ax.plot(startingPoint["EFGD"], color=efplt.colors["EFGD"], linewidth=LINEWIDTH, linestyle=efplt.linestyles["EFGD"], label=r"\bf{EFGD}" if i == 0 else None)

        ax.set_title(r"\bf{" + dname + "}")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration", labelpad=-15)

        efplt.hide_frame([ax])
        ax.set_xticks([0, 20])

    plot_boston(fig.add_subplot(gs[0]))
    fig.legend(loc='right', bbox_to_anchor=(0.998, 0.85), ncol=1, borderpad=.25, handletextpad=0.4, borderaxespad=0)
    return fig


def make_angle_single_plot(ax1, ax2, dname, problemFunc, results, N_ITERS, startingPointFunc, writelabels=False):
    ef_best_run = find_best_run(results[dname]["EFGD"])
    cosines = collect_cosines(dname, problemFunc, ef_best_run["lr"], ef_best_run["damping"], N_ITERS, startingPointFunc)

    print(dname + " --- Best hyperparameters")
    best_results = {}
    for opt_name, opt_results in results[dname].items():
        best_results[opt_name] = find_best_run(opt_results)
        print("    " + opt_name + ": gamma = 10^%.3f, lambda = 10^%.3f" % (np.log10(best_results[opt_name]["lr"]), np.log10(best_results[opt_name]["damping"]) if "damping" in best_results[opt_name] else np.nan))

    for opt_name, opt_result in best_results.items():
        ax1.plot(
            opt_result["losses"],
            linewidth=LINEWIDTH,
            linestyle=efplt.linestyles[opt_name],
            color=efplt.colors[opt_name],
            label=(r"\bf{" + opt_name + "}" if writelabels else None)
        )

    ax2.plot(cosines, '.', color=efplt.colors["EFGD"])

    ax2.set_ylim([-1, 1])
    ax2.set_yticklabels(["-1", "", "1"])

    ax1.set_title(r"\bf{" + dname + r"}", pad=5)
    ax2.set_xticklabels([0, 25, "", 75, 100])
    ax2.set_xlabel("Iteration", labelpad=-15)
    ax2.grid(axis="y")

    ax1.set_yscale("log")
    ax1.set_xlim([0, LAST_ITER])
    ax1.set_xticklabels([])
    ax2.set_xlim([0, LAST_ITER])

    efplt.hide_frame([ax1])
    efplt.hide_frame([ax2])


def make_individual_angle_plots(dataset_to_problems, results, N_ITERS, startingPointFunc):

    figures = []
    for i, dname in enumerate(dataset_to_problems):
        problemFunc = dataset_to_problems[dname]
        fig, ax1, ax2 = make_angle_single_figure_grid()
        make_angle_single_plot(ax1, ax2, dname, problemFunc, results, N_ITERS, startingPointFunc)

        if "Yacht" in dname:
            ax1.set_yticklabels([], minor=True)

        ax1.set_ylabel(r"\huge \bf{Loss}")
        ax2.set_ylabel(r"\huge \begin{gather*}\text{\textbf{Cosine}}\\[-.5em]\text{\textbf{(\LARGE NGD,EFG)}}\end{gather*}", labelpad=5)
        figures.append(fig)
    return figures


def make_angle_plot(dataset_to_problems, results, N_ITERS, startingPointFunc):

    fig, axes1, axes2 = make_angle_figure_grid()

    for i, dname in enumerate(dataset_to_problems):
        problemFunc = dataset_to_problems[dname]
        make_angle_single_plot(axes1[i], axes2[i], dname, problemFunc, results, N_ITERS, startingPointFunc, writelabels=(i == 0))

        if dname is "a1a":
            axes1[i].set_xlim([0, 50])
            axes2[i].set_xlim([0, 50])
            axes2[i].set_xticks([0, 25, 50])
            axes2[i].set_xticklabels([0, "", 50])
            axes1[i].set_ylim([axes1[i].get_ylim()[0], 1])
            axes1[i].set_yticklabels([], minor=True)

    axes1[0].set_ylabel(r"\huge \bf{Loss}")
    axes2[0].set_ylabel(r"\huge \begin{gather*}\text{\textbf{Cosine}}\\[-.5em]\text{\textbf{(\LARGE NGD,EFG)}}\end{gather*}", labelpad=5)

    fig.legend(loc='right', bbox_to_anchor=(1.01, 0.8), ncol=1, borderpad=.2, handletextpad=0.2, borderaxespad=0, labelspacing=0.3)

    return fig


###
# Main plotting functions functions
#


def plot(optimizers, dataset_to_problems, lrs, dampings, N_ITERS, startingPointFunc, results):
    angle_fig = make_angle_plot(dataset_to_problems, results, N_ITERS, startingPointFunc)
    startingpoint_fig = make_starting_point_plot("Boston", dataset_to_problems["Boston"], optimizers, results["Boston"])
    return angle_fig, startingpoint_fig


def plot_appendix(optimizers, dataset_to_problems, lrs, dampings, N_ITERS, startingPointFunc, results):
    angle_fig = make_angle_plot(dataset_to_problems, results, N_ITERS, startingPointFunc)
    indiv_figs = make_individual_angle_plots(dataset_to_problems, results, N_ITERS, startingPointFunc)
    return angle_fig, indiv_figs
