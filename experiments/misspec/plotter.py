import numpy as np
from efplt import plt
from efplt import matplotlib
import efplt

GRID_DENS = 50
LEVELS = 4
LINEWIDTH = 4

###
# Helper functions
#


def plot_loss_contour(ax, lossFunc, color=efplt.grays["medium"], makelabels=False):
    thetas = [
        np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], GRID_DENS),
        np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], GRID_DENS)
    ]

    def compute_losses():
        losses = np.zeros((GRID_DENS, GRID_DENS))
        for i, t1 in enumerate(thetas[0]):
            for j, t2 in enumerate(thetas[1]):
                theta = np.array([t1, t2]).reshape(-1, 1)
                losses[i, j] = lossFunc(theta)
        return losses

    losses = compute_losses()
    ax.contour(thetas[0], thetas[1], losses.T, LEVELS, colors=[color])
    if makelabels:
        ax.plot(-2 * np.abs(ax.get_xlim()[0]), 0, '-', color=color, label=r"\bf{Loss contour}")


def plot_quadratic_approximations(ax, problem, makelabels=False):
    H = problem.hess(problem.thetaStar) * problem.N
    EF = problem.ef(problem.thetaStar) * problem.N

    i = 3
    ax.plot(*efplt.ellipsis(i, (H), problem.thetaStar), linestyle=efplt.linestyles["NGD"], color=efplt.colors["NGD"], linewidth=LINEWIDTH, label=(r"\bf{Fisher}" if makelabels else None))
    ax.plot(*efplt.ellipsis(i, (EF), problem.thetaStar), linestyle=efplt.linestyles["EFGD"], color=efplt.colors["EF"], linewidth=LINEWIDTH, label=(r"\bf{emp. Fisher}" if makelabels else None))
    ax.plot(problem.thetaStar[0], problem.thetaStar[1], '*k', label=(r"\bf{Minimum}" if makelabels else None))


def plot_dataset(ax, X, y):
    def minibatch(X, y, B=10):
        idx_list = np.array_split(np.array(range(len(y))), B)
        return zip([X[idx, :] for idx in idx_list], [y[idx] for idx in idx_list])

    for X_, y_ in minibatch(X, y, B=200):
        ax.plot(X_[y_.reshape((-1,)) == 0, 0], X_[y_.reshape((-1,)) == 0, 1], '^', markersize=3, color=efplt.colors_classes[0], alpha=0.4)
        ax.plot(X_[y_.reshape((-1,)) == 1, 0], X_[y_.reshape((-1,)) == 1, 1], 'x', markersize=4, color=efplt.colors_classes[1], alpha=0.4)


def plot_dataset_linear(ax, X, y):
    thetaStar = np.linalg.lstsq(X, y.reshape((-1,)), rcond=None)[0]

    ax.plot(X[:, 0], y, 'x', markersize=4, color=efplt.colors_classes[1], alpha=0.4)

    xs = np.array([[ax.get_xlim()[0], 1], [ax.get_xlim()[1], 1]])
    ys = xs @ thetaStar.reshape((-1,))

    ax.plot(ax.get_xlim(), ys, '-', color=efplt.colors_classes[0], linewidth=LINEWIDTH / 2)


def make_figure():
    fig = plt.figure(figsize=(12, 4))
    gs1 = matplotlib.gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
    gs2 = matplotlib.gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
    gs1.update(left=0.06, right=0.995, bottom=0.6, top=0.9, hspace=0.45, wspace=0.1)
    gs2.update(left=0.06, right=0.995, bottom=0.1, top=0.55, hspace=0.45, wspace=0.1)

    ax11 = fig.add_subplot(gs1[0])
    ax12 = fig.add_subplot(gs1[1])
    ax13 = fig.add_subplot(gs1[2])
    ax21 = fig.add_subplot(gs2[0])
    ax22 = fig.add_subplot(gs2[1])
    ax23 = fig.add_subplot(gs2[2])
    axes = [ax11, ax12, ax13, ax21, ax22, ax23]
    return fig, axes

###
# Main plotting functions
#


def plot(Xs, ys, ps):
    fig, axes = make_figure()

    for i in range(len(Xs)):
        plot_dataset(axes[i], Xs[i], ys[i])

    for i, ax in enumerate(axes[:3]):
        ax.set_xlim([-9, 9])
        ax.set_ylim([-9, 9])

    axes[3].set_xlim([-0.7, -0.3])
    axes[3].set_ylim([-0.7, -0.3])
    axes[4].set_xlim([-0.8, -0.4])
    axes[4].set_ylim([-0.8, -0.4])
    axes[5].set_xlim([-1.0, -0.5])
    axes[5].set_ylim([-1.0, -0.5])

    axes[0].set_title(r"\bf{Correct}")
    axes[1].set_title(r"\bf{Misspecified (A)}")
    axes[2].set_title(r"\bf{Misspecified (B)}")

    for i, p in enumerate(ps):
        plot_loss_contour(axes[3 + i], p.loss, makelabels=(i == 0))
        plot_quadratic_approximations(axes[3 + i], p, makelabels=(i == 0))

    efplt.strip_axes(axes)

    axes[0].set_ylabel(r"{\huge \bf{Dataset}}", labelpad=10)
    axes[3].set_ylabel(r"{\huge \begin{gather*}\bf{Quadratic}\\[-.5em]\bf{approximation}\end{gather*}", labelpad=10)

    fig.legend(loc='lower center', bbox_to_anchor=(0.06, 0.0, 0.995 - 0.06, 0), ncol=4, borderpad=.25, borderaxespad=0, handlelength=4.0, mode="expand")

    return fig


def plot_appendix(Xs, ys, ps):
    fig, axes = make_figure()

    for i in range(len(Xs)):
        plot_dataset_linear(axes[i], Xs[i], ys[i])

    for i, ax in enumerate(axes[:3]):
        ax.set_xlim([-4, 4])
        ax.set_ylim([-6, 10])

    axes[0].set_title(r"\bf{Correct}")
    axes[1].set_title(r"\bf{Misspecified (A)}")
    axes[2].set_title(r"\bf{Misspecified (B)}")

    axes[3].set_xlim([0.8, 0.8 + 0.3])
    axes[4].set_xlim([0.775, 0.775 + 0.3])
    axes[5].set_xlim([0.775, 0.775 + 0.3])
    axes[3].set_ylim([0.85, 0.85 + 0.3])
    axes[4].set_ylim([0.85, 0.85 + 0.3])
    axes[5].set_ylim([1.325, 1.325 + 0.3])
    for i, p in enumerate(ps):
        plot_loss_contour(axes[3 + i], p.loss, makelabels=(i == 0))
        plot_quadratic_approximations(axes[3 + i], p, makelabels=(i == 0))

    efplt.strip_axes(axes)
    axes[0].set_ylabel(r"{\huge \bf{Dataset}}", labelpad=10)
    axes[3].set_ylabel(r"{\huge \begin{gather*}\bf{Quadratic}\\[-.5em]\bf{approximation}\end{gather*}", labelpad=10)

    fig.legend(loc='lower center', bbox_to_anchor=(0.06, 0.0, 0.995 - 0.06, 0), ncol=4, borderpad=.25, borderaxespad=0, handlelength=4.0, mode="expand")

    return fig
