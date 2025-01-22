import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy as sp


font = {'family': 'serif', 'size': 18}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"


def rgb_to_u(xs):
    return list([x / 255.0 for x in xs])


colorschemes = {
    "PT": {
        "High-contrast": {
            "white": rgb_to_u([255, 255, 255]),
            "yellow": rgb_to_u([221, 170, 51]),
            "red": rgb_to_u([187, 85, 102]),
            "blue": rgb_to_u([0, 68, 136]),
            "black": rgb_to_u([0, 0, 0]),
        },
        "Vibrant": {
            "blue": rgb_to_u([0, 119, 187]),
            "red": rgb_to_u([204, 51, 17]),
            "orange": rgb_to_u([238, 119, 51]),
            "cyan": rgb_to_u([51, 187, 238]),
            "teal": rgb_to_u([0, 153, 136]),
            "magenta": rgb_to_u([238, 51, 119]),
            "grey": rgb_to_u([187, 187, 187]),
        },
        "Bright": {
            "blue": rgb_to_u([68, 119, 170]),
            "cyan": rgb_to_u([102, 204, 238]),
            "green": rgb_to_u([34, 136, 51]),
            "yellow": rgb_to_u([204, 187, 68]),
            "red": rgb_to_u([238, 102, 119]),
            "purple": rgb_to_u([170, 51, 119]),
            "gray": rgb_to_u([187, 187, 187]),
        },
        "RGB": {
            "blue": rgb_to_u([25, 101, 176]),
            "green": rgb_to_u([78, 178, 101]),
            "red": rgb_to_u([220, 5, 12]),
        }
    }
}

grays = {
    "light": [.8, .8, .8],
    "medium": [0.7, 0.7, 0.7],
    "dark": [.6, .6, .6],
}

color_sets = [
    ["PT", "Vibrant", "blue", "orange", "cyan"],
    ["PT", "Vibrant", "cyan", "red", "teal"],
    ["PT", "High-contrast", "blue", "red", "yellow"],
    ["PT", "Bright", "blue", "red", "yellow"],
    ["PT", "RGB", "blue", "red", "green"],
]

csid = 2
colors = {
    "GD": colorschemes[color_sets[csid][0]][color_sets[csid][1]][color_sets[csid][2]],
    "EFGD": colorschemes[color_sets[csid][0]][color_sets[csid][1]][color_sets[csid][3]],
    "EF": colorschemes[color_sets[csid][0]][color_sets[csid][1]][color_sets[csid][3]],
    "NGD": colorschemes[color_sets[csid][0]][color_sets[csid][1]][color_sets[csid][4]],
}
colors_classes = [
    colorschemes["PT"]["High-contrast"]["blue"], colorschemes["PT"]["High-contrast"]["yellow"]
#    colorschemes["PT"]["Vibrant"]["teal"], colorschemes["PT"]["Vibrant"]["red"]
]

linestyles = {
    "GD": "--",
    "EFGD": "-.",
    "EF": "-.",
    "NGD": "-",
}


def annotate_arrow(ax, text, start, end, color, arrowprop=None, rad=0):
    if arrowprop is None:
        arrowprop = {
            "width": 3,
            "headlength": 8,
            "headwidth": 8,
            "color": color,
            "connectionstyle": "arc3,rad=" + str(float(rad)),
        }
    ax.annotate(text, start, end, arrowprops=arrowprop, color=color, fontweight="bold")


def hide_frame(axes, top=True, right=True, left=False, bottom=False):
    for ax in axes:
        ax.spines['top'].set_visible(not top)
        ax.spines['right'].set_visible(not right)
        ax.spines['left'].set_visible(not left)
        ax.spines['bottom'].set_visible(not bottom)


def hide_ticks(axes):
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])


def hide_labels(axes):
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")


def save(fig, name):
    fig.savefig(name, bbox_inches='tight', transparent=True)


def strip_axes(axes):
    hide_ticks(axes)
    hide_labels(axes)


def ellipsis(scaling, var, mu=None, DO_SQRT=True):
    CIRCLEGRID = np.linspace(0, 1, 100) * 2 * np.pi

    if mu is None:
        mu = np.zeros((var.shape[0], 1))
    if DO_SQRT:
        var = scaling * sp.linalg.sqrtm(np.linalg.inv(var))
    else:
        var = scaling * np.linalg.inv(var)

    xs = var[0, 0] * np.sin(CIRCLEGRID) + var[0, 1] * np.cos(CIRCLEGRID) + mu[0]
    ys = var[1, 1] * np.cos(CIRCLEGRID) + var[1, 0] * np.sin(CIRCLEGRID) + mu[1]
    return xs, ys
