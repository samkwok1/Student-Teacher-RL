from typing import (
    List,
    Tuple,
)
import os
import pandas as pd 
import numpy as np
import json

import colorsys
import seaborn as sns
from matplotlib.axes import Axes
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker
from sentence_transformers import SentenceTransformer, util
from scipy.stats import sem
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import json

def plot_grid(grid: np.ndarray, policy: bool, q_table: np.ndarray):
    n = grid.shape[0]
    fig, ax = plt.subplots()
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)

    # Draw the grid lines
    for i in range(n + 1):
        ax.axhline(i, color='gray', linewidth=0.5)
        ax.axvline(i, color='gray', linewidth=0.5)

    palette = sns.color_palette("viridis", n_colors=2)
    scale = 0.25
    arrows = {2: (1,0), 0: (-1, 0), 1: (0, 1), 3: (0, -1)}
    
    for i in range(n):
        for j in range(n):
            color = palette[1] if grid[i, j] == 1 else palette[0]
            square = patches.Rectangle((j, n-i-1), 1, 1, linewidth=1, edgecolor='none', facecolor=color)
            ax.add_patch(square)
            state = i * n + j
            if policy and grid[i, j] == 1 and not all(x == 0 for x in q_table[state]) :
                chosen_action = np.argmax(q_table[state])
                
                # Arrow positioning: center of the square
                # Adjusted arrow coordinates to point from the center of each square
                center_x, center_y = j + 0.5, n - i - 0.5
                dx, dy = scale * arrows[chosen_action][0], scale * arrows[chosen_action][1]
                ax.arrow(center_x, center_y, dx, dy, head_width=0.1, head_length=0.1, fc='black', ec='black')

    # Customize the plot
    ax.set_aspect('equal')
    ax.grid(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.title('Q-Learning Generated Policy')
    plt.show()

def lighten_color(
    color, 
    amount=0.5, 
    desaturation=0.2,
) -> Tuple[float, float, float]:
    """
    Copy-pasted from Eric's slack.
    Lightens and desaturates the given color by multiplying (1-luminosity) by the given amount
    and decreasing the saturation by the specified desaturation amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    Examples:
    >> lighten_color('g', 0.3, 0.2)
    >> lighten_color('#F034A3', 0.6, 0.4)
    >> lighten_color((.3,.55,.1), 0.5, 0.1)
    """
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    h, l, s = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(h, 1 - amount * (1 - l), max(0, s - desaturation))


def plot_policy_graph(
                   linewidth: int = 2,
                   zorder: int = 1,
                   scatter_color: str = 'black'):
    file_path = 'experiments/res.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

    x = ["1", "0.9"]
    pre_advice_epsilon = "0.2"
    post_advice_weight = "0.1"
    pre_advice_dict = data["pre_advice"]
    post_advice_dict = data["post_advice"]
    errors = []
    values = []
    for reliability in x:
        pre_advice_epsilon_dict = post_advice_dict[reliability]
        y_vals = pre_advice_epsilon_dict[post_advice_weight]
        errors.append(sem(y_vals))
        values.append(np.mean(y_vals))
    x = [1, 0.9]
    _, ax = plt.subplots(figsize=(20, 10))
    palette = sns.color_palette("mako", 2)
    x_fin = [f"{int(x_val * 100)}%" for x_val in x]
    line = ax.plot(x_fin, values, color=palette[0], linewidth=linewidth, zorder=zorder)
    ax.scatter(x_fin, values, color=[lighten_color(scatter_color)] * len(x)) 
    ax.fill_between(x_fin, [y_val - 1.95 * err if y_val is not np.nan else 0 for y_val, err in zip(values, errors)], [y_val + 1.95 * err if y_val is not np.nan else 0 for y_val, err in zip(values, errors)], color=palette[0], alpha=0.3)
    plot_proposal_line_graph(ax=ax,
                            x=x_fin,
                            graph_title=f"Parent Policy",
                            xlabel='Parent Reliability',
                            ylabel='Number of Convergence Steps',
                            directory="plots"
                            )

def main():
    plot_policy_graph()

if __name__ == "__main__":
    main()