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

def plot_proposal_line_graph(ax: Axes,
                             x: list,
                             graph_title: str,
                             xlabel: str,
                             ylabel: str,
                             directory: str,
                             font_family: str = 'Avenir',
                             font_size: int = 24,
                             y_label_coords: Tuple[float, float] = (-0.07, 0.5),
                             y_ticks: List[int] = [0, 4000, 8000, 12000, 16000, 20000],
                             y_ticklabels: List[str] = [0, '4k', '8k', '12k', '16k', '20k'],
                             y_lim: Tuple[float, float] = (-0.1, 20001),
                             legend: bool = False,
                             legend_title: str = 'Agent',
                             legend_loc: str = 'center left',
                             bbox_to_anchor: Tuple[float, float] = (1.0, 0.6),
                             ):

    plt.xlabel(xlabel, family=font_family, size=font_size)
    sns.despine(left=True, bottom=False)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, fontsize=font_size)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True)) 

    ax.set_ylabel(ylabel, family=font_family, size=font_size)
    ax.yaxis.set_label_coords(*y_label_coords)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels, size=font_size)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5, zorder=-100)
    plt.ylim(y_lim)
    plt.subplots_adjust(left=0.1, right=0.8)

    plt.title(" ".join(graph_title.split('_')), family=font_family, size=font_size + 5)
    if legend:
        ax.legend(title=legend_title, 
                  frameon=False,
                  ncol=1, 
                  bbox_to_anchor=bbox_to_anchor,
                  loc=legend_loc,
                  fontsize=font_size,  # Change the font size of legend items
                  title_fontsize=font_size
                  )
    plt.savefig(f'{directory}_{graph_title}.png', format='png')
    plt.clf()

def main():
    plot_policy_graph()

if __name__ == "__main__":
    main()