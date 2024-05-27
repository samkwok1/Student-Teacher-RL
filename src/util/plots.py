import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns

def plot_grid(grid):
    n = grid.shape[0]
    fig, ax = plt.subplots()
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)

    # Draw the grid lines
    for i in range(n + 1):
        ax.axhline(i, color='gray', linewidth=0.5)
        ax.axvline(i, color='gray', linewidth=0.5)

    palette = sns.color_palette("viridis", n_colors=2)
    for i in range(n):
        for j in range(n):
            color = palette[1] if grid[i][j] == 1 else palette[0]
            square = patches.Rectangle((j, n-i-1), 1, 1, linewidth=1, edgecolor='none', facecolor=color)
            ax.add_patch(square)

    # Customize the plot
    ax.set_aspect('equal')
    ax.grid(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


    plt.show()



def plot_policy(q_table, agent_name):
    pass


def main():
    plot_grid()

if __name__ == "__main__":
    main()