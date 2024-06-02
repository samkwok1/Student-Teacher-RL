import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns

# def plot_grid(grid: np.ndarray, 
#               policy: bool,
#               q_table: np.ndarray):
#     n = grid.shape[0]
#     fig, ax = plt.subplots()
#     ax.set_xlim(0, n)
#     ax.set_ylim(0, n)

#     # Draw the grid lines
#     for i in range(n + 1):
#         ax.axhline(i, color='gray', linewidth=0.5)
#         ax.axvline(i, color='gray', linewidth=0.5)

#     palette = sns.color_palette("viridis", n_colors=2)
#     scale = 0.25
#     arrows = {2: (1,0), 0: (-1, 0), 1: (0, 1), 3: (0, -1)}
#     for r, i in enumerate(range(n)):
#         for c, j in enumerate(range(n)):
#             color = palette[1] if grid[i][j] == 1 else palette[0]
#             square = patches.Rectangle((j, n-i-1), 1, 1, linewidth=1, edgecolor='none', facecolor=color)
#             ax.add_patch(square)
#             if policy:
#                 state = i * n + j
#                 chosen_action = np.argmax(q_table[state])
#                 plt.arrow(i, j, scale*arrows[chosen_action][0], scale*arrows[chosen_action][1], head_width = 0.1)


#     # Customize the plot
#     ax.set_aspect('equal')
#     ax.grid(False)
#     ax.xaxis.set_visible(False)
#     ax.yaxis.set_visible(False)
#     plt.title('Q-Learning Generated Policy')
#     plt.show()
#     plt.show()

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
            if policy:
                state = i * n + j
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


def plot_policy(q_table: np.ndarray, 
                agent_name: str, 
                policy: bool):
    n = q_table.shape[0]




def main():
    plot_grid()

if __name__ == "__main__":
    main()