import numpy as np

class RewardMaze():
    def __init__(self, num_actions, len_shortest_path, grid) -> None:
        self.num_actions = num_actions
        self.grid = grid
        self.size = len(self.grid)
        self.len_shortest_path = len_shortest_path
        self.reward_maze = np.zeros((len(self.grid) ** 2, self.num_actions))

    def make_r_maze(self):
        # Action map
        #         1 
        #         ^
        #         |
        #    0 <–   -> 2
        #         |
        #         <
        #         3
        # All that the proceeding chunk of code does it make it so that every time the agent takes a step,
        # their reward is -1. However, if they take a step into a wall or out of bounds, their reward is -2.
        # Reward here is sparse, with the end being the only place that gets a positive reward.
        # The reward is stored in an array of states x actions, and indexing into it gives the reward when an
        # action is taken from a state (reward_maze[state][action])
        actions = [0, 1, 2, 3]
        for i in range(len(self.grid)):
            for j in range(len(self.grid)):
                state = self.size * i + j
                for a in actions:
                    if a == 0:
                        # If out of bounds of there is a wall
                        if j == 0 or self.grid[i][j - 1] == 0:
                            self.reward_maze[state][a] = -2
                        self.reward_maze[state][a] = -1
                    if a == 1:
                        if i == 0 or self.grid[i - 1][j] == 0:
                            self.reward_maze[state][a] = -2
                        self.reward_maze[state][a] = -1
                    if a == 2:
                        if j == self.size - 1 or self.grid[i][j + 1] == 0:
                            self.reward_maze[state][a] = -2
                        self.reward_maze[state][a] = -1
                    if a == 3:
                        if i == self.size - 1 or self.grid[i + 1][j] == 0:
                            self.reward_maze[state][a] = -2
                        self.reward_maze[state][a] = -1

        # Moving into the goal cell gives a positive reward of the length of the shortest path +1.
        self.reward_maze[self.size**2 - 1 - 1][2] = self.len_shortest_path + 1
        self.reward_maze[self.size**2 - self.size - 1][3] = self.len_shortest_path + 1
