import sys
import numpy as np
import pandas
import random
import time
from tqdm import tqdm

class Q_():
    # Seems like a lot of hyperparameters
    def __init__(self, 
                 num_states: int, 
                 num_actions: int, 
                 gamma: float, 
                 alpha: float, 
                 epsilon: float, 
                 num_episodes: int, 
                 maximum_steps: int,
                 size: int,
                 grid: np.ndarray,
                 reward_grid: np.ndarray):
        self.Q_table = np.zeros((num_states, num_actions))
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.maximum_steps = maximum_steps
        self.grid = grid
        self.reward_grid = reward_grid
        self.size = size

    
    def train(self):

        def get_state_given_action(state, action):
            #         1 
            #         ^
            #         |
            #    0 <–   -> 2
            #         |
            #         <
            #         3
            # If there's a wall, and the agent tries to walk into it, put them in the same spot
            # Otherwise, update their location
            loc_x, loc_y = state // self.size, state % self.size

            if action == 0:
                 if loc_y == 0 or self.grid[loc_x][loc_y - 1] == 0:
                     return state
                 return state - 1
            if action == 1:
                if loc_x == 0 or self.grid[loc_x - 1][loc_y] == 0:
                    return state
                return state - self.size
            if action == 2:
                if loc_y == self.size - 1 or self.grid[loc_x][loc_y + 1] == 0:
                    return state
                return state + 1
            if action == 3:
                if loc_x == self.size - 1 or self.grid[loc_x + 1][loc_y] == 0:
                    return state
                return state + self.size

        # Q-learning algorithm
        actions = [0, 1, 2, 3]
        for _ in tqdm(range(self.num_episodes)):
            cur_state = 0
            for _ in range(self.maximum_steps):
                random_number = random.uniform(0, 1)
                if random_number < self.epsilon:
                    action = random.choice(actions)
                else:
                    action = np.argmax(self.Q_table[cur_state])
                new_state = get_state_given_action(cur_state, action)

                self.Q_table[cur_state][action] += self.alpha * (self.reward_grid[cur_state][action] + self.gamma * max(self.Q_table[new_state]) - self.Q_table[cur_state][action])

                if new_state == (self.size**2) - 1:
                    break
                
                cur_state = new_state
                
        print(self.Q_table)

    def eval(self):
        pass

def main():
    pass
if __name__ == '__main__':
    main()