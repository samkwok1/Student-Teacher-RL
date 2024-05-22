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
                 grid: np.ndarray,
                 reward_grid: np.ndarray):
        self.Q_table = np.zeros((num_states, num_actions))
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.maximum_steps = maximum_steps

    
    def train(self):

        def get_state_given_action(state, action):
            pass

        for _ in tqdm(range(self.num_episodes)):
            cur_state = 0
            for _ in range(self.maximum_steps):
                pass







    
    def eval(self):
        pass

def main():
    pass
if __name__ == '__main__':
    main()