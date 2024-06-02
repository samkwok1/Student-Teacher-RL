import sys
import numpy as np
import pandas
import random
import time
import pdb
from tqdm import tqdm
from scipy import special
from scipy.stats import entropy

class Q_agent():
    # Seems like a lot of hyperparameters
    def __init__(self, 
                 num_states: int, 
                 num_actions: int, 
                 gamma: float, 
                 alpha: float, 
                 epsilon: float, 
                 num_episodes: int, 
                 num_eval_episodes: int,
                 maximum_steps: int,
                 parent: bool,
                 parent_Q_table: np.ndarray,
                 child: bool,
                 pre_advice: bool,
                 pre_advice_epsilon: int,
                 post_advice: bool,
                 post_advice_weight: float,
                 reliability: int,
                 size: int,
                 grid: np.ndarray,
                 reward_grid: np.ndarray,
                 verbose: bool,
                 shortest_path_length: float, # when initializing the agent do "shortest_path_length = maze_instance.calculate_shortest_path_length"
                 convergence_threshold: float,
                 min_convergence_steps: int
                 ):
        # Hyperparameters related to the Q update rule (you can find it online)
        self.Q_table = np.zeros((num_states, num_actions))
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.num_eval_episodes = num_eval_episodes
        self.maximum_steps = maximum_steps

        # The grid, reward grid, and the size of the grid
        self.grid = grid
        self.reward_grid = reward_grid
        self.size = size

        # Whether this agent is a parent, the Q_table of the parent, and whether this agent is a child
        self.parent = parent
        self.parent_Q_table = parent_Q_table
        self.child = child

        # Whether we are using the "pre-advice" strategy for the child and when to take the parent's advice
        self.pre_advice = pre_advice
        self.pre_advice_epsilon = pre_advice_epsilon

        # Whether we are using the "post-advice" strategy for the child and how much to weight post advice
        self.post_advice = post_advice
        self.post_advice_weight = post_advice_weight

        # How reliable the agent is, if it's a parent agent
        self.reliability=reliability
        # Note that it may seem like we have many more hyperparameters that necessary. For instance, if we're
        # training a parent, why do we need to distinguish whether the parent is pre_advice or post_advice? 
        # Doesn't this only apply to the agent if the agent is a child? This is a design choice just to make them all
        # one class, if we're training a parent we won't actually use the child hyperparameters, and vice versa.
        # When these unused values are passed in in main.py, they're just garbage values.
        self.shortest_path_length = shortest_path_length
        self.verbose = verbose

        self.convergence_threshold = convergence_threshold
        self.min_convergence_steps = min_convergence_steps
        self.convergence_steps = None
        self.old_q_table = None


    def get_state_given_action(self, state, action):
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

    def get_action(self, cur_state):
        # Gets the actions according to the agent's properties.
        actions = [0, 1, 2, 3]
        random_number = random.uniform(0, 1)
        if random_number < self.epsilon:
            normal_action = random.choice(actions)
        else:
            normal_action = np.argmax(self.Q_table[cur_state])
        # If the agent is a parent or its a child that gets post-action advice, have it choose an action the
        # normal way, as defined by epsilon greedy.
        if self.parent or (self.post_advice and self.child):
            action = normal_action
        # If the agent is a pre-advice child...
        elif self.child and self.pre_advice:
            # ...have it choose an action if it is more than pre_advice_epsilon% unsure about
            # which action to choose (if it doesn't have a Q value that holds more than 50% of the probability
            # mass for its current state) then have it choose the action the parent recommends. Otherwise, 
            # have it choose normally (as according to epsilon greedy)
            dist = special.softmax(self.Q_table[cur_state])
            if max(dist) < self.pre_advice_epsilon:
                action = np.argmax(self.parent_Q_table[cur_state])
            # Have it choose normally
            else:
                action = normal_action
        return action

    # if the agent is a parent, see whether the policy actually is optimal (returns True if optimal)
    def is_policy_optimal(self):
        if not self.parent:
            return False

        cur_state = 0
        path_length = 0
        visited_states = set()

        while cur_state != (self.size**2 - 1):
            action = np.argmax(self.Q_table[cur_state])
            new_state = self.get_state_given_action(cur_state, action)
            if new_state in visited_states:
                return False
            visited_states.add(new_state)
            path_length += 1
            cur_state = new_state
            if path_length > self.shortest_path_length:
                return False
        
        return path_length + 1 == self.shortest_path_length
     
    def scramble_policy(self, reliability):
        if not self.parent:
            return  # Only scramble if the agent is a parent

        num_states_to_scramble = int((1 - reliability) * self.Q_table.shape[0])
        states_to_scramble = random.sample(range(self.Q_table.shape[0]), num_states_to_scramble)

        for state in states_to_scramble:
            optimal_action = np.argmax(self.Q_table[state])
            possible_actions = list(range(self.Q_table.shape[1]))
            possible_actions.remove(optimal_action)  # Remove the optimal action

            # Choose a new action randomly from the remaining possible actions
            new_action = random.choice(possible_actions)
            current_optimal_value = self.Q_table[state][optimal_action]

            # Reduce the Q-value of the current optimal action
            self.Q_table[state][optimal_action] *= 0.9  # Reduce slightly to maintain some original learning

            # Increase the Q-value of the new action to make it the new optimal
            self.Q_table[state][new_action] = current_optimal_value * 1.1  # Boost to ensure it's the new max


    def train(self):


        # def has_converged(Q_table_old, Q_table_new):
        #     return np.max(np.abs(Q_table_old - Q_table_new)) < self.convergence_threshold

        # Q_table_old = np.copy(self.Q_table)
        # steps_to_converge = 0   

        # TODO here - if the agent is a parent, scramble the "optimal policy" in its Q-table
        # the logic here is that we calcuate the number of states to be scrambled based on the reliability score, then randomly select the states 
        # to be scrambled. For each state, we take out the optimal action and change it to one of the sub-optimal actions (again by choosing randomly)
        
        total_steps = 0
        # Traditional Q-learning algorithm
        for ep in tqdm(range(self.num_episodes)):
            cur_state = 0
            for step in range(self.maximum_steps):
                action = self.get_action(cur_state)
                new_state = self.get_state_given_action(cur_state, action)
                reward = self.reward_grid[cur_state][action]
                # If the agent is a post-action advice child, then incorporate the KL divergence term to weigh its action probability distribution
                # against the parent's probability distribution for the previous state, not the newly picked state.
                if self.child and self.post_advice:
                    reward -= self.post_advice_weight * entropy(special.softmax(self.Q_table[cur_state]), special.softmax(self.parent_Q_table[cur_state]))

                self.Q_table[cur_state][action] += self.alpha * (reward + self.gamma * max(self.Q_table[new_state]) - self.Q_table[cur_state][action])
                
                # If the agent has made it to the end, break
                if new_state == (self.size**2) - 1:
                    break
                # Move the agent through the maze.
                cur_state = new_state

                total_steps += 1
                if self.is_policy_optimal():
                    break
            


            # Q_table_old = np.copy(self.Q_table)

        self.convergence_steps = total_steps

                
        # TODO here - eval whether current Q_table is optimal, if it is, record the number of steps it took to converge, store that in a parameters
        # Eli (done above, 6/1/24, 1:33pm)
       
        if self.verbose:
            print(f"Q_table {self.Q_table.shape}:")
            print(self.Q_table)
            print(f"Converged in {self.convergence_steps} steps")

        assert self.is_policy_optimal() == True

        # print('Whether the policy is optimal: ', self.is_policy_optimal())
        # Save the "optimal q table"
        self.old_q_table = self.Q_table



        # Maybe TODO - Eval using distance
    

        # TODO - plot the graphs
        # Sam
        

def main():
    pass
if __name__ == '__main__':
    main()