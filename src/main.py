import hydra
from omegaconf import DictConfig, OmegaConf
from environment import maze, reward_maze
from algo import q
from util import plots
import numpy as np
from tqdm import tqdm
import pdb 
import random
import json

RANDOM_SEEDS = {
    9:5,
    10:3,
    25:3043,
}

def find_maze(maze_args, verbose):
    # Init Maze
    random_seed = maze_args.random_seed

    num_paths = 0

    print(f"Searching for a compatible maze of size {maze_args.size}")
    while num_paths < 2:
        Maze = maze.Maze(size=maze_args.size,
                         random_seed=random_seed,
                         verbose=verbose)
        num_paths = Maze.num_paths
        random_seed += 1

    print("Found a compatible Maze!")

    if maze_args.size not in RANDOM_SEEDS:
        print(f"Sucessful random seed: {random_seed - 1}")
        print("Please input the successful random seed as an entry in RANDOM_SEEDS (found at the top of the file) as such... MAZE_SIZE:RANDOM_SEEDS")

    print(f"Number of paths: {Maze.num_paths}")
    print(f"Length of each path: {Maze.path_lengths}")
    return Maze

@hydra.main(config_path="config", config_name="config")
def main(args: DictConfig) -> None:
    verbose = args.sim.verbose
    Maze_args = args.Maze

    Maze = find_maze(maze_args=Maze_args, verbose=verbose)
    # Init Reward Maze
    min_steps = min(Maze.path_lengths)
    Reward_maze = reward_maze.RewardMaze(num_actions=Maze_args.num_actions,
                                         len_shortest_path=min_steps,
                                         grid=Maze.Grid)
    Reward_maze.make_r_maze()

    # Ideally, we should have two metrics - exploration and exploitations
    # Init Parent agent
    max_steps = max(Maze.path_lengths)
    Q_hyper = args.Q_hyper
    num_steps_to_converge = 0
    zero_to_hundred = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    gamma_dict, alpha_dict, epsilon_dict = {}, {}, {}
    for g in zero_to_hundred:
        for a in zero_to_hundred:
            for e in zero_to_hundred:
                num_steps_to_converge, total_convergences = 0, 0
                for _ in range(15):
                    Parent_Q = q.Q_agent(num_states=Maze_args.size**2,
                                            num_actions=Maze_args.num_actions,
                                            gamma=g,
                                            alpha=a,
                                            epsilon=e,
                                            num_episodes=Q_hyper.num_oracle_episodes,
                                            maximum_steps=max_steps,
                                            parent=True,
                                            parent_Q_table=np.zeros((1, 1)),
                                            child=False,
                                            pre_advice=False,
                                            pre_advice_epsilon=0,
                                            post_advice=False,
                                            post_advice_weight=0,
                                            size=Maze_args.size,
                                            grid=Maze.Grid,
                                            reward_grid=Reward_maze.reward_maze,
                                            verbose=verbose,
                                            shortest_path_length=min_steps)
                    Parent_Q.train()
                    if Parent_Q.convergence_steps != None:
                        num_steps_to_converge += Parent_Q.convergence_steps
                        total_convergences += 1
                    else:
                        num_steps_to_converge += 0
                    

                    
    # Init for results -  a nested dict with pre/post - reliability - parameter
    num_steps_to_converge =  {'pre_advice': {}, 'post_advice': {}}

    # Init parameters
    parent_reliabilities=[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    pre_advice_epsilons=[0]
    post_advice_weights=[0.1, 0.2, 0.3]
    num_trials = 10

    for reliability in tqdm(parent_reliabilities):
        # pre advice mode - varying epsilon
        num_steps_to_converge['pre_advice'][reliability] = dict()
        for pre_advice_epsilon in pre_advice_epsilons:
            trial_steps = []
            for _ in range(num_trials):
                # "Optimal policy is kept track of"
                Parent_Q.Q_table = Parent_Q.Q_optimal
                Parent_Q.scramble_policy(reliability=reliability)
                # assert Parent_Q.Q_optimal.all() != Parent_Q.Q_table.all()

                # initialize a child 
                Child_pre = q.Q_agent(num_states=Maze_args.size**2,
                        num_actions=Maze_args.num_actions,
                        gamma=Q_hyper.gamma,
                        alpha=Q_hyper.alpha,
                        epsilon=Q_hyper.epsilon,
                        num_episodes=Q_hyper.num_episodes,
                        maximum_steps=max_steps,
                        parent=False,
                        parent_Q_table=Parent_Q.Q_table,
                        child=True,
                        pre_advice=True,
                        pre_advice_epsilon=pre_advice_epsilon,
                        post_advice=False,
                        post_advice_weight=None,
                        size=Maze_args.size,
                        grid=Maze.Grid,
                        reward_grid=Reward_maze.reward_maze,
                        verbose=verbose,
                        shortest_path_length=min_steps)
                # child class - train the child on the randomly scrambled q_table
                Child_pre.train()
                trial_steps.append(Child_pre.convergence_steps)
            # (reliability, pre_advice epsilon)
            num_steps_to_converge['pre_advice'][reliability][pre_advice_epsilon] = trial_steps

        # post advice mode - varying weight
        num_steps_to_converge['post_advice'][reliability] = dict()
        for post_advice_weight in post_advice_weights:
            trial_steps = []
            for _ in range(num_trials):
                # "Optimal policy is kept track of"
                Parent_Q.Q_table = Parent_Q.Q_optimal
                Parent_Q.scramble_policy(reliability=reliability)
                # initialize a child 
                Child_post = q.Q_agent(num_states=Maze_args.size**2,
                        num_actions=Maze_args.num_actions,
                        gamma=Q_hyper.gamma,
                        alpha=Q_hyper.alpha,
                        epsilon=Q_hyper.epsilon,
                        num_episodes=Q_hyper.num_episodes,
                        maximum_steps=max_steps,
                        parent=False,
                        parent_Q_table=Parent_Q.Q_table,
                        child=True,
                        pre_advice=False,
                        pre_advice_epsilon=None,
                        post_advice=True,
                        post_advice_weight=post_advice_weight,
                        size=Maze_args.size,
                        grid=Maze.Grid,
                        reward_grid=Reward_maze.reward_maze,
                        verbose=verbose,
                        shortest_path_length=min_steps)
                # child class - train the child on the randomly scrambled q_table
                Child_post.train()
                trial_steps.append(Child_post.convergence_steps)
                plots.plot_grid(Maze.Grid, True, Child_post.Q_table)
            
            # (reliability, pre_advice epsilon)
            num_steps_to_converge['post_advice'][reliability][post_advice_weight] = trial_steps

    with open('res.json', 'w') as f: 
        json.dump(num_steps_to_converge, f, indent=4)

    plots.plot_grid(grid=Maze.Grid)

if __name__ == "__main__":
    main()