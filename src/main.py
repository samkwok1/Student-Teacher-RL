import hydra
from omegaconf import DictConfig, OmegaConf
from environment import maze, reward_maze
from algo import q
from util import plots
import numpy as np
from tqdm import tqdm

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

def eval(self):

    # TODO here - Search hyperparameter space for pre-advice and post-advice child stuff
    # TODO implement evaluation 
    # https://colab.research.google.com/drive/1Ur_pYvL_IngmAttMBSZlBRwMNnpzQuY_#scrollTo=KASNViqL4tZn
    # evaluate on reward over n number of episodes by actioning on the learned Q-table
    rewards = []
    is_optimal_policies = []
    for _ in tqdm(range(self.num_eval_episodes)):
        cur_state = 0  # Reset environment to initial state for each episode
        episode_reward = 0
        for _ in range(self.maximum_steps):
            # Take the action (index) that have the maximum reward
            action = np.argmax(self.Q_table[cur_state])
            new_state = self.get_state_given_action(cur_state, action)
            reward = self.reward_grid[cur_state][action]
            episode_reward += reward
            
            if new_state == (self.size**2) - 1:
                break
            cur_state = new_state

        rewards.append(episode_reward)
        is_optimal_policies.append(self.is_policy_optimal())

    print('Mean rewards: ', sum(rewards)/len(rewards))
    print('Is optimal policy: ', sum(is_optimal_policies)/len(is_optimal_policies))

    return 

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

    # Init Parent agent
    max_steps = max(Maze.path_lengths)
    Q_hyper = args.Q_hyper
    Parent_Q = q.Q_agent(num_states=Maze_args.size**2,
                         num_actions=Maze_args.num_actions,
                         gamma=Q_hyper.gamma,
                         alpha=Q_hyper.alpha,
                         epsilon=Q_hyper.epsilon,
                         num_episodes=Q_hyper.num_episodes,
                         num_eval_episodes=Q_hyper.num_eval_episodes,
                         maximum_steps=max_steps,
                         parent=True,
                         parent_Q_table=np.zeros((1, 1)),
                         child=False,
                         pre_advice=False,
                         pre_advice_epsilon=0,
                         post_advice=False,
                         post_advice_weight=0,
                         reliability=Q_hyper.parent_reliability,
                         size=Maze_args.size,
                         grid=Maze.Grid,
                         reward_grid=Reward_maze.reward_maze,
                         verbose=verbose,
                         shortest_path_length=min_steps,
                         convergence_threshold=0.001,
                         min_convergence_steps=1)
    
    Parent_Q.train()

    for trial in range(num_trials):
        

    plots.plot_grid(grid=Maze.Grid)
    # # Init Child agent - PRE-advice
    # max_steps = max(Maze.path_lengths)
    # Q_hyper = args.Q_hyper
    # Child_params = args.Child_params
    # Child_pre = q.Q_agent(num_states=Maze_args.size**2,
    #                      num_actions=Maze_args.num_actions,
    #                      gamma=Q_hyper.gamma,
    #                      alpha=Q_hyper.alpha,
    #                      epsilon=Q_hyper.epsilon,
    #                      num_episodes=Q_hyper.num_episodes,
    #                      maximum_steps=max_steps,
    #                      parent=False,
    #                      parent_Q_table=Parent_Q.Q_table,
    #                      child=True,
    #                      pre_advice=True,
    #                      pre_advice_epsilon=Child_params.pre_advice_epsilon,
    #                      post_advice=False,
    #                      post_advice_weight=Child_params.post_advice_weight,
    #                      reliability=Q_hyper.parent_reliability,
    #                      size=Maze_args.size,
    #                      grid=Maze.Grid,
    #                      reward_grid=Reward_maze.reward_maze)
    # Child_pre.train()
    # Child_pre.eval()
    
    # # Init Child agent - POST-advice
    # max_steps = max(Maze.path_lengths)
    # Q_hyper = args.Q_hyper
    # Child_post = q.Q_agent(num_states=Maze_args.size**2,
    #                      num_actions=Maze_args.num_actions,
    #                      gamma=Q_hyper.gamma,
    #                      alpha=Q_hyper.alpha,
    #                      epsilon=Q_hyper.epsilon,
    #                      num_episodes=Q_hyper.num_episodes,
    #                      maximum_steps=max_steps,
    #                      parent=False,
    #                      parent_Q_table=Parent_Q.Q_table,
    #                      child=True,
    #                      pre_advice=False,
    #                      pre_advice_epsilon=Child_params.pre_advice_epsilon,
    #                      post_advice=True,
    #                      post_advice_weight=Child_params.post_advice_weight,
    #                      reliability=Q_hyper.parent_reliability,
    #                      size=Maze_args.size,
    #                      grid=Maze.Grid,
    #                      reward_grid=Reward_maze.reward_maze)
    # Child_post.train()
    # Child_post.eval()

if __name__ == "__main__":
    main()