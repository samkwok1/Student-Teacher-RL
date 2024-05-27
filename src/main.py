import hydra
from omegaconf import DictConfig, OmegaConf
from environment import maze, reward_maze
from algo import q
from util import plots
import numpy as np

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

    # Init Parent agent
    max_steps = max(Maze.path_lengths)
    Q_hyper = args.Q_hyper
    Parent_Q = q.Q_agent(num_states=Maze_args.size**2,
                         num_actions=Maze_args.num_actions,
                         gamma=Q_hyper.gamma,
                         alpha=Q_hyper.alpha,
                         epsilon=Q_hyper.epsilon,
                         num_episodes=Q_hyper.num_episodes,
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
                         verbose=verbose)
    
    Parent_Q.train()
    Parent_Q.eval()

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