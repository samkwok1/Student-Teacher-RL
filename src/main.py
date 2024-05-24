import hydra
from omegaconf import DictConfig, OmegaConf
from environment import maze, reward_maze
from algo import q
import numpy as np

@hydra.main(config_path="config", config_name="config")
def main(args: DictConfig) -> None:
    # Init Maze
    Maze_args = args.Maze
    Maze = maze.Maze(size=Maze_args.size,
                     random_seed=Maze_args.random_seed,
                     modifications=Maze_args.modifications)
    assert Maze.num_paths >= 1

    # Init Reward Maze
    min_steps = min(Maze.path_lengths)
    Reward_maze = reward_maze.RewardMaze(num_actions=Maze_args.num_actions,
                                         len_shortest_path=min_steps,
                                         grid=Maze.Grid)
    Reward_maze.make_r_maze()

    # Init Parent agent
    max_steps = max(Maze.path_lengths) + 1000
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
                         reward_grid=Reward_maze.reward_maze)
    
    Parent_Q.train()
    # Parent_Q.eval()


    # # Init Child agent - PRE-advice
    # max_steps = max(Maze.path_lengths) + 1000
    # Q_hyper = args.Q_hyper
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
    #                      pre_advice_epsilon=0.5,
    #                      post_advice=False,
    #                      post_advice_weight=0,
    #                      reliability=Q_hyper.parent_reliability,
    #                      size=Maze_args.size,
    #                      grid=Maze.Grid,
    #                      reward_grid=Reward_maze.reward_maze)
    # Child_pre.train()
    # Child_pre.eval()
    
    # # Init Child agent - POST-advice
    # max_steps = max(Maze.path_lengths) + 1000
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
    #                      pre_advice_epsilon=0,
    #                      post_advice=True,
    #                      post_advice_weight=1,
    #                      reliability=Q_hyper.parent_reliability,
    #                      size=Maze_args.size,
    #                      grid=Maze.Grid,
    #                      reward_grid=Reward_maze.reward_maze)
    # Child_post.train()
    # Child_post.eval()

if __name__ == "__main__":
    main()