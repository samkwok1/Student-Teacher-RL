# ref: https://github.com/ZJLAB-AMMI/LLM4Teach/tree/main/env
# ref: https://medium.com/practical-coders-chronicles/conquering-openais-minigrid-a-comprehensive-guide-to-mastering-gridworld-in-python-bfe4f2a76c2e

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
import random

# creates the new maze environment with class inherited from MiniGridEnv
class MazeEnv(MiniGridEnv):
    def __init__(
        self,
        random_seed,
        size,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=256,
            **kwargs,
        )
        self.random_seed = random_seed

    @staticmethod
    # define the mission sapce
    def _gen_mission():
        return "grand mission"
    
    
    # TODO rewrite reward to overwrite the default reward function
    def _reward(self):
        return 0
    
    # Use Prim's algorithm to generate the maze
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        if self.random_seed is not None:
            random.seed(self.random_seed)

        



    
def main():
    env = MazeEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

    
if __name__ == "__main__":
    main()