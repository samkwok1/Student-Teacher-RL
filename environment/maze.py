# ref: https://github.com/ZJLAB-AMMI/LLM4Teach/tree/main/env
# ref: https://medium.com/practical-coders-chronicles/conquering-openais-minigrid-a-comprehensive-guide-to-mastering-gridworld-in-python-bfe4f2a76c2e

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


# creates the new maze environment with class inherited from MiniGridEnv
class MazeEnv(MiniGridEnv):
    def __init__(
        self,
        size=8,
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

    @staticmethod
    # define the mission sapce
    def _gen_mission():
        return "grand mission"
    
    
    # TODO rewrite reward to overwrite the default reward function
    def _reward(self):
        return 0
    
def main():
    env = MazeEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

    
if __name__ == "__main__":
    main()