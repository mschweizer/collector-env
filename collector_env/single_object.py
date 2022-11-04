from gym.spaces import Discrete
from gym_minigrid.minigrid import MiniGridEnv, Grid

from collector_env import ValuedBall


class SingleObjectEnv(MiniGridEnv):

    def __init__(self, size=7, agent_start_pos=(1, 1), agent_start_dir=0, collection_reward: float = 1.0,
                 max_steps=None, **kwargs):

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.collection_reward = collection_reward

        super().__init__(grid_size=size, max_steps=max_steps, see_through_walls=True, **kwargs)

        # Only allow the 4 used actions
        self.action_space = Discrete(4)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Place a blue ball at a random position on the grid
        self.objects = [
            ValuedBall("blue", value=self.collection_reward)
        ]

        for obj in self.objects:
            self.place_obj(obj)

        self.mission = "Collect as many items as possible."

    def step(self, action):
        observation, reward, terminated, info = super().step(action)

        if self.carrying:
            reward = self.carrying.value
            self._replace_item()
        else:
            reward = 0.0

        return observation, reward, terminated, info

    def _replace_item(self):
        picked_up_item = self.carrying
        self.carrying = None
        self.place_obj(picked_up_item)


class SingleObjectEnv5x5(SingleObjectEnv):
    def __init__(self, collection_reward: float = 1., max_steps=200):
        super().__init__(size=5, max_steps=max_steps, collection_reward=collection_reward)
