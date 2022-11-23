from gym.spaces import Discrete
from gym.vector.utils import spaces
from gym_minigrid.minigrid import MiniGridEnv, Grid

from collector_env.actions import Actions
from collector_env.valued_objects import ValuedBall, ValuedKey


class CollectorEnv(MiniGridEnv):
    """
        ### Description

        A single agent lives in a gridworld environment that has two types of items, blue balls and green keys.
        At every point in time, exactly one instance of each item type is present in the environment.
        Initially, the agent is placed in the bottom left corner and the items are placed at random, distinct positions
        on the grid.
        The agent can pick up items.
        Whenever an item is collected, an item of the same type reappears at a random, distinct location.
        As long as an item is not collected, it will not change its position.
        The environment is fully observable and is perceived as symbolic representation.
        Except for item placement, the environment is deterministic.


        ### Mission Space

        "collect valuable items"

        ### Action Space

        | Num | Name         | Action               |
        |-----|--------------|----------------------|
        | 0   | left         | Turn left            |
        | 1   | right        | Turn right           |
        | 2   | forward      | Move forward         |
        | 3   | pickup       | Pick up an object    |

        ### Observation Encoding

        - Each tile is encoded as a 3 dimensional tuple:
            `(OBJECT_IDX, COLOR_IDX, STATE)`
        - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
            [minigrid/minigrid.py](minigrid/minigrid.py)
        - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

        ### Rewards

        There are two possible item values, by default -1 and 1.
        Whenever one type has value 1, the other has value -1.
        The value of the items switch in regular intervals, given by `value_update_interval`,
        in our example from 1 to -1 or vice versa.

        ### Termination

        The episode ends after a given number of steps (see `max_steps`).

        ### Registered Configurations

        - `Collector-5x5-v0`
        - `Collector-7x7-v0`

        """

    def __init__(self, size=7, agent_start_pos=(1, 1), agent_start_dir=0, positive_object_reward: float = 1.0,
                 negative_object_reward: float = -1.0, turn_reward=0., move_reward=0., bump_reward=0.,
                 value_update_interval=None, max_value_updates=None, max_steps=None, **kwargs):
        self.mission = None
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.positive_object_reward = positive_object_reward
        self.negative_object_reward = negative_object_reward
        self.key_reward = self.positive_object_reward
        self.ball_reward = self.negative_object_reward
        self.turn_reward = turn_reward
        self.move_reward = move_reward
        self.bump_reward = bump_reward
        self.value_update_interval = value_update_interval
        self.max_value_updates = max_value_updates
        self.value_update_count = 0

        max_steps = max_steps if max_steps else 4 * size * size

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs
        )
        self.actions = Actions
        self.action_space = Discrete(len(self.actions))

        # Only allow the 4 used actions
        self.action_space = Discrete(4)

        self.total_step_count = 0

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

        # Place two objects at random positions on the grid
        self.objects = [
            ValuedKey("green", value=self.key_reward),
            ValuedBall("blue", value=self.ball_reward)
        ]

        for obj in self.objects:
            self.place_obj(obj)

        self.mission = self._create_mission_statement()

    def _create_mission_statement(self):
        return "Current object valuations: {}, {}".format(self.objects[0], self.objects[1])

    def step(self, action):
        self.total_step_count += 1

        forward_cell = self.grid.get(*self.front_pos)

        observation, _, terminated, info = super().step(action)

        reward = self._compute_reward(action, forward_cell)

        if self.carrying:
            info["pickup"] = True
            self._replace_item()

        if self.value_update_interval:
            if (self.max_value_updates is None or self.value_update_count < self.max_value_updates) \
                    and self.total_step_count % self.value_update_interval == 0:
                self._switch_object_values()
                self.value_update_count += 1
                self.mission = self._create_mission_statement()

        return observation, reward, terminated, info

    def _compute_reward(self, action, forward_cell):
        if self.carrying:
            reward = self.carrying.value
        elif action in (self.actions.left, self.actions.right):
            reward = self.turn_reward
        elif action == self.actions.forward:
            if forward_cell is not None and not forward_cell.can_overlap():
                reward = self.bump_reward
            else:
                reward = self.move_reward
        else:
            reward = 0.0
        return reward

    def _replace_item(self):
        picked_up_item = self.carrying
        self.carrying = None
        self.place_obj(picked_up_item)

    def _switch_object_values(self):
        self.key_reward, self.ball_reward = self.ball_reward, self.key_reward
        self._update_object_values()

    def _update_object_values(self):
        for obj in self.objects:
            if isinstance(obj, ValuedBall):
                obj.value = self.ball_reward
            elif isinstance(obj, ValuedKey):
                obj.value = self.key_reward


class CollectorEnv7x7(CollectorEnv):
    def __init__(self, positive_object_reward: float = 1.0, negative_object_reward: float = -1.0, turn_reward=0.,
                 move_reward=0., bump_reward=0., max_steps=200, value_update_interval=None):
        super().__init__(size=7, positive_object_reward=positive_object_reward,
                         negative_object_reward=negative_object_reward, turn_reward=turn_reward,
                         move_reward=move_reward, bump_reward=bump_reward, value_update_interval=value_update_interval,
                         max_steps=max_steps)


class CollectorEnv5x5(CollectorEnv):
    def __init__(self, positive_object_reward: float = 1.0, negative_object_reward: float = -1.0, turn_reward=0.,
                 move_reward=0., bump_reward=0., max_steps=200, value_update_interval=None):
        super().__init__(size=5, positive_object_reward=positive_object_reward,
                         negative_object_reward=negative_object_reward, turn_reward=turn_reward,
                         move_reward=move_reward, bump_reward=bump_reward, value_update_interval=value_update_interval,
                         max_steps=max_steps)

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(5, 5, 3),
            dtype='uint8'
        )
