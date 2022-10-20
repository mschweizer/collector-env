import random

from gym_minigrid.minigrid import MiniGridEnv, Grid

from collector_env.valued_objects import ValuedBall, ValuedKey


class CollectorEnv(MiniGridEnv):
    """
        ### Description

        A single agent lives in a gridworld environment that has two types of items, blue and red balls.
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
        | 4   | drop         | Unused               |
        | 5   | toggle       | Unused               |
        | 6   | done         | Unused               |

        ### Observation Encoding

        - Each tile is encoded as a 3 dimensional tuple:
            `(OBJECT_IDX, COLOR_IDX, STATE)`
        - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
            [minigrid/minigrid.py](minigrid/minigrid.py)
        - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

        ### Rewards

        The value of blue and red balls is either -1 or 1.
        It switches in regular intervals given by `value_update_interval`.
        Whenever one type has value 1, the other has value -1.

        ### Termination

        The episode ends after a given number of steps (see `max_steps`).

        ### Registered Configurations

        - `Collector-v0`

        """

    def __init__(self, size=7, agent_start_pos=(1, 1), agent_start_dir=0, value_update_interval=None, max_steps=None,
                 **kwargs):
        self.mission = None
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.value_update_interval = value_update_interval

        max_steps = max_steps if max_steps else 4 * size * size

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs
        )

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

        # Place a red and blue ball at random positions on the grid
        self.objects = [ValuedKey("green"), ValuedBall("blue")]
        self._assign_initial_values()
        for obj in self.objects:
            self.place_obj(obj)

        self.mission = self._create_mission_statement()

    def _create_mission_statement(self):
        return "Current object valuations: {}, {}".format(self.objects[0], self.objects[1])

    def step(self, action):
        obs, reward, terminated, info = super().step(action)

        if self.carrying:
            reward = self.carrying.value
            self._replace_item()
        else:
            reward = 0.0

        if self.value_update_interval:
            if self.step_count % self.value_update_interval == 0:
                self._switch_object_values()
                self.mission = self._create_mission_statement()

        return obs, reward, terminated, info

    def _replace_item(self):
        picked_up_item = self.carrying
        self.carrying = None
        self.place_obj(picked_up_item)

    def _assign_initial_values(self):
        assert len(self.objects) == 2, "Expected exactly 2 object types, found {}".format(len(self.objects))
        selected_obj = self.objects[0]
        for obj in self.objects:
            if obj == selected_obj:
                obj.value = 1.0
            else:
                obj.value = -1.0

    def _switch_object_values(self):
        for obj in self.objects:
            self._switch_value(obj)

    @staticmethod
    def _switch_value(obj):
        if obj.value == 1.0:
            obj.value = -1.0
        else:
            obj.value = 1.0


class CollectorEnv7x7(CollectorEnv):
    def __init__(self):
        super().__init__(size=7, max_steps=200)


class CollectorEnv5x5(CollectorEnv):
    def __init__(self):
        super().__init__(size=5, max_steps=200)

