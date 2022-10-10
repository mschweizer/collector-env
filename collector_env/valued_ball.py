from minigrid.minigrid_env import Ball


class ValuedBall(Ball):
    def __init__(self, color="blue"):
        super().__init__(color)
        self.value = 0.0

    def __str__(self):
        return "Ball (color: {color}, value: {value})".format(color=self.color, value=self.value)

