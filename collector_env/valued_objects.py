from gym_minigrid.minigrid import Ball, Key


class ValuedBall(Ball):
    def __init__(self, color="blue"):
        super().__init__(color)
        self.value = 0.0

    def __str__(self):
        return "Ball (color: {color}, value: {value})".format(color=self.color, value=self.value)


class ValuedKey(Key):
    def __init__(self, color="green"):
        super().__init__(color)
        self.value = 0.0

    def __str__(self):
        return "Key (color: {color}, value: {value})".format(color=self.color, value=self.value)
