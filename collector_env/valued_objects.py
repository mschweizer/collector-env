from gym_minigrid.minigrid import Ball, Key


class ValuedBall(Ball):
    def __init__(self, color="blue", value=0.0):
        super().__init__(color)
        self.value = value

    def __str__(self):
        return "Ball (color: {color}, value: {value})".format(color=self.color, value=self.value)


class ValuedKey(Key):
    def __init__(self, color="green", value=0.0):
        super().__init__(color)
        self.value = value

    def __str__(self):
        return "Key (color: {color}, value: {value})".format(color=self.color, value=self.value)
