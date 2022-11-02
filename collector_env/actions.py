from enum import IntEnum


class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2

    # Pick up an object
    pickup = 3
