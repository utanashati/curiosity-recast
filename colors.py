import numpy as np


def same_1(num_rooms):
    return [1] * num_rooms


def same_16(num_rooms):
    return [16] * num_rooms


def diff_1_num_rooms(num_rooms):
    return range(1, num_rooms + 1)


def diff_1_num_rooms_random(num_rooms):
    colors = np.arange(1, num_rooms + 1)
    np.random.shuffle(colors)
    return colors.tolist()
