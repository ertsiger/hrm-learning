import numpy as np
import random


def randargmax(input_vector):
    return random.choice(np.flatnonzero(input_vector == np.max(input_vector)))
