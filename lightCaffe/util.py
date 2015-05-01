__author__ = 'sichen'

import numpy as np


def is_near_enough(a, b, epsilon=1e-3):
    return np.linalg.norm(a / np.linalg.norm(a) - b / np.linalg.norm(b)) < epsilon