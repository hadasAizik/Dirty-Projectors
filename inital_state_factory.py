import numpy as np
from state import State
import image_processing as impr


class InitStateFactory:

    def __init__(self, shape):
        self._shape = shape

    def random_init(self, black_ratio=0.01):
        """
        @black_ratio - ratio of black pixels out of all pixels given.
        """
        shape_x, shape_y, shape_z = self._shape
        random = np.random.rand(shape_x, shape_y, shape_z)
        return State(impr.quantize_image_reverse(random, 1 - black_ratio))

    def blank_slate(self):
        return State(np.zeros(self._shape))

    def black_slate(self):
        return State(np.ones(self._shape))

    def duplicate_state(self, matrix):
        return State(matrix)
