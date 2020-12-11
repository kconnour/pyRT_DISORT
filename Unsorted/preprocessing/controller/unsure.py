# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.preprocessing.controller.size import Size


class Unsure:
    """ I really don't know where to put these things in the code.. or what all of them are

    """
    def __init__(self, size):
        self.size = size
        assert isinstance(self.size, Size)

    def make_h_lyr(self):
        # Make the variable "h_lyr"
        return np.zeros(self.size.n_layers+1)
