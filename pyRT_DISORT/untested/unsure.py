import numpy as np
from pyRT_DISORT.controller import ComputationalParameters


# TODO: Become sure of what this does and remove it
class Unsure:
    """This class makes the variable "h_lyr". I don't really know what this
    variable does since it can be all 0s.

    """
    def __init__(self, cp: ComputationalParameters):
        self.cp = cp
        self.__h_lyr = np.zeros(self.cp.n_layers+1)

    @property
    def h_lyr(self):
        return self.__h_lyr
