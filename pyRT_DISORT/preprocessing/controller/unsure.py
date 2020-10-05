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

    def make_rhoq(self):
        # Make the variable "rhoq"
        return np.zeros((int(0.5*self.size.n_streams), int(0.5*self.size.n_streams+1), self.size.n_streams))

    def make_rhou(self):
        # Make the variable "rhou"
        return np.zeros((self.size.n_umu, int(0.5*self.size.n_streams+1), self.size.n_streams))

    def make_rho_accurate(self):
        # Make the variable "rho_accurate"
        return np.zeros((self.size.n_umu, self.size.n_phi))

    def make_bemst(self):
        # Make the variable "bemst"
        # In DISOBRDF.f it's called the directional emissivity at quadrature angles
        return np.zeros(int(0.5*self.size.n_streams))

    def make_emust(self):
        # Make the variable "emust"
        # In DISOBRDF.f it's called the directional emissivity at user angles
        return np.zeros(self.size.n_umu)
