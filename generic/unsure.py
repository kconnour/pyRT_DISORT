# 3rd-party imports
import numpy as np

# Local imports
from generic.size import Size


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
        return np.zeros((int(0.5*self.size.n_cmu), int(0.5*self.size.n_cmu+1), self.size.n_cmu))

    def make_rhou(self):
        # Make the variable "rhou"
        return np.zeros((self.size.n_umu, int(0.5*self.size.n_cmu+1), self.size.n_cmu))

    def make_rho_accurate(self):
        # Make the variable "rho_accurate"
        return np.zeros((self.size.n_umu, self.size.n_phi))

    def make_bemst(self):
        # Make the variable "bemst"
        return np.zeros(int(0.5*self.size.n_cmu))

    def make_emust(self):
        # Make the variable "emust"
        return np.zeros(self.size.n_umu)
