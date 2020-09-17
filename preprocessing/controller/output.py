# 3rd-party imports
import numpy as np

# Local imports
from generic.size import Size


class Output:
    def __init__(self, size):
        self.size = size
        assert isinstance(self.size, Size)

    def make_direct_beam_flux(self):
        # Make the variable "rfldir"
        return np.zeros(self.size.n_user_levels)

    def make_diffuse_down_flux(self):
        # Make the variable "rfldn" (total minus direct-beam)
        return np.zeros(self.size.n_user_levels)

    def make_diffuse_up_flux(self):
        # Make the variable "flup"
        return np.zeros(self.size.n_user_levels)

    def make_flux_divergence(self):
        # Make the variable "dfdt" (d(net_flux) / d(optical_depth)) this is an exact result
        return np.zeros(self.size.n_user_levels)

    def make_mean_intensity(self):
        # Make the variable "uavg"
        return np.zeros(self.size.n_user_levels)

    def make_intensity(self):
        # Make the variable "uu"
        return np.zeros((self.size.n_umu, self.size.n_user_levels, self.size.n_phi))

    def make_albedo_medium(self):
        # Make the variable "albmed"
        return np.zeros(self.size.n_umu)

    def make_transmissivity_medium(self):
        # Make the variable "trnmed"
        return np.zeros(self.size.n_umu)
