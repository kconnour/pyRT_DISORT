import numpy as np


class Output:
    def __init__(self, n_user_levels, n_phi, n_umu, n_boundaries, n_cmu):
        self.levels = n_user_levels
        self.boundaries = n_boundaries
        self.phi = n_phi
        self.umu = n_umu
        self.cmu = n_cmu

    def make_h_lyr(self):
        # Make the variable "h_lyr"
        return np.zeros(self.boundaries)

    def make_rhoq(self):
        # Make the variable "rhoq"
        return np.zeros((0.5*self.cmu, 0.5*self.cmu+1, self.cmu))

    def make_rhou(self):
        # Make the variable "rhou"
        return np.zeros((self.umu, 0.5*self.cmu+1, self.cmu))

    def make_rho_accurate(self):
        # Make the variable "rho_accurate"
        return np.zeros((self.umu, self.phi))

    def make_bemst(self):
        # Make the variable "bemst"
        return np.zeros(0.5*self.cmu)

    def make_emust(self):
        # Make the variable "emust"
        return np.zeros(self.umu)

    def make_direct_beam_flux(self):
        # Make the variable "rfldir"
        return np.zeros(self.levels)

    def make_diffuse_down_flux(self):
        # Make the variable "rfldn" (total minus direct-beam)
        return np.zeros(self.levels)

    def make_diffuse_up_flux(self):
        # Make the variable "flup"
        return np.zeros(self.levels)

    def make_flux_divergence(self):
        # Make the variable "dfdt" (d(net_flux) / d(optical_depth)) this is an exact result
        return np.zeros(self.levels)

    def make_mean_intensity(self):
        # Make the variable "uavg"
        return np.zeros(self.levels)

    def make_intensity(self):
        # Make the variable "uu"
        return np.zeros((self.umu, self.levels, self.phi))

    def make_albedo_medium(self):
        # Make the variable "albmed"
        return np.zeros(self.umu)

    def make_transmissivity_medium(self):
        # Make the variable "trnmed"
        return np.zeros(self.umu)
