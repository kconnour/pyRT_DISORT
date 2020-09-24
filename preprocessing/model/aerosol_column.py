# 3rd-party imports
import numpy as np

# Local imports
from preprocessing.model.aerosol import Aerosol
from preprocessing.model.atmosphere import Layers


class Column:
    def __init__(self, aerosol, layers, aerosol_scale_height, conrath_nu, effective_radii,
                 column_integrated_optical_depths):
        """ Initialize the class

        Parameters
        ----------
        aerosol: Aerosol
            An aerosol object
        layers: Layers
            A Layer object
        aerosol_scale_height: float
            The scale height of the aerosol
        conrath_nu: float
            The Conrath nu parameter
        effective_radii: np.ndarray
            The effective radii of this aerosol
        column_optical_depths: np.ndarray
            The column-integrated optical depth at the effective radii
        """
        self.aerosol = aerosol
        self.layers = layers
        self.H = aerosol_scale_height
        self.nu = conrath_nu
        self.effective_radii = effective_radii
        self.column_integrated_optical_depths = column_integrated_optical_depths

        assert isinstance(self.aerosol, Aerosol), 'aerosol needs to be an instance of Aerosol.'
        assert isinstance(self.layers, Layers), 'layers needs to be an instance of Layers.'
        self.check_conrath_parameters_do_not_suck()
        assert isinstance(self.effective_radii, np.ndarray), 'effective_radii needs to be a numpy array'
        assert isinstance(self.column_integrated_optical_depths, np.ndarray), 'column_optical_depths needs to be a numpy array'
        assert len(self.effective_radii) == len(self.column_integrated_optical_depths), 'effective_radii need to be of len(ODs)'

        # Make hidden arrays
        self.optical_depths_with_size = self.calculate_optical_depths()
        self.single_scattering_albedos_with_size = self.calculate_single_scattering_albedos()

        # Make the normal arrays
        self.optical_depths = self.reduce_optical_depths_size_dim()
        self.single_scattering_albedos = self.reduce_single_scattering_albedos_size_dim()

    def check_conrath_parameters_do_not_suck(self):
        if np.isnan(self.H):
            print('You wily bastard, you input a nan as the Conrath dust scale height. Fix that.')
        elif np.isnan(self.nu):
            print('You wily bastard, you input a nan as the Conrath nu parameter. Fix that.')
        if self.H < 0:
            raise SystemExit('Bad Conrath aerosol scale height: it cannot be negative. Fix...')
        elif self.nu < 0:
            raise SystemExit('Bad Conrath nu parameter: it cannot be negative. Fix...')

    def make_conrath_profile(self):
        """Calculate the vertical dust distribution assuming a Conrath profile, i.e.
        q(z) / q(0) = exp( nu * (1 - exp(z / H)))
        where q is the mass mixing ratio

        Returns
        -------
        fractional_mixing_ratio: np.ndarray (len(altitude_layer))
            The fraction of the mass mixing ratio at the midpoint altitudes
        """

        fractional_mixing_ratio = np.exp(self.nu * (1 - np.exp(self.layers.altitude_layers / self.H)))
        return fractional_mixing_ratio

    def calculate_optical_depths(self, optical_depth_minimum=10**-7):
        """ Make the optical depths within each layer

        Returns
        -------
        tau_aerosol: np.ndarray (n_layers, n_wavelengths)
            The optical depths in each layer at each wavelength
        """
        vertical_mixing_ratio = self.make_conrath_profile()
        dust_scaling = np.sum(self.layers.column_density_layers * vertical_mixing_ratio)
        tau_aerosol = np.multiply.outer(np.outer(vertical_mixing_ratio * self.layers.column_density_layers,
                                        self.column_integrated_optical_depths), self.aerosol.extinction_ratios) / dust_scaling

        # Make sure each grid point is at least optical_depth_minimum
        return np.where(tau_aerosol < optical_depth_minimum, optical_depth_minimum, tau_aerosol)

    def calculate_single_scattering_albedos(self):
        return self.optical_depths_with_size * self.aerosol.single_scattering_albedos

    def reduce_optical_depths_size_dim(self):
        # I'm assuming the total optical depth is just the sum over the size dimension
        return np.sum(self.optical_depths_with_size, axis=1)

    def reduce_single_scattering_albedos_size_dim(self):
        # I'm assuming the SSA is the weighted sum over the size dimension
        return np.average(self.single_scattering_albedos_with_size, axis=1, weights=self.column_integrated_optical_depths)
