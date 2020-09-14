# 3rd-party imports
import numpy as np

# Local imports
from generic.aerosol import Aerosol


class Column:
    def __init__(self, aerosol, aerosol_scale_height, conrath_nu, column_optical_depth):
        """ Initialize the class

        Parameters
        ----------
        aerosol: Aerosol
            An aerosol object
        aerosol_scale_height: float
            The scale height of the aerosol
        conrath_nu: float
            The Conrath nu parameter
        column_optical_depth: float
            The column-integrated optical depth
        """
        self.aerosol = aerosol
        self.H = aerosol_scale_height
        self.nu = conrath_nu
        self.column_optical_depth = column_optical_depth
        self.check_conrath_parameters()

        assert isinstance(self.aerosol, Aerosol), 'aerosol needs to be an instance of Aerosol.'

    def check_conrath_parameters(self):
        """ Check that the Conrath parameters don't suck

        Returns
        -------
        None
        """
        if np.isnan(self.H):
            print('You wily bastard, you input a nan as the Conrath dust scale height. Fix that.')
        elif np.isnan(self.nu):
            print('You wily bastard, you input a nan as the Conrath nu parameter. Fix that.')
        if self.H < 0:
            raise SystemExit('Bad Conrath aerosol scale height: it cannot be negative. Fix...')
        elif self.nu < 0:
            raise SystemExit('Bad Conrath nu parameter: it cannot be negative. Fix...')

    def make_conrath_profile(self, altitude_layer):
        """Calculate the vertical dust distribution assuming a Conrath profile, i.e.
        q(z) / q(0) = exp( nu * (1 - exp(z / H)))
        where q is the mass mixing ratio

        Parameters
        ----------
        altitude_layer: np.ndarray
            The altitudes of each layer's midpoint

        Returns
        -------
        fractional_mixing_ratio: np.ndarray (len(altitude_layer))
            The fraction of the mass mixing ratio at the midpoint altitudes
        """

        fractional_mixing_ratio = np.exp(self.nu * (1 - np.exp(altitude_layer / self.H)))
        return fractional_mixing_ratio

    def calculate_aerosol_optical_depths(self, altitude_layer, column_density_layers):
        """ Make the optical depths within each layer

        Parameters
        ----------
        altitude_layer: np.ndarray (len(n_layers))
            The altitudes to create the profile for
        column_density_layers: np.ndarray (len(n_layers))
            The column density if each of the layers

        Returns
        -------
        tau_aerosol: np.ndarray (n_layers, n_wavelengths)
            The optical depths in each layer at each wavelength
        """
        vertical_mixing_ratio = self.make_conrath_profile(altitude_layer)
        dust_scaling = np.sum(column_density_layers * vertical_mixing_ratio)
        tau_aerosol = np.outer(column_density_layers * vertical_mixing_ratio, self.aerosol.extinction_ratio) * \
            self.column_optical_depth / dust_scaling
        return tau_aerosol
