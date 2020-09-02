# 3rd-party imports
import numpy as np

# Local imports
from atmosphere import Atmosphere
from planets.mars.mars_aerosols import MarsDust


class AtmosphericDust(Atmosphere):
    def __init__(self, atmosphere_file, dust_file, dust_scale_height, conrath_nu):
        super().__init__(atmosphere_file)
        self.dust_file = dust_file
        self.H = dust_scale_height
        self.nu = conrath_nu
        self.check_input()

    def check_input(self):
        """ Check that the input Conrath parameters don't suck

        Returns
        -------
        None
        """
        if np.isnan(self.H):
            print('You wily bastard, you input a nan as the Conrath dust scale height. Fix that.')
        elif np.isnan(self.nu):
            print('You wily bastard, you input a nan as the Conrath nu parameter. Fix that.')
        if self.H < 0:
            print('Bad Conrath dust scale height: it cannot be negative. Fix...')
            raise SystemExit('')
        elif self.nu < 0:
            print('Bad Conrath nu parameter: it cannot be negative. Fix...')
            raise SystemExit('')

    def make_conrath_profile(self):
        """Calculate the vertical dust distribution assuming a Conrath profile, i.e.
        q(z) / q(0) = exp( nu * (1 - exp(z / H)))
        where q is the mass mixing ratio

        Returns
        -------
        fractional_mixing_ratio: np.ndarray
            The fraction of the mass mixing ratio at the midpoint altitudes
        """

        fractional_mixing_ratio = np.exp(self.nu * (1 - np.exp(self.z_midpoints / self.H)))
        return fractional_mixing_ratio

    def calculate_dust_optical_depths(self, column_optical_depth, wavelength):
        """ Make the optical depths within each layer

        Parameters
        ----------
        column_optical_depth: float
            The column-integrated optical depths for each layer
        wavelength: float
            The wavelength at which to get the reference wavelength scaling

        Returns
        -------
        tau_dust: np.ndarray
            The optical depths in each layer
        """
        dust = MarsDust(self.dust_file)
        scaling = dust.make_wavelength_scaling(wavelength)
        fractional_mxing_ratio = self.conrath_profile()
        dust_scaling = np.sum(self.n * fractional_mxing_ratio)
        tau_dust = scaling * column_optical_depth * fractional_mxing_ratio * self.n / dust_scaling
        return tau_dust


#dust = DustAtmosphere('/home/kyle/repos/pyRT_DISORT/planets/mars/aux/mars_atm.npy',
#                      '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/dust.npy', np.nan, 0.5)
