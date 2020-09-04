# 3rd-party imports
import numpy as np

# Local imports
from atmosphere import Atmosphere
from planets.mars.mars_aerosols import MarsDust


class AtmosphericDust(Atmosphere):
    def __init__(self, atmosphere_file, dust_file, legendre_file, dust_scale_height, conrath_nu, column_optical_depth,
                 wavelength):
        super().__init__(atmosphere_file)
        self.dust_file = dust_file
        self.legendre = legendre_file
        self.H = dust_scale_height
        self.nu = conrath_nu
        self.check_input()

        # Make sure the object "knows" about its parameters
        self.column_optical_depth = column_optical_depth
        self.wavelength = wavelength
        self.optical_depth = self.calculate_dust_optical_depths()

        # Define variables from the dust file
        self.wavelengths, self.c_ext, self.c_sca, self.kappa, self.g = self.read_dust_file()
        self.scattering_ratio = self.calculate_dust_scattering_coefficient()
        self.legendre_coefficients = self.get_legendre_coefficients()

    def read_dust_file(self):
        dust_file = np.load(self.dust_file)
        wavelengths = dust_file[:, 0]
        c_extinction = dust_file[:, 1]
        c_scattering = dust_file[:, 2]
        kappa = dust_file[:, 3]
        g = dust_file[:, 4]
        return wavelengths, c_extinction, c_scattering, kappa, g

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

    def calculate_dust_optical_depths(self):
        """ Make the optical depths within each layer

        Returns
        -------
        tau_dust: np.ndarray
            The optical depths in each layer
        """
        dust = MarsDust(self.dust_file)
        scaling = dust.calculate_wavelength_scaling(self.wavelength)
        fractional_mixing_ratio = self.make_conrath_profile()
        dust_scaling = np.sum(self.n * fractional_mixing_ratio)
        tau_dust = scaling * self.column_optical_depth * fractional_mixing_ratio * self.n / dust_scaling
        return tau_dust

    def calculate_dust_scattering_coefficient(self):
        """ Calculate the scattering coefficient, C_scattering / C_extinction at a wavelength

        Returns
        -------
        scattering_coefficient: float
            The scattering coefficient
        """
        interpolated_extinction = np.interp(self.wavelength, self.wavelengths, self.c_ext)
        interpolated_scattering = np.interp(self.wavelength, self.wavelengths, self.c_sca)
        scattering_coefficient = interpolated_scattering / interpolated_extinction
        return scattering_coefficient

    def get_legendre_coefficients(self):
        return np.load(self.legendre)

#dust = DustAtmosphere('/home/kyle/repos/pyRT_DISORT/planets/mars/aux/mars_atm.npy',
#                      '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/dust.npy', np.nan, 0.5)

atmfile = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/mars_atm.npy'
dustfile = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/dust.npy'
polyfile = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/legendre_coeff_dust.npy'
atm = Atmosphere(atmfile)
atm.add_rayleigh_co2_optical_depth(1)
dust = AtmosphericDust(atmfile, dustfile, polyfile, 10, 0.5, 1, 9.3)
atm.add_aerosol(dust)
#print(atm.calculate_single_scattering_albedo())
m = atm.calculate_polynomial_moments()

#atmtest = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/mars_atm_test.npy'
#atm0 = Atmosphere(atmtest)
