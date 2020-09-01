# 3rd-party imports
import numpy as np

# Local imports
from aerosol import Aerosol


class MarsDust(Aerosol):
    def __init__(self, phase_function_file, aerosol_file, theta, wavelength, wavelength_reference=9.3*10**3):
        """Initialize the Dust class

        Parameters
        ----------
        phase_function_file: str
            Unix-style path to the empirical phase function
        theta: float
            The angle at which to evaluate the phase functions in radians
        """
        self.phase_file = phase_function_file
        self.aerosol_file = aerosol_file
        self.theta = theta
        self.wavelength = wavelength
        self.wave_ref = wavelength_reference

    def get_phase_coefficients(self):
        """ Read in the the Legendre coefficients for dust

        Returns
        -------
        np.ndarray of the coefficients
        """
        return np.load(self.phase_file, allow_pickle=True)

    def evaluate_phase_function(self):
        """Evaluate the empirical phase function from Legendre coefficients at a given angle

        Returns
        -------
        float
        """
        return np.polynomial.legendre.legval(self.theta, self.get_phase_coefficients())

    def read_dust_file(self):
        return np.load(self.aerosol_file, allow_pickle=True)

    def interpolate_dust_asymmetry_parameter(self):
        """ Interpolate the dust HG asymmetry parameter at a wavelength

        Returns
        -------
        float: the interpolated parameter
        """
        dust_info = self.read_dust_file()
        wavelengths = dust_info[:, 0]
        g = dust_info[:, -1]
        return np.interp(self.wavelength, wavelengths, g)

    def conrath_profile(self, altitudes, dust_scale_height, nu):
        """Calculate the vertical dust distribution assuming a Conrath profile, i.e.
        q(z) / q(0) = exp( nu * (1 - exp(z / H)))
        where q is the mass mixing ratio

        Parameters
        ----------
        altitudes: np.ndarray
        dust_scale_height: float
        nu: float

        Returns
        -------
        """
        if np.any(altitudes < 0):
            print('Bad Conrath altitudes... they cannot be negative.')
            return
        elif dust_scale_height < 0:
            print('Bad Conrath scale height... it cannot be negative.')
            return
        elif nu < 0:
            print('Bad Conrath parameter nu... it cannot be negative.')
            return

        fractional_mixing_ratio = np.exp(nu * (1 - np.exp(altitudes / dust_scale_height)))
        return fractional_mixing_ratio

    def wavelength_scaling(self):
        """Make the wavelength scaling

        Returns
        -------

        """
        dust_info = self.read_dust_file()
        wavelengths = dust_info[:, 0]
        c_extinction = dust_info[:, 1]
        coefficients = np.interp(np.array([self.wavelength, self.wave_ref]), wavelengths, c_extinction)
        scaling = coefficients[0] / coefficients[1]
        return scaling





















class MarsWaterIce(Aerosol):
    def __init__(self, phase_function_file, theta, wavelength_reference=12.1*10**3):
        """Initialize the WaterIce class

        Parameters
        ----------
        phase_function_file: str
            Unix-style path to the empirical phase function
        theta: float
            The angle at which to evaluate the phase functions in radians
        """
        self.phase_file = phase_function_file
        self.theta = theta
        self.wave_ref = wavelength_reference

    def get_phase_coefficients(self):
        """Read in the the Legendre coefficients for water ice

        Returns
        -------
        np.ndarray of the coefficients
        """
        return np.load(self.phase_file, allow_pickle=True)

    def evaluate_phase_function(self):
        """Evaluate the empirical phase function from Legendre coefficients at a given angle

        Returns
        -------
        float
        """
        return np.polynomial.legendre.legval(self.theta, self.get_phase_coefficients())

    def read_ice_file(self):
        return np.load(self.aerosol_file, allow_pickle=True)

    def interpolate_ice_asymmetry_parameter(self):
        """Interpolate the ice HG asymmetry parameter at a wavelength

        Returns
        -------
        float: the interpolated parameter
        """
        ice_info = self.read_ice_file()
        wavelengths = ice_info[:, 0]
        g = ice_info[:, -1]
        return np.interp(self.wavelength, wavelengths, g)
