# 3rd-party imports
import numpy as np

# Local imports
from aerosol import Aerosol

# Right now MarsDust is identical to MarsWaterIce, but I imagine they'll be extended in the future


class MarsDust(Aerosol):
    def __init__(self, phase_function_file, aerosol_file, theta, wavelength):
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


class MarsWaterIce(Aerosol):
    def __init__(self, phase_function_file, theta):
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

    def get_phase_coefficients(self):
        """ Read in the the Legendre coefficients for water ice

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
        """ Interpolate the ice HG asymmetry parameter at a wavelength

        Returns
        -------
        float: the interpolated parameter
        """
        ice_info = self.read_ice_file()
        wavelengths = ice_info[:, 0]
        g = ice_info[:, -1]
        return np.interp(self.wavelength, wavelengths, g)
