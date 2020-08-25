import numpy as np


# Make the super class
class Aerosol(object):
    def check_valid_asymmetry_parameter(self, g):
        """Check the HG asymmetry parameter is valid

        Parameters
        ----------
        g: float
            The Henyey-Greenstein asymmetry parameter. Can range from [-1, 1]

        Returns
        -------
        True if the parameter is within [-1, 1]; False otherwise
        """
        if -1 <= g <= 1:
            return True
        else:
            return False

    def make_hg_phase_function(self, g, theta):
        """Evaluate the Henyey-Greenstein phase function for a generalized aerosol

        Parameters
        ----------
        g: float
            The Henyey-Greenstein asymmetry parameter. Can range from [-1, 1]
        theta: float
            The angle at which to evaluate the HG phase function

        Returns
        -------

        """
        if self.check_valid_asymmetry_parameter(g):
            return 1 / (4*np.pi) * (1 - g**2) / (1 + g**2 - 2*g*np.cos(theta))**(3/2)
        else:
            print('The asymmetry parameter is not valid. Ensure it\'s between [-1, 1]')
            return


# Make the sub-classes
class Dust(Aerosol):
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


# Right now this class doesn't really seem necessary but it could be a useful extension later on...
class WaterIce(Aerosol):
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
