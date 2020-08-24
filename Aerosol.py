import numpy as np

# Make the super class
class Aerosol():
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
class Dust(Aerosol):   # or whatever for g.. I just made up a number for now
    def __init__(self, phase_function_file, theta):
        """Initialize the Dust class

        Parameters
        ----------
        phase_function_file: str
            Unix-style path to the empirical phase function
        theta: float
            The angle at which to evaluate the phase functions in radians
        """
        self.phase_file = phase_function_file
        self.g = 0.78
        self.theta = theta

    def get_phase_coefficients(self):
        """ Read in the the Legendre coefficients for dust

        Returns
        -------
        np.ndarray of the coefficients
        """
        return np.load(self.phase_file, allow_pickle=True)

    def empirical_phase_function(self):
        """Evaluate the empirical phase function from Legendre coefficients at a given angle

        Returns
        -------
        float
        """
        return np.polynomial.legendre.legval(self.theta, self.get_phase_coefficients())


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
        self.g = 0.5
        self.theta = theta

    def get_phase_coefficients(self):
        """ Read in the the Legendre coefficients for water ice

        Returns
        -------
        np.ndarray of the coefficients
        """
        return np.load(self.phase_file, allow_pickle=True)

    def empirical_phase_function(self):
        """Evaluate the empirical phase function from Legendre coefficients at a given angle

        Returns
        -------
        float
        """
        return np.polynomial.legendre.legval(self.theta, self.get_phase_coefficients())


''' #Examples
dustfile = '/Users/kyco2464/pyRT_DISORT/aux/legendre_coeff_dust.npy'
icefile = '/Users/kyco2464/pyRT_DISORT/aux/legendre_coeff_h2o_ice.npy'
dust = Dust(dustfile, 0.5)
print(dust.empirical_phase_function())
print(dust.make_hg_phase_function(0.78, 0.5))
ice = WaterIce(icefile, 0.5)
print(ice.empirical_phase_function())
print(ice.make_hg_phase_function(0.78, 0.5))'''
