import numpy as np


class EmpiricalPhaseFunction:
    def __init__(self, legendre_coefficients_file, n_moments):
        self.legendre_file = legendre_coefficients_file
        self.n_moments = n_moments
        self.moments = self.make_phase_moments()

    def get_phase_function(self):
        return np.load(self.legendre_file, allow_pickle=True)

    def make_phase_moments(self):
        """ Make an array of the phase function moments. It will truncate at n_moments, or append 0s to the coefficients
        found in legendre_coefficients_file

        Returns
        -------
        moments: np.ndarray
            The empirical phase function moments
        """
        phase_function = self.get_phase_function()
        if len(phase_function) < self.n_moments:
            moments = np.zeros(self.n_moments)
            moments[: len(phase_function)] = phase_function
            return moments
        else:
            moments = phase_function[: self.n_moments]
            return moments


class HenyeyGreenstein:
    def __init__(self, g, n_moments):
        self.g = self.make_valid_asymmetry_parameter(g)
        self.n_moments = n_moments
        self.moments = self.calculate_hg_legendre_coefficients()

    @staticmethod
    def make_valid_asymmetry_parameter(g):
        """Check the HG asymmetry parameter is valid

        Parameters
        ----------
        g: float
            The Henyey-Greenstein asymmetry parameter. Can range from [-1, 1]

        Returns
        -------
        g if the parameter is valid; -1 or +1 if it's outside the valid range
        """
        if -1 <= g <= 1:
            return g
        else:
            print('The value of g ({}) isn\'t in the range [-1, 1]. Choosing the nearest value'.format(g))
            return np.sign(g)

    def calculate_hg_legendre_coefficients(self):
        """ Make the Legendre coefficients of a Henyey-Greenstein phase function up to n_moments

        Returns
        -------
        coefficients: np.ndarray
            The coefficients of the Legendre polynomials
        """
        orders = np.linspace(0, self.n_moments, num=self.n_moments)
        coefficients = (2*orders + 1) * self.g**orders
        return coefficients
