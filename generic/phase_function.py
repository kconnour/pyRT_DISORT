# 3rd-party imports
import numpy as np


class EmpiricalPhaseFunction:
    def __init__(self, legendre_coefficients_file, n_moments):
        self.legendre_file = legendre_coefficients_file
        self.n_moments = n_moments

    def get_phase_function(self):
        return np.load(self.legendre_file, allow_pickle=True)

    def update_phase_function(self):
        """ Make an array of empirical phase function moments

        Returns
        -------
        moments: np.ndarray (n_moments)
            A truncated array of moments, or the array of empirical moments with 0s appended
        """
        phase_function_coefficients = self.get_phase_function()
        if len(phase_function_coefficients) < self.n_moments:
            moments = np.zeros(self.n_moments)
            moments[: len(phase_function_coefficients)] = phase_function_coefficients
            return moments
        else:
            moments = phase_function_coefficients[: self.n_moments]
            return moments

    def make_phase_function(self, n_layers, n_wavelengths):
        """ Make an empirical phase function

        Parameters
        ----------
        n_layers: int
            The number of atmospheric layers
        n_wavelengths: int
            The number of wavelengths

        Returns
        -------
        empirical_phase_function: np.ndarray (n_moments, n_layers, n_wavelengths)
            An array of the empirical coefficients
        """
        phase_function = self.update_phase_function()
        print(phase_function)
        empirical_phase_function = np.zeros((self.n_moments, n_layers, n_wavelengths))
        normalization = np.linspace(0, self.n_moments-1, num=self.n_moments) * 2 + 1
        for i in range(n_layers):
            for j in range(n_wavelengths):
                empirical_phase_function[:, i, j] = phase_function / normalization

        print(np.amax(empirical_phase_function[1:, :, :]))
        return empirical_phase_function


class HenyeyGreensteinPhaseFunction:
    def __init__(self, n_moments):
        self.n_moments = n_moments

    def calculate_hg_legendre_coefficients(self, g):
        """ Calculate the HG Legendre coefficients up to n_moments for a given g asymmetry parameter

        Parameters
        ----------
        g: float
            The HG asymmetry parameter

        Returns
        -------
        coefficients: np.ndarray (n_moments)
            The coefficients for this g
        """
        orders = np.linspace(0, self.n_moments-1, num=self.n_moments)
        coefficients = (2*orders + 1) * g**orders
        return coefficients

    def make_phase_function(self, n_layers, g_values):
        """ Make a Henyey-Greenstein phase function

        Parameters
        ----------
        n_layers: int
            The number of atmospheric layers
        g_values: np.ndarray
            The g asymmetry parameters for each of the wavelengths

        Returns
        -------
        hg_phase_function: np.ndarray (n_moments, n_layers, n_wavelengths)
            An array of HG coefficients
        """
        hg_phase_function = np.zeros((self.n_moments, n_layers, len(g_values)))
        for i in range(n_layers):
            for counter, g in enumerate(g_values):
                hg_phase_function[:, i, counter] = self.calculate_hg_legendre_coefficients(g)

        return hg_phase_function


class RayleighPhaseFunction:
    def __init__(self, n_moments):
        self.n_moments = n_moments

    def make_phase_function(self, n_layers, n_wavelengths):
        """ Make a Rayleigh phase function

        Parameters
        ----------
        n_layers: int
            The number of atmospheric layers
        n_wavelengths: int
            The number of wavelengths

        Returns
        -------
        rayleigh_phase_function: np.ndarray (n_moments, n_layers, n_wavelengths)
            An array of the Rayleigh coefficients
        """
        rayleigh_phase_function = np.zeros((self.n_moments, n_layers, n_wavelengths))
        rayleigh_phase_function[0, :, :] = 1
        rayleigh_phase_function[2, :, :] = 0.1
        return rayleigh_phase_function
