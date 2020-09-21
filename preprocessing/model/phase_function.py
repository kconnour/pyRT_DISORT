# Built-in imports
import os
import fnmatch as fnm

# 3rd-party imports
import numpy as np


class EmpiricalPhaseFunction:
    def __init__(self, legendre_coefficients_file, wavelengths, effective_radii, user_wavelength, user_radius):
        self.legendre_file = legendre_coefficients_file
        self.radii = effective_radii
        self.wavelengths = wavelengths
        self.radius = user_radius
        self.wavelength = user_wavelength

    def get_phase_function(self):
        radius_indices = self.get_nearest_indices(self.radius, self.radii)
        wavelength_indices = self.get_nearest_indices(self.wavelength, self.wavelengths)
        coefficients = np.load(self.legendre_file, allow_pickle=True)
        # I'm not sure why I cannot do this on one line...
        phase_function = coefficients[:, :, radius_indices]
        phase_function = phase_function[:, wavelength_indices, :]
        return phase_function

    @staticmethod
    def get_nearest_indices(values, array):
        diff = (values.reshape(1, -1) - array.reshape(-1, 1))
        indices = np.abs(diff).argmin(axis=0)
        return indices

    def update_phase_function(self, n_moments):
        """ Make an array of empirical phase function moments

        Returns
        -------
        moments: np.ndarray (n_moments)
            A truncated array of moments, or the array of empirical moments with 0s appended
        """
        phase_function_coefficients = self.get_phase_function()
        if phase_function_coefficients.shape[0] < n_moments:
            moments = np.zeros((n_moments, phase_function_coefficients.shape[1], phase_function_coefficients.shape[2]))
            moments[: len(phase_function_coefficients), :, :] = phase_function_coefficients
            return moments
        else:
            moments = phase_function_coefficients[:n_moments, :, :]
            return moments

    def make_phase_function(self, n_layers, n_moments):
        """ Make an empirical phase function

        Parameters
        ----------
        n_layers: int
            The number of atmospheric layers
        n_moments: int
            The number of Legendre phase function moments

        Returns
        -------
        empirical_phase_function: np.ndarray (n_moments, n_layers, n_wavelengths)
            An array of the empirical coefficients
        """
        phase_function = self.update_phase_function(n_moments)
        empirical_phase_function = np.zeros((n_moments, n_layers, len(self.wavelength), len(self.radius)))
        normalization = np.linspace(0, n_moments-1, num=n_moments) * 2 + 1
        for i in range(n_layers):
            empirical_phase_function[:, i, :, :] = (phase_function.T / normalization).T

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

    def make_phase_function(self, n_layers, n_wavelengths, n_radii):
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
        rayleigh_phase_function = np.zeros((self.n_moments, n_layers, n_wavelengths, n_radii))
        rayleigh_phase_function[0, :, :, :] = 1
        rayleigh_phase_function[2, :, :, :] = 0.1
        return rayleigh_phase_function


file = '/home/kyle/repos/pyRT_DISORT/preprocessing/planets/mars/aux/phase_functions.npy'
effective_radius = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1, 1.5, 2, 2.5, 3])
wavelengths = np.load('/home/kyle/repos/pyRT_DISORT/preprocessing/planets/mars/aux/dust.npy')[:, 0]
r = np.array([1, 1.6])
w = np.array([1, 1.23])
e = EmpiricalPhaseFunction(file, wavelengths, effective_radius, w, r)
epf = e.make_phase_function(14, 128)
