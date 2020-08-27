import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quadrature
from scipy.constants import Boltzmann


class Layers:
    def __init__(self, n_layers, n_moments, quadrature_order, scale_height, atmosphere_file):
        self.layers = n_layers
        self.moments = n_moments
        self.order = quadrature_order
        self.H = scale_height
        self.atm = atmosphere_file

    def make_constant_altitude_boundaries(self):
        """Make the boundaries for layers equally spaced in altitude

        Returns
        -------
        boundaries: np.ndarray
            The boundaries
        """
        boundaries = np.linspace(self.top_altitude, self.bottom_altitude, num=self.layers+1, endpoint=True)
        return boundaries

    def make_constant_pressure_boundaries(self):
        """Make the boundaries for layers equally spaced in pressure. Assume an exponential pressure profile:
        P(z) = P_o * np.exp(-z/H)

        Returns
        -------
        boundaries: np.ndarray
            The boundaries
        """
        top_pressure = self.pressure_profile(self.top_altitude)
        bottom_pressure = self.pressure_profile(self.bottom_altitude)
        pressures = np.linspace(top_pressure, bottom_pressure, num=self.layers+1, endpoint=True)
        boundaries = -self.H * np.log(pressures)
        return boundaries

    def pressure_profile(self, z):
        """Create a pressure profile for an exponential atmosphere

        Parameters
        ----------
        z: np.ndarray
            The altitudes to create the profile for

        Returns
        -------
        frac: np.ndarray
            The fraction of the surface pressure at a given altitude
        """
        frac = np.exp(-z / self.H)
        return frac

    def read_atmosphere(self):
        """Read in the atmospheric layers

        Returns
        -------
        z: np.ndarray
            The altitudes
        P: np.ndarray
            The pressures at the provided altitudes
        T: np.ndarray
            The temperatures at the provided altitudes
        """
        atm = np.load(self.atm)
        z = atm[:, 0]
        P = atm[:, 1]
        T = atm[:, 2]
        return z, P, T

    def calculate_number_density(self, altitudes):
        """Calculate number density (particles / unit volume) at any altitude.
        Assume the atmosphere obeys the ideal gas law.

        Parameters
        ----------
        altitudes: np.ndarray
            The altitudes at which to compute the number density

        Returns
        -------
        number_density: np.ndarray
            The number density at the input altitudes
        """
        z, P, T = self.read_atmosphere()
        interp_pressure = np.interp(altitudes, z, P)
        interp_temperature = np.interp(altitudes, z, T)
        number_density = interp_pressure / interp_temperature / Boltzmann
        return number_density

    def calculate_column_density(self, layer_bottom, layer_top):
        """Calculate the column density (particles / unit area) in a given layer. This is computed by integrating
        number density from the layer bottom to layer top with Gaussian quadrature.

        Parameters
        ----------
        layer_bottom: float
            The altitude of the bottom of the layer
        layer_top: float
            The altitude of hte top of the layer

        Returns
        -------
        integral: float
            The number density in the given layer
        """
        integral, absolute_error = quadrature(self.calculate_number_density, layer_bottom, layer_top)
        return integral, error


class Atmosphere:
    def __init__(self, altitudes, temperatures, pressures):
        self.altitudes = altitudes
        self.temperatures = temperatures
        self.pressures = pressures
        if not self.check_valid_input():
            print('The shapes of altitude, temperature, and/or pressure don\'t match. Exiting...')
            raise SystemExit()

    def check_valid_input(self):
        if len(self.altitudes) == len(self.temperatures) == len(self.pressures):
            return True
        else:
            return False
