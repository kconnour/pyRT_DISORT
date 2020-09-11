import numpy as np
from scipy.constants import Boltzmann


class Atmosphere:
    def __init__(self, atmosphere_file, number_boundaries=15, z_bottom=0, z_top=100, p_surface=492, scale_height=10,
                 constant_pressure=True, top_temperature=160, bottom_temperature=212):

        self.atm_file = atmosphere_file
        # Read in as much of the atmosphere as possible
        self.altitude_boundaries, self.pressure_boundaries, self.temperature_boundaries, \
            self.number_density_boundaries = self.read_atmosphere()
        self.n_boundaries = len(self.altitude_boundaries)
        self.n_layers = self.n_boundaries - 1

        # If no z is provided, make the altitude and pressure grid at constant altitude
        if np.all(self.altitude_boundaries == 0) and not constant_pressure:
            self.n_boundaries = number_boundaries
            self.n_layers = self.n_boundaries - 1
            self._z_bottom = z_bottom
            self._z_top = z_top
            self._H = scale_height
            self.altitude_boundaries = self.make_constant_altitude_boundaries()
            self.pressure_boundaries = self.make_pressure_profile(self.altitude_boundaries) * p_surface

        # Otherwise make the altitudes and pressures at constant pressure levels
        elif np.all(self.altitude_boundaries == 0) and constant_pressure:
            self.n_boundaries = number_boundaries
            self.n_layers = self.n_boundaries - 1
            self._z_bottom = z_bottom
            self._z_top = z_top
            self._H = scale_height
            self.altitude_boundaries, unscaled_pressures = self.make_constant_pressure_boundaries()
            self.pressure_boundaries = unscaled_pressures * p_surface

        # If temperature isn't provided, make it here
        if np.all(self.temperature_boundaries == 0):
            self._T_bottom = bottom_temperature
            self._T_top = top_temperature
            self.temperature_boundaries = self.make_constant_temperature_boundaries()

        # Make the midpoint z, P, and T values
        self.altitude_layers = self.calculate_midpoints(self.altitude_boundaries)
        self.pressure_layers = self.calculate_midpoints(self.pressure_boundaries)
        self.temperature_layers = self.calculate_midpoints(self.temperature_boundaries)

        # If number density isn't provided, make it here. Assume the ideal gas law always
        if np.all(self.number_density_boundaries == 0):
            self.number_density_boundaries = self.calculate_number_density(self.altitude_boundaries)
            self.number_density_layers = self.calculate_number_density(self.altitude_layers)

        # And finally, make the column density in the layers
        self.column_density_layers = self.calculate_column_density()

    def read_atmosphere(self):
        """ Read in equation of state variables from atmosphere_file

        Returns
        -------
        altitudes: np.ndarray
            The altitudes from atmosphere_file
        pressures: np.ndarray
            The pressures from atmosphere_file
        temperatures: np.ndarray
            The temperatures from atmosphere_file
        number_density: np.ndarray
            The number density from atmosphere_file
        column_density: np.ndarray
            The column density from atmosphere_file
        """
        atmosphere = np.load(self.atm_file, allow_pickle=True)
        altitudes = atmosphere[:, 0]
        pressures = atmosphere[:, 1]
        temperatures = atmosphere[:, 2]
        number_density = atmosphere[:, 3]
        return altitudes, pressures, temperatures, number_density

    def make_constant_altitude_boundaries(self):
        """ Make the boundaries for constant altitude layers

        Returns
        -------
        altitudes: np.ndarray
            An array of equally spaced boundaries of the altitudes
        """
        altitudes = np.linspace(self._z_bottom, self._z_top, num=self.n_layers+1)
        return altitudes

    def make_constant_pressure_boundaries(self):
        """Make the boundaries for layers equally spaced in pressure. Assume an exponential pressure profile:
        P(z) = P_o * np.exp(-z/H)

        Returns
        -------
        boundaries: np.ndarray
            The boundaries
        """
        top_pressure = self.make_pressure_profile(self._z_top)
        bottom_pressure = self.make_pressure_profile(self._z_bottom)
        fractional_pressures = np.linspace(bottom_pressure, top_pressure, num=self.n_layers+1, endpoint=True)
        boundaries = -self._H * np.log(fractional_pressures)
        return boundaries, fractional_pressures

    def make_pressure_profile(self, z):
        """Create a pressure profile for an exponential atmosphere

        Parameters
        ----------
        z: np.ndarray
            The altitudes to create the profile for

        Returns
        -------
        frac: np.ndarray or float
            The fraction of the surface pressure at a given altitude
        """
        frac = np.exp(-z / self._H)
        return frac

    def make_constant_temperature_boundaries(self):
        """ Make the boundaries for constant temperature layers

        Returns
        -------
        temperatures: np.ndarray
            An array of equally spaced boundaries of the temperatures
        """
        temperatures = np.linspace(self._T_bottom, self._T_top, num=self.n_layers+1)
        return temperatures

    @staticmethod
    def calculate_midpoints(quantity):
        """ Compute the quantity at the midpoint of each layer

        Returns
        -------
        midpoints: np.ndarray
            The midpoint quantities
        """
        midpoints = (quantity[:-1] + quantity[1:]) / 2
        return midpoints

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
        interpolated_pressure = np.interp(altitudes, self.altitude_boundaries, self.pressure_boundaries)
        interpolated_temperature = np.interp(altitudes, self.altitude_boundaries, self.temperature_boundaries)
        number_density = interpolated_pressure / interpolated_temperature / Boltzmann
        return number_density

    def calculate_column_density(self):
        """Calculate the column density (particles / unit area) by integrating number density by the trapezoidal rule

        Returns
        -------
        column_density: np.ndarray
            The column density of each layer
        """
        number_density_midpoints = self.calculate_midpoints(self.number_density_boundaries)
        column_density = number_density_midpoints * np.diff(self.altitude_boundaries) * 1000
        return column_density
