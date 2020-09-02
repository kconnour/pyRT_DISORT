# 3rd-party imports
import numpy as np
from scipy.constants import Boltzmann
from scipy.integrate import quadrature


class Atmosphere:
    def __init__(self, atmosphere_file, n_layers=14, z_bottom=0, z_top=100, p_surface=492, scale_height=10,
                 constant_pressure=True, top_temperature=160, bottom_temperature=212):
        # Needed for creating an atmosphere from user-defined values
        self.atm_file = atmosphere_file

        # Read in as much of the atmosphere as possible
        self.z, self.P, self.T, self.n, self.N = self.read_atmosphere()

        # If no z is provided, make the altitude and pressure grid at constant altitude
        if np.all(self.z == 0) and not constant_pressure:
            self._n_layers = n_layers
            self._z_bottom = z_bottom
            self._z_top = z_top
            self._H = scale_height
            self.z = self.make_constant_altitude_boundaries()
            self.P = self.make_pressure_profile(self.z) * p_surface

        # Otherwise make the altitudes and pressures at constant pressures
        elif np.all(self.z == 0) and constant_pressure:
            self._n_layers = n_layers
            self._z_bottom = z_bottom
            self._z_top = z_top
            self._H = scale_height
            self.z, unscaled_pressures = self.make_constant_pressure_boundaries()
            self.P = unscaled_pressures * p_surface

        # If temperature isn't provided, make it here
        if np.all(self.T == 0):
            self._T_bottom = bottom_temperature
            self._T_top = top_temperature
            self.T = self.make_constant_temperature_boundaries()

        self.z_midpoints = self.calculate_altitude_midpoints()

        # If number density isn't provided, make it here
        if np.all(self.n == 0):
            self.n = self.calculate_number_density(self.z_midpoints)

        # If column density isn't provided, make it here
        if np.all(self.N == 0):
            self.N = self.calculate_column_density()

        # Make a list to hold any aerosols that can be added to the atmosphere
        self.aerosols = []

    def read_atmosphere(self):
        """ Read in the atmospheric quantities from atmosphere_file

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
        column_density = atmosphere[:, 4]
        return altitudes, pressures, temperatures, number_density, column_density

    def make_constant_altitude_boundaries(self):
        """ Make the boundaries for constant altitude layers

        Returns
        -------
        altitudes: np.ndarray
            An array of equally spaced boundaries of the altitudes
        """
        altitudes = np.linspace(self._z_bottom, self._z_top, num=self._n_layers+1)
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
        fractional_pressures = np.linspace(bottom_pressure, top_pressure, num=self._n_layers+1, endpoint=True)
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
        temperatures = np.linspace(self._T_bottom, self._T_top, num=self._n_layers+1)
        return temperatures

    def calculate_altitude_midpoints(self):
        """ Compute the altitudes at the midpoint between each layer

        Returns
        -------
        midpoints: np.ndarray
            The midpoint altitudes, such that len(midpoint) = len(self.z) - 1
        """
        midpoints = (self.z[:-1] + self.z[1:]) / 2
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
        interpolated_pressure = np.interp(altitudes, self.z, self.P)
        interpolated_temperature = np.interp(altitudes, self.z, self.T)
        number_density = interpolated_pressure / interpolated_temperature / Boltzmann
        return number_density

    def calculate_column_density(self):
        """Calculate the column density (particles / unit area) by integrating number density with Gaussian quadrature.

        Returns
        -------
        column_density: np.ndarray
            The column density of each layer
        """
        column_density = np.zeros(len(self.z_midpoints))
        for i in range(len(column_density)):
            integral, absolute_error = quadrature(self.calculate_number_density, self.z[i]*1000, self.z[i+1]*1000)
            column_density[i] = integral * 1000

        return column_density

    def add_aerosol(self, aerosol):
        """ Add an aerosol to the atmosphere object

        Parameters
        ----------
        aerosol: tuple
            The aerosol's optical depths, single scattering albedo, and phase function

        Returns
        -------
        None
        """
        self.aerosols.append(aerosol)


#atmfile = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/mars_atm.npy'
#atm = Atmosphere(atmfile)
#atmtest = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/mars_atm_test.npy'
#atm0 = Atmosphere(atmtest)
