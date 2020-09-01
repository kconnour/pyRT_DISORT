# 3rd-party imports
import numpy as np
from scipy.constants import Boltzmann
from scipy.integrate import quadrature


class Atmosphere:
    def __init__(self, atmosphere_file):
        # Needed for creating an atmosphere from user-defined values
        self.atm_file = atmosphere_file

        # Read in / make the atmosphere
        self.z, self.P, self.T = self.read_atmosphere()
        self.z_midpoints = self.calculate_altitude_midpoints()
        self.n = self.calculate_number_density(self.z_midpoints)
        self.N = self.calculate_column_density()

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
        """
        atmosphere = np.load(self.atm_file, allow_pickle=True)
        altitudes = atmosphere[:, 0]
        pressures = atmosphere[:, 1]
        temperatures = atmosphere[:, 2]
        return altitudes, pressures, temperatures

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
            integral, absolute_error = quadrature(self.calculate_number_density, self.z[i], self.z[i+1])
            column_density[i] = integral

        return column_density


#atmfile = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/mars_atm.npy'
#atm = Atmosphere(atmosphere_file=atmfile)
#print(atm.z_midpoints)
#print(atm.N)
