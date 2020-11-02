# 3rd-party imports
import numpy as np
from scipy.constants import Boltzmann


class Boundaries:
    """ Make a class to calculate the equation of state variables at the model boundaries."""
    def __init__(self, atmosphere):
        """
        Parameters
        ----------
        atmosphere: np.ndarray
            An array of the equations of state in the atmosphere
        """
        self.atmosphere = atmosphere
        self.altitude_boundaries, self.pressure_boundaries, self.temperature_boundaries, \
            self.number_density_boundaries = self.__read_atmosphere()

    def __read_atmosphere(self):
        altitudes = self.atmosphere[:, 0]
        pressures = self.atmosphere[:, 1]
        temperatures = self.atmosphere[:, 2]
        number_density = self.atmosphere[:, 3]
        return altitudes, pressures, temperatures, number_density


class Layers(Boundaries):
    """Make a class to calculate the equation of state variable at the midpoints of the boundaries."""
    def __init__(self, atmosphere):
        """
        Parameters
        ----------
        atmosphere: np.ndarray
            An array of the equations of state in the atmosphere
        """
        super().__init__(atmosphere)
        self.altitude_layers = self.__calculate_midpoints(self.altitude_boundaries)
        self.pressure_layers = self.__calculate_midpoints(self.pressure_boundaries)
        self.temperature_layers = self.__calculate_midpoints(self.temperature_boundaries)
        self.number_density_layers = self.__calculate_number_density_layers()
        self.column_density_layers = self.__calculate_column_density_layers()
        self.n_layers = len(self.altitude_layers)

    @staticmethod
    def __calculate_midpoints(array):
        return (array[:-1] + array[1:]) / 2

    def __calculate_number_density_layers(self):
        return self.pressure_layers / (Boltzmann * self.temperature_layers)

    def __calculate_column_density_layers(self):
        return self.number_density_layers * np.abs(np.diff(self.altitude_boundaries)) * 1000
