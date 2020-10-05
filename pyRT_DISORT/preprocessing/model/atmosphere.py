# 3rd-party imports
import numpy as np
from scipy.constants import Boltzmann


class Boundaries:
    """ Make a class to calculate the equation of state variables at the model boundaries."""
    def __init__(self, atmosphere_file):
        """
        Parameters
        ----------
        atmosphere_file: str
            The complete path to the atmosphere file
        """
        self.atm_file = atmosphere_file
        self.altitude_boundaries, self.pressure_boundaries, self.temperature_boundaries, \
            self.number_density_boundaries = self.__read_atmosphere()

    def __read_atmosphere(self):
        atmosphere = np.load(self.atm_file, allow_pickle=True)
        altitudes = atmosphere[:, 0]
        pressures = atmosphere[:, 1]
        temperatures = atmosphere[:, 2]
        number_density = atmosphere[:, 3]
        return altitudes, pressures, temperatures, number_density


class Layers(Boundaries):
    """Make a class to calculate the equation of state variable at the midpoints of the boundaries."""
    def __init__(self, atmosphere_file):
        """
        Parameters
        ----------
        atmosphere_file: str
            The complete path to the atmosphere file
        """
        super().__init__(atmosphere_file)
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
