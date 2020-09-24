# 3rd-party imports
import numpy as np
from scipy.constants import Boltzmann


class Boundaries:
    def __init__(self, atmosphere_file):
        self.atm_file = atmosphere_file
        # Read in as much of the atmosphere as possible
        self.altitude_boundaries, self.pressure_boundaries, self.temperature_boundaries, \
            self.number_density_boundaries = self.read_atmosphere()

    def read_atmosphere(self):
        atmosphere = np.load(self.atm_file, allow_pickle=True)
        altitudes = atmosphere[:, 0]
        pressures = atmosphere[:, 1]
        temperatures = atmosphere[:, 2]
        number_density = atmosphere[:, 3]
        return altitudes, pressures, temperatures, number_density


class Layers(Boundaries):
    def __init__(self, atmosphere_file):
        """ Initialize the class to create the equations of state of the layers

        Parameters
        ----------
        atmosphere_file: str
            The complete path to the atmosphere file
        """
        super().__init__(atmosphere_file)
        self.altitude_layers = self.calculate_midpoints(self.altitude_boundaries)
        self.pressure_layers = self.calculate_midpoints(self.pressure_boundaries)
        self.temperature_layers = self.calculate_midpoints(self.temperature_boundaries)
        self.number_density_layers = self.calculate_number_density_layers()
        self.column_density_layers = self.calculate_column_density_layers()
        self.n_layers = len(self.altitude_layers)

    @staticmethod
    def calculate_midpoints(array):
        return (array[:-1] + array[1:]) / 2

    def calculate_number_density_layers(self):
        return self.pressure_layers / (Boltzmann * self.temperature_layers)

    def calculate_column_density_layers(self):
        return self.number_density_layers * np.diff(self.altitude_boundaries) * 1000
