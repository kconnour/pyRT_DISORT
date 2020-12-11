# 3rd-party imports
import numpy as np
from scipy.integrate import quadrature
from scipy.constants import Boltzmann

# Local imports
from pyRT_DISORT.preprocessing.utilities.array_checks import ArrayChecker


class InputAtmosphere:
    """ Ingest equation of state variables on a known grid """
    def __init__(self, atmosphere):
        """
        Parameters
        ----------
        atmosphere: np.ndarray
            A 2D array of the atmospheric state variables (z, P, T, and n). Variables must be specified in that order

        Attributes
        ----------
        atmosphere: np.ndarray
            The input atmosphere
        altitude_grid: np.ndarray
            The first column from atmosphere
        pressure_grid: np.ndarray
            The second column from atmosphere
        temperature_grid: np.ndarray
            The third column from atmosphere
        number_density_grid: np.ndarray
            The fourth column from atmosphere
        """
        self.atmosphere = atmosphere
        self.__check_atmosphere_values_are_plausible()
        self.__check_atmosphere_columns()
        self.altitude_grid, self.pressure_grid, self.temperature_grid, self.number_density_grid = \
            self.__read_atmosphere()
        self.__check_altitudes_are_decreasing()

    def __check_atmosphere_values_are_plausible(self):
        atmosphere_checker = ArrayChecker(self.atmosphere, 'atmosphere')
        atmosphere_checker.check_object_is_array()
        atmosphere_checker.check_ndarray_is_numeric()
        atmosphere_checker.check_ndarray_is_finite()
        atmosphere_checker.check_ndarray_is_non_negative()
        atmosphere_checker.check_ndarray_is_2d()

    def __check_atmosphere_columns(self):
        if self.atmosphere.shape[1] != 4:
            raise IndexError('The atmosphere must have 4 columns')

    def __read_atmosphere(self):
        altitudes = self.atmosphere[:, 0]
        pressures = self.atmosphere[:, 1]
        temperatures = self.atmosphere[:, 2]
        number_density = self.atmosphere[:, 3]
        return altitudes, pressures, temperatures, number_density

    def __check_altitudes_are_decreasing(self):
        altitude_grid_checker = ArrayChecker(self.altitude_grid, 'altitude_grid')
        altitude_grid_checker.check_1d_array_is_monotonically_decreasing()


class ModelGrid(InputAtmosphere):
    """ Calculate equation of state variables at user-defined altitudes"""
    def __init__(self, atmosphere, model_altitudes):
        """
        Parameters
        ----------
        atmosphere: np.ndarray
            An array of the equations of state in the atmosphere
        model_altitudes: np.ndarray
            1D array of the desired boundary altitudes to use in the model

        Attributes
        ----------
        model_altitudes: np.ndarray
            The user-input altitudes of the model at the boundaries
        model_pressures: np.ndarray
            The pressures at model_altitudes
        model_temperatures: np.ndarray
            The temperatures at model_altitudes
        model_number_denisty: np.ndarray
            The number density at model_altitudes
        column_density_layers: np.ndarray
            The column densities in the layers
        n_layers: int
            The number of layers in the model
        """
        super().__init__(atmosphere)
        self.model_altitudes = model_altitudes
        self.n_boundaries = len(self.model_altitudes)
        self.n_layers = self.n_boundaries - 1
        self.__check_model_altitudes()
        self.model_pressures, self.model_temperatures, self.model_number_densities = \
            self.__regrid_atmospheric_variables()
        self.altitude_layers = self.__calculate_midpoint_altitudes()
        self.column_density_layers = self.__calculate_column_density_layers()

    def __check_model_altitudes(self):
        altitude_checker = ArrayChecker(self.model_altitudes, 'model_altitudes')
        altitude_checker.check_object_is_array()
        altitude_checker.check_ndarray_is_numeric()
        altitude_checker.check_ndarray_is_finite()
        altitude_checker.check_ndarray_is_non_negative()
        altitude_checker.check_ndarray_is_1d()
        altitude_checker.check_1d_array_is_monotonically_decreasing()

    def __regrid_atmospheric_variables(self):
        regridded_pressures = self.__interpolate_variable_to_new_altitudes(self.pressure_grid)
        regridded_temperatures = self.__interpolate_variable_to_new_altitudes(self.temperature_grid)
        regridded_number_density = self.__interpolate_variable_to_new_altitudes(self.number_density_grid)
        return regridded_pressures, regridded_temperatures, regridded_number_density

    def __interpolate_variable_to_new_altitudes(self, variable_grid):
        # I'm forced to flip the arrays because numpy.interp demands xp be monotonically increasing
        return np.interp(self.model_altitudes, np.flip(self.altitude_grid), np.flip(variable_grid))

    def __calculate_number_density(self, altitude):
        interp_pressure = np.interp(altitude, np.flip(self.altitude_grid), np.flip(self.pressure_grid))
        interp_temp = np.interp(altitude, np.flip(self.altitude_grid), np.flip(self.temperature_grid))
        return interp_pressure / (Boltzmann * interp_temp)

    def __calculate_midpoint_altitudes(self):
        return (self.model_altitudes[:-1] + self.model_altitudes[1:]) / 2
    # I wish I knew why this refused to work
    '''def __calculate_column_density_layers(self):
        integral = np.zeros(self.n_layers)
        for i in range(self.n_layers):
            print(self.model_altitudes[i], self.model_altitudes[i+1])
            integral[i] = quadrature(self.__calculate_number_density, self.model_altitudes[i+1], self.model_altitudes[i])[0]
        return integral * 1000'''

    # This is the bad old way
    def __calculate_column_density_layers(self):
        num_den_layers = self.__calculate_number_density(self.altitude_layers)
        return num_den_layers * np.abs(np.diff(self.model_altitudes)) * 1000
