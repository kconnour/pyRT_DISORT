# 3rd-party imports
import numpy as np
from scipy.constants import Boltzmann

# Local imports
from old.utilities import ArrayChecker


class InputAtmosphere:
    """An InputAtmosphere object simply holds on input equation of state variables on a known grid."""

    def __init__(self, atmosphere):
        """
        Parameters
        ----------
        atmosphere: np.ndarray
            2D array of the atmospheric state variables (z, P, T, and n). Variables must be specified in that order

        Attributes
        ----------
        altitude_grid: np.ndarray
            The first column from atmosphere [km]
        pressure_grid: np.ndarray
            The second column from atmosphere [Pascals]
        temperature_grid: np.ndarray
            The third column from atmosphere [K]
        number_density_grid: np.ndarray
            The fourth column from atmosphere [particles / m**3]
        """
        self.__atmosphere = atmosphere
        self.__check_atmosphere_values_are_physical()
        self.__check_atmosphere_has_4_columns()
        self.altitude_grid, self.pressure_grid, self.temperature_grid, self.number_density_grid = \
            self.__read_atmosphere()
        self.__check_altitudes_are_decreasing()

    def __check_atmosphere_values_are_physical(self):
        atmosphere_checker = ArrayChecker(self.__atmosphere, 'atmosphere')
        atmosphere_checker.check_object_is_array()
        atmosphere_checker.check_ndarray_is_numeric()
        atmosphere_checker.check_ndarray_is_finite()
        atmosphere_checker.check_ndarray_is_non_negative()
        atmosphere_checker.check_ndarray_is_2d()

    def __check_atmosphere_has_4_columns(self):
        if self.__atmosphere.shape[1] != 4:
            raise IndexError('The atmosphere must have 4 columns')

    def __read_atmosphere(self):
        altitudes = self.__atmosphere[:, 0]
        pressures = self.__atmosphere[:, 1]
        temperatures = self.__atmosphere[:, 2]
        number_density = self.__atmosphere[:, 3]
        return altitudes, pressures, temperatures, number_density

    def __check_altitudes_are_decreasing(self):
        altitude_grid_checker = ArrayChecker(self.altitude_grid, 'altitude_grid')
        altitude_grid_checker.check_1d_array_is_monotonically_decreasing()


class ModelGrid(InputAtmosphere):
    """A ModelGrid object computes the equation of state variables at user-defined altitudes."""

    def __init__(self, atmosphere, boundary_altitudes):
        """
        Parameters
        ----------
        atmosphere: np.ndarray
            2D array of the atmospheric state variables (z, P, T, and n). Variables must be specified in that order
        boundary_altitudes: np.ndarray
            1D array of the desired boundary altitudes to use in the model. Must be decreasing

        Attributes
        ----------
        boundary_altitudes: np.ndarray
            The user-input altitudes of the model at the boundaries
        boundary_pressures: np.ndarray
            The pressures at model_altitudes
        boundary_temperatures: np.ndarray
            The temperatures at model_altitudes
        boundary_number_denisty: np.ndarray
            The number density at model_altitudes
        layer_column_density: np.ndarray
            The column densities in the layers
        n_layers: int
            The number of layers in the model
        """
        super().__init__(atmosphere)
        self.boundary_altitudes = boundary_altitudes
        self.__check_boundary_altitudes()
        self.n_layers = len(self.boundary_altitudes) - 1
        self.boundary_pressures, self.boundary_temperatures, self.boundary_number_densities = \
            self.__regrid_atmospheric_variables()
        self.layer_altitudes = self.__calculate_midpoint_altitudes()
        self.column_density_layers = self.__calculate_layer_column_density()

    def __check_boundary_altitudes(self):
        altitude_checker = ArrayChecker(self.boundary_altitudes, 'boundary_altitudes')
        altitude_checker.check_object_is_array()
        altitude_checker.check_ndarray_is_numeric()
        altitude_checker.check_ndarray_is_finite()
        altitude_checker.check_ndarray_is_non_negative()
        altitude_checker.check_ndarray_is_1d()
        altitude_checker.check_1d_array_is_monotonically_decreasing()
        altitude_checker.check_1d_array_is_at_least(2)

    def __regrid_atmospheric_variables(self):
        regridded_pressures = self.__interpolate_variable_to_new_altitudes(self.pressure_grid)
        regridded_temperatures = self.__interpolate_variable_to_new_altitudes(self.temperature_grid)
        regridded_number_density = self.__interpolate_variable_to_new_altitudes(self.number_density_grid)
        return regridded_pressures, regridded_temperatures, regridded_number_density

    def __interpolate_variable_to_new_altitudes(self, variable_grid):
        # I'm forced to flip the arrays because numpy.interp demands xp be monotonically increasing
        return np.interp(self.boundary_altitudes, np.flip(self.altitude_grid), np.flip(variable_grid))

    def __calculate_midpoint_altitudes(self):
        return (self.boundary_altitudes[:-1] + self.boundary_altitudes[1:]) / 2

    def __calculate_number_density(self, altitude):
        interp_pressure = np.interp(altitude, np.flip(self.altitude_grid), np.flip(self.pressure_grid))
        interp_temp = np.interp(altitude, np.flip(self.altitude_grid), np.flip(self.temperature_grid))
        return interp_pressure / (Boltzmann * interp_temp)

    def __calculate_layer_column_density(self):
        num_den_layers = self.__calculate_number_density(self.layer_altitudes)
        return num_den_layers * np.abs(np.diff(self.boundary_altitudes)) * 1000
