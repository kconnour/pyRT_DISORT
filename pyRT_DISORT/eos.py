"""The eos module contains data structures to compute and hold equation of state
variables used throughout pyRT_DISORT.
"""
import numpy as np
from scipy.constants import Boltzmann
from scipy.integrate import quadrature as quad


'''class Hydrostatic:
    """A data structure computes a hydrostatic equation of state.

    Hydrostatic accepts pressure and temperature, and computes the
    corresponding number density to ensure the atmosphere follows the
    equation

    .. math::
       P = n k_B T

    where :math:`P` is the pressure, :math:`n` is the number density,
    :math:`k_B` is Boltzmann's constant, and :math:`T` is the temperature.

    """

    def __init__(self, pressure: np.ndarray, temperature: np.ndarray) -> None:
        """
        Parameters
        ----------
        pressure
            The pressure [Pa] of an arbitrarily sized grid.
        temperature
            The temperature [K] of an arbitrarily sized grid.

        Raises
        ------
        TypeError
            Raised if the inputs are not numeric arrays.
        ValueError
            Raised if the input arrays cannot be broadcast together or if they
            contain non-positive values.

        """
        self.__pressure = pressure
        self.__temperature = temperature

        self.__raise_value_error_if_inputs_are_not_positive()

        self.__number_density = self.__compute_number_density()

    # TODO: this does more than one thing
    def __raise_value_error_if_inputs_are_not_positive(self) -> None:
        try:
            if np.any(self.__pressure <= 0):
                message = 'pressure must contain only positive values.'
                raise ValueError(message)
        except TypeError as te:
            message = 'pressure must be a numpy.ndarray of numeric values.'
            raise TypeError(message) from te

        try:
            if np.any(self.__temperature <= 0):
                message = 'temperature must contain only positive values.'
                raise ValueError(message)
        except TypeError as te:
            message = 'temperature must be a numpy.ndarray of numeric values.'
            raise TypeError(message) from te

    def __compute_number_density(self) -> np.ndarray:
        try:
            return self.__pressure / self.__temperature / Boltzmann
        except ValueError as ve:
            message = 'The input arrays must have the same shape.'
            raise ValueError(message) from ve

    @property
    def pressure(self) -> np.ndarray:
        """Get the input pressure.

        Returns
        -------
        np.ndarray
            The input pressure.

        """
        return self.__pressure

    @property
    def temperature(self) -> np.ndarray:
        """Get the input temperature.

        Returns
        -------
        np.ndarray
            The input temperature.

        """
        return self.__temperature

    @property
    def number_density(self) -> np.ndarray:
        """Get the number density computed from the input pressure and
        temperature.

        Returns
        -------
        np.ndarray
            The computed number density.

        """
        return self.__number_density'''


class Hydrostatic:
    """A data structure computes a hydrostatic equation of state.

    Hydrostatic accepts equation of state variables and computes atmospheric
    properties from them, assuming the atmosphere follows the equation

    .. math::
       P = n k_B T

    where :math:`P` is the pressure, :math:`n` is the number density,
    :math:`k_B` is Boltzmann's constant, and :math:`T` is the temperature.

    """

    def __init__(self, altitude_grid: np.ndarray, pressure_grid: np.ndarray,
                 temperature_grid: np.ndarray, altitude_boundaries: np.ndarray,
                 particle_mass: float, gravity: float) -> None:
        """
        Parameters
        ----------
        altitude_grid
            The altitude grid [km] over which the equation of state variables
            are defined.
        pressure_grid
            The pressure [Pa] at the corresponding altitude.
        temperature_grid
            The temperature [K] at the corresponding altitude.
        altitude_boundaries
            The altitude to interpolate the pressure and temperature. This must
            have the same pixel dimension shape as altitude_grid.
        particle_mass
            The average mass [kg] of atmospheric particles.
        gravity
            The gravitational acceleration [kg m/s**2] of the atmosphere.

        """

        self.__altitude_grid = altitude_grid
        self.__altitude = altitude_boundaries

        self.__n_layers = self.__extract_n_layers()

        self.__pressure = \
            self.__interpolate_to_boundary_alts(pressure_grid)
        self.__temperature = \
            self.__interpolate_to_boundary_alts(temperature_grid)
        self.__number_density = \
            self.__compute_number_density(self.__pressure, self.__temperature)
        self.__column_density = \
            self.__compute_column_density()
        self.__scale_height = \
            self.__compute_scale_height(particle_mass, gravity)

    def __extract_n_layers(self) -> int:
        return self.__altitude.shape[0] - 1

    # TODO: Ideally I'd like to vectorize this
    def __interpolate_to_boundary_alts(self, grid: np.ndarray) -> np.ndarray:
        flattened_altitude_grid = \
            self.__flatten_along_pixel_dimension(self.__altitude_grid)
        flattened_boundaries = \
            self.__flatten_along_pixel_dimension(self.__altitude)
        flattened_quantity_grid = self.__flatten_along_pixel_dimension(grid)
        interpolated_quantity = np.zeros(flattened_boundaries.shape)

        for pixel in range(flattened_boundaries.shape[1]):
            interpolated_quantity[:, pixel] = \
                np.interp(flattened_boundaries[:, pixel],
                          flattened_altitude_grid[:, pixel],
                          flattened_quantity_grid[:, pixel])
        return interpolated_quantity.reshape(self.__altitude.shape)

    @staticmethod
    def __flatten_along_pixel_dimension(grid: np.ndarray) -> np.ndarray:
        return grid.reshape(grid.shape[0], int(grid.size / grid.shape[0]))

    @staticmethod
    def __compute_number_density(pressure, temperature) -> np.ndarray:
        return pressure / temperature / Boltzmann

    # TODO: Ideally I'd like to vectorize this
    # TODO: Mike said to do this in log(z) space. Is this still necessary?
    def __compute_column_density(self) -> np.ndarray:
        flattened_boundaries = \
            self.__flatten_along_pixel_dimension(self.__altitude)
        flattened_pressure = \
            self.__flatten_along_pixel_dimension(self.__pressure)
        flattened_temperature = \
            self.__flatten_along_pixel_dimension(self.__temperature)
        column_density = np.zeros((self.__n_layers,
                                   flattened_boundaries.shape[1]))
        for pixel in range(flattened_boundaries.shape[1]):
            colden = [quad(self.__make_n_at_altitude,
                           flattened_boundaries[i+1, pixel],
                           flattened_boundaries[i, pixel],
                           args=(flattened_boundaries[:, pixel],
                                 flattened_pressure[:, pixel],
                                 flattened_temperature[:, pixel]))[0]
                      for i in range(self.__n_layers)]
            column_density[:, pixel] = np.array(colden)
        return column_density

    def __make_n_at_altitude(self, z: float, alt_grid, p_grid, t_grid) -> np.ndarray:
        p = np.interp(z, alt_grid, p_grid)
        t = np.interp(z, alt_grid, t_grid)
        return self.__compute_number_density(p, t)

    def __compute_scale_height(self, particle_mass: float,
                               gravity: float) -> np.ndarray:
        return Boltzmann * self.__temperature / (particle_mass * gravity)

    @property
    def n_layers(self) -> int:
        """Get the number of layers in the model.

        Returns
        -------
        int
            The number of layers.

        """
        return self.__n_layers

    @property
    def altitude(self) -> np.ndarray:
        """Get the input boundary altitude.

        Returns
        -------
        np.ndarray
            The boundary altitude.

        """
        return self.__altitude

    @property
    def pressure(self) -> np.ndarray:
        """Get the pressure at the boundary altitude.

        Returns
        -------
        np.ndarray
            The boundary pressure.

        """
        return self.__pressure

    @property
    def temperature(self) -> np.ndarray:
        """Get the temperature at the boundary altitude.

        Returns
        -------
        np.ndarray
            The boundary pressure.

        """
        return self.__temperature

    @property
    def number_density(self) -> np.ndarray:
        """Get the number density at the boundary altitude.

        Returns
        -------
        np.ndarray
            The boundary number density.

        """
        return self.__number_density

    @property
    def column_density(self) -> np.ndarray:
        return self.__column_density

    @property
    def scale_height(self) -> np.ndarray:
        return self.__scale_height





















'''class ModelEquationOfState:
    """Compute equation of state variables on a model grid.

    ModelEquationOfState accepts altitudes [km], pressures [Pa],
    temperatures [K], and number densities [particles / m**3], along with the
    altitudes where the model is defined. It linearly interpolates
    pressures and temperatures onto this new grid and assumes the ideal gas law
    to calculate the number density at the new grid. It also computes the
    column density within each layer using Gaussian quadrature.

    """

    def __init__(self, altitude_grid: np.ndarray, pressure_grid: np.ndarray,
                 temperature_grid: np.ndarray, number_density_grid: np.ndarray,
                 altitude_boundaries: np.ndarray, particle_mass: float,
                 gravity: float) -> None:
        """
        Parameters
        ----------
        altitude_grid: np.ndarray
            The altitudes [km] at which the equation of state variables are
            defined.
        pressure_grid: np.ndarray
            The pressures [Pa] at the corresponding altitudes.
        temperature_grid: np.ndarray
            The temperatures [K] at the corresponding altitudes.
        number_density_grid: np.ndarray
            The number densities [particles / m**3] at the corresponding
            altitudes.
        altitude_boundaries: np.ndarray
            The desired altitudes [km] of the model boundaries.
        particle_mass: float
            The average mass [kg] of atmospheric particles.
        gravity: float
            The planetary gravity [kg * m / s**2].

        Raises
        ------
        IndexError
            Raised if the input grids do not have the same shape.
        TypeError
            Raised if any of the inputs are not np.ndarrays.
        ValueError
            Raised if any of the inputs have unphysical values.

        """
        self.__altitude_grid = altitude_grid
        self.__pressure_grid = pressure_grid
        self.__temperature_grid = temperature_grid
        self.__number_density_grid = number_density_grid
        self.__altitude_boundaries = altitude_boundaries

        self.__raise_error_if_input_variables_are_bad()
        self.__flip_grids_if_altitudes_are_mono_decreasing()

        self.__n_layers = len(self.__altitude_boundaries) - 1
        self.__pressure_boundaries = self.__make_pressure_model()
        self.__temperature_boundaries = self.__make_temperature_model()
        self.__number_density_boundaries = self.__make_number_density_model()
        self.__column_density_layers = self.__compute_column_density_layers()
        self.__scale_height_boundaries = self.__make_scale_height_boundaries(
            particle_mass, gravity)

    def __raise_error_if_input_variables_are_bad(self) -> None:
        self.__raise_error_if_altitude_grid_is_bad()
        self.__raise_error_if_pressure_grid_is_bad()
        self.__raise_error_if_temperature_grid_is_bad()
        self.__raise_error_if_number_density_grid_is_bad()
        self.__raise_error_if_altitude_model_is_bad()
        self.__raise_index_error_if_grids_are_not_same_shape()

    def __raise_error_if_altitude_grid_is_bad(self) -> None:
        self.__raise_error_if_grid_is_bad(self.__altitude_grid, 'altitude_grid')

    def __raise_error_if_pressure_grid_is_bad(self) -> None:
        self.__raise_error_if_grid_is_bad(self.__pressure_grid, 'pressure_grid')

    def __raise_error_if_temperature_grid_is_bad(self) -> None:
        self.__raise_error_if_grid_is_bad(
            self.__temperature_grid, 'temperature_grid')

    def __raise_error_if_number_density_grid_is_bad(self) -> None:
        self.__raise_error_if_grid_is_bad(
            self.__number_density_grid, 'number_density_grid')

    def __raise_index_error_if_grids_are_not_same_shape(self) -> None:
        if not self.__altitude_grid.shape == self.__temperature_grid.shape == \
               self.__pressure_grid.shape == self.__number_density_grid.shape:
            raise IndexError('All input grids must have the same shape.')

    def __raise_error_if_altitude_model_is_bad(self) -> None:
        self.__raise_error_if_grid_is_bad(
            self.__altitude_boundaries, 'altitude_boundaries')
        self.__raise_value_error_if_model_altitudes_are_not_mono_decreasing()
        self.__raise_value_error_if_too_few_layers_are_included()

    def __raise_error_if_grid_is_bad(self, array: np.ndarray, name: str) \
            -> None:
        try:
            checks = self.__make_grid_checks(array)
        except TypeError:
            raise TypeError(f'{name} is not a np.ndarray.') from None
        if not all(checks):
            raise ValueError(
                f'{name} must be a 1D array containing positive, finite '
                f'values.')

    @staticmethod
    def __make_grid_checks(grid: np.ndarray) -> list[bool]:
        grid_checker = ArrayChecker(grid)
        checks = [grid_checker.determine_if_array_is_1d(),
                  grid_checker.determine_if_array_is_finite(),
                  grid_checker.determine_if_array_is_non_negative()]
        return checks

    def __raise_value_error_if_model_altitudes_are_not_mono_decreasing(self) \
            -> None:
        model_checker = ArrayChecker(self.__altitude_boundaries)
        if not model_checker.determine_if_array_is_monotonically_decreasing():
            raise ValueError('altitude_model must be monotonically decreasing.')

    def __raise_value_error_if_too_few_layers_are_included(self) -> None:
        if len(self.__altitude_boundaries) < 2:
            raise ValueError('The model must contain at least 2 boundaries '
                             '(i.e. one layer).')

    def __flip_grids_if_altitudes_are_mono_decreasing(self) -> None:
        altitude_checker = ArrayChecker(self.__altitude_grid)
        if altitude_checker.determine_if_array_is_monotonically_decreasing():
            self.__altitude_grid = np.flip(self.__altitude_grid)
            self.__pressure_grid = np.flip(self.__pressure_grid)
            self.__temperature_grid = np.flip(self.__temperature_grid)
            self.__number_density_grid = np.flip(self.__number_density_grid)

    def __make_pressure_model(self) -> np.ndarray:
        return self.__interpolate_variable_to_model_altitudes(
            self.__pressure_grid)

    def __make_temperature_model(self) -> np.ndarray:
        return self.__interpolate_variable_to_model_altitudes(
            self.__temperature_grid)

    def __make_number_density_model(self) -> np.ndarray:
        return self.__pressure_boundaries / \
               (Boltzmann * self.__temperature_boundaries)

    def __interpolate_variable_to_model_altitudes(self, grid: np.ndarray) \
            -> np.ndarray:
        return np.interp(self.__altitude_boundaries, self.__altitude_grid, grid)

    def __compute_column_density_layers(self) -> np.array:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            colden = [quad(self.__make_n_at_altitude,
                           self.__altitude_boundaries[i + 1],
                           self.__altitude_boundaries[i])[0]
                      for i in range(self.__n_layers)]
            return np.array(colden)

    # TODO: This logic seems to duplicate functions above... try to fix
    def __make_n_at_altitude(self, z: float) -> float:
        p = np.interp(z, self.__altitude_grid, self.__pressure_grid)
        t = np.interp(z, self.__altitude_grid, self.__temperature_grid)
        return 1 / Boltzmann * p / t

    def __make_scale_height_boundaries(
            self, particle_mass: float, gravity: float) -> np.ndarray:
        return Boltzmann * self.__temperature_boundaries / \
               (particle_mass * gravity) / 1000

    @property
    def altitude_boundaries(self) -> np.ndarray:
        """Get the input altitudes at the model boundaries.

        Returns
        -------
        np.ndarray
            The boundary altitudes.

        """
        return self.__altitude_boundaries

    @property
    def column_density_layers(self):
        """Get the column density within each of the layers.

        Returns
        -------
        np.ndarray
            The layer column densities.

        """
        return self.__column_density_layers

    @property
    def n_layers(self) -> int:
        """Get the number of layers in the model.

        Returns
        -------
        int
            The number of layers.

        """
        return self.__n_layers

    @property
    def number_density_boundaries(self) -> np.ndarray:
        """Get the number density at the model boundaries.

        Returns
        -------
        np.ndarray
            The boundary number densities.

        """
        return self.__number_density_boundaries

    @property
    def pressure_boundaries(self) -> np.ndarray:
        """Get the pressure at the model boundaries.

        Returns
        -------
        np.ndarray
            The boundary pressures.

        """
        return self.__pressure_boundaries

    @property
    def temperature_boundaries(self) -> np.ndarray:
        """Get the temperature at the model boundaries.

        Returns
        -------
        np.ndarray
            The boundary temperatures.

        Notes
        -----
        In DISORT, this variable is named "TEMPER".

        """
        return self.__temperature_boundaries

    @property
    def scale_height_boundaries(self) -> np.ndarray:
        """Get the scale height at the model boundaries.

        Returns
        -------
        np.ndarray
            The scale heights at the boundaries

        Notes
        -----
        In DISORT, this variable is named "H_LYR". There is no documentation for
        this variable, so I'm assuming it to be in km. Also, the scale height
        for each layer seems like an odd name but f2py assures me that it's
        assumed to be of shape n_layers + 1, i.e. n_boundaries. It is only used
        if do_pseudo_sphere (defined in ModelBehavior) == True.

        """
        return self.__scale_height_boundaries'''


if __name__ == '__main__':
    z = np.linspace(100, 0, num=500).reshape((10, 25, 2))
    P = z + 1
    T = P * 2
    n = T * 10
    alts = np.linspace(50, 0, num=200).reshape((4, 25, 2))
    eos = Hydrostatic(z, P, T, alts, 9.8, 1)
    print(eos.pressure.shape)
    print(eos.column_density.shape)
