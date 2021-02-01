"""eos.py contains data structures to hold equation of state variables used
throughout the model.
"""
import warnings
import numpy as np
from scipy.constants import Boltzmann
from pyRT_DISORT.utilities.array_checks import ArrayChecker
from scipy.integrate import quadrature as quad


class ModelEquationOfState:
    """ModelEquationOfState computes EoS variables on a model grid.

    ModelEquationOfState accepts altitudes [km], pressures [Pa],
    temperatures [K], and number densities [particles / m**3], along with the
    altitudes where the model is defined. It linearly interpolates
    pressures and temperatures onto this new grid and assumes the ideal gas law
    to calculate the number density at the new grid. It also computes the
    column density within each layer using Gaussian quadrature. For these
    computations, quantities are assumed to be in MKS units.

    """
    def __init__(self, altitude_grid: np.ndarray, pressure_grid: np.ndarray,
                 temperature_grid: np.ndarray, number_density_grid: np.ndarray,
                 altitude_boundaries: np.ndarray) -> None:
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

        """
        return self.__temperature_boundaries


# TODO: This function does way more than one thing, so clean it up
# TODO: Decide if I want to input a string or something else. An array would be
#  maximally flexible but a string is most convenient.
def eos_from_file(array_path: str, altitude_boundaries: np.ndarray) \
        -> ModelEquationOfState:
    """Create a ModelEquationOfState from a file containing atmospheric equation
    of state variables. The array is assumed to be a 2D array, with the columns
    having the following meanings:

    0. The altitudes [km] at which the other equation of state variables are
    defined.
    1. The pressures [Pa] at the corresponding altitudes.
    2. The temperatures [K] at the corresponding altitudes.
    3. The number densities [particles / m**3] at the corresponding altitudes.

    Parameters
    ----------
    array_path: str
        Absolute path to the 2D array of atmospheric EoS variables.
    altitude_boundaries: np.ndarray
        The desired altitudes [km] of the model boundaries.

    Returns
    -------
    ModelEquationOfState
        The modeled equation of state object.

    Raises
    ------
    FileNotFoundError
        Raised if the input file path does not exist.
    IndexError
        Raised in the input array does not have the expected shape.
    TypeError
        Raised if the input array path is not a string.
    ValueError
        Raised if the input array path does not lead to a .npy file.

    """
    def read_file(file):
        try:
            return np.load(file)
        except TypeError:
            raise TypeError('array_path must be a str.') from None
        except FileNotFoundError:
            raise FileNotFoundError(f'No such file or directory: {file}.') \
                from None
        except ValueError:
            raise ValueError('The file cannot be opened. This likely means the '
                             'file is not a .npy file.')

    input_atmosphere = read_file(array_path)
    try:
        return ModelEquationOfState(input_atmosphere[:, 0],
                                    input_atmosphere[:, 1],
                                    input_atmosphere[:, 2],
                                    input_atmosphere[:, 3],
                                    altitude_boundaries)
    except IndexError:
        raise IndexError('Too many indices for array. This probably means the '
                         'input file is a 1D array instead of the expected 2D '
                         'array.')
