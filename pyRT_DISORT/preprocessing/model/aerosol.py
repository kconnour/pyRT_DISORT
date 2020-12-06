# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.preprocessing.utilities.array_checks import ArrayChecker


class ForwardScatteringProperty:
    """A ForwardScatteringProperty object represents a single forward scattering property for an aerosol, along with
    the particle sizes and wavelengths for which the values are defined."""
    def __init__(self, property_values, particle_size_grid=None, wavelength_grid=None):
        """
        Parameters
        ----------
        property_values: np.ndarray
            An (M)xN array of a scattering property. If the array has shape MxN, M corresponds to the particle size
            dimension and N corresponds to the wavelength dimension; if the array has shape N, N corresponds to
            whichever dimension is included.
        particle_size_grid: np.ndarray or None
            1D array of the particle sizes where property_values are defined. Default is None
        wavelength_grid: np.ndarray or None
            1D array of the wavelengths where property_values are defined. Default is None

        Attributes
        ----------
        property_values: np.ndarray
            The input property_values
        particle_size_grid: np.ndarray
            The input particle_size_grid
        wavelength_grid: np.ndarray
            The input wavelength_grid

        Notes
        -----
        In the special case where property_values is a singleton array, both particle_size_grid and wavelength_grid
        can be None.
        """
        self.property_values = property_values
        self.particle_size_grid = particle_size_grid
        self.wavelength_grid = wavelength_grid
        self.__property_dims = np.ndim(self.property_values)
        self.__scattering_property_is_singleton = self.__determine_scattering_property_is_singleton()
        self.__check_property_and_grids_are_physical()

    def __determine_scattering_property_is_singleton(self):
        return True if self.__property_dims == 1 and len(self.property_values) == 1 else False

    def __check_property_and_grids_are_physical(self):
        self.__check_property_is_plausible()
        self.__check_property_dimensions()
        self.__check_grid_is_plausible(self.particle_size_grid, 'particle_size_grid')
        self.__check_grid_is_plausible(self.wavelength_grid, 'wavelength_grid')
        self.__check_grids_match_2d_property()
        self.__check_grids_match_1d_property()
        self.__check_grids_match_0d_property()

    def __check_property_is_plausible(self):
        properties_checker = ArrayChecker(self.property_values, 'scattering_property')
        properties_checker.check_object_is_array()
        properties_checker.check_ndarray_is_numeric()
        properties_checker.check_ndarray_is_finite()

    def __check_property_dimensions(self):
        if self.__property_dims not in [1, 2]:
            raise IndexError('scattering_property must be a 1D or 2D np.ndarray')

    @staticmethod
    def __check_grid_is_plausible(grid, grid_name):
        if grid is not None:
            grid_checker = ArrayChecker(grid, grid_name)
            grid_checker.check_object_is_array()
            grid_checker.check_ndarray_is_numeric()
            grid_checker.check_ndarray_is_positive_finite()
            grid_checker.check_ndarray_is_1d()
            grid_checker.check_1d_array_is_monotonically_increasing()

    def __check_grids_match_2d_property(self):
        if self.__property_dims != 2:
            return
        scattering_property_shape = self.property_values.shape
        if self.particle_size_grid is None or self.wavelength_grid is None:
            raise ReferenceError('For 2D scattering_property, you need to include both particle_size_grid and '
                                 'wavelength_grid')
        if scattering_property_shape[0] != len(self.particle_size_grid):
            raise IndexError('For 2D scattering_property, its first dimension must be the same length as '
                             'particle_size_grid')
        if scattering_property_shape[1] != len(self.wavelength_grid):
            raise IndexError('For 2D scattering_property, its second dimension must be the same length as '
                             'wavelength_grid')

    def __check_grids_match_1d_property(self):
        if self.__property_dims != 1 or self.__scattering_property_is_singleton:
            return
        if not (self.particle_size_grid is None) ^ (self.wavelength_grid is None):
            raise TypeError(
                'For 1D scattering_property, provide one and only one of particle_size_grid and wavelength_grid')
        scattering_property_shape = self.property_values.shape
        if self.particle_size_grid is not None:
            if scattering_property_shape[0] != len(self.particle_size_grid):
                raise IndexError('For 1D scattering_property, its first dimension must be the same length as '
                                 'particle_size_grid or wavelength_grid, whichever is provided')
        if self.wavelength_grid is not None:
            if scattering_property_shape[0] != len(self.wavelength_grid):
                raise IndexError('For 1D scattering_property, its first dimension must be the same length as '
                                 'particle_size_grid or wavelength_grid, whichever is provided')

    def __check_grids_match_0d_property(self):
        if not self.__scattering_property_is_singleton:
            return
        if self.particle_size_grid is not None or self.wavelength_grid is not None:
            raise TypeError('For 0D scattering_property, do not provide particle_size_grid or wavelength_grid')


class ForwardScatteringProperties:
    def __init__(self, c_scattering, c_extinction, g=None):
        self.c_scattering = c_scattering
        self.c_extinction = c_extinction
        self.g = g
        self.__ensure_properties_are_fsc()

    def __ensure_properties_are_fsc(self):
        self.__ensure_property_is_ForwardScatteringProperty(self.c_scattering, 'c_scattering')
        self.__ensure_property_is_ForwardScatteringProperty(self.c_extinction, 'c_extinction')
        if self.g is not None:
            self.__ensure_property_is_ForwardScatteringProperty(self.g, 'g')

    def __ensure_property_is_ForwardScatteringProperty(self, forward_property, property_name):
        if not isinstance(forward_property, ForwardScatteringProperty):
            raise TypeError(f'{property_name} must be an instance of ForwardScatteringProperty')
