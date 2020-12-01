# 3rd-party imports
import numpy as np

# Local import
from pyRT_DISORT.preprocessing.utilities.array_checks import ArrayChecker


class AerosolProperties:
    """ An AerosolProperties object is a container for an aerosol's properties"""
    def __init__(self, aerosol_properties, particle_size_grid=None, wavelength_grid=None, debug=False):
        """
        Parameters
        ----------
        aerosol_properties: np.ndarray
            An (M)xNx3 array of the aerosol's properties. If the array is MxNx3, M corresponds to the particle size
            dimension and N corresponds to the wavelength dimension; if the array is Nx3, N corresponds to whichever
            dimension is included. The final dimension is expected to have the extinction coefficient, scattering
            coefficient, and asymmetry parameter in that order.
        particle_size_grid: np.ndarray, semi-optional
            A 1D array of particle sizes corresponding to the first dimension of aerosol_properties, if applicable. If
            aerosol_properties is an MxNx3 array, particle_size_grid is must be be shape M; otherwise for 2D input it
            must be shape N. Default is None.
        wavelength_grid: np.ndarray, semi-optional
            1D array of wavelengths corresponding to one dimension of aerosol_properties, if applicable. It must be of
            shape N. Default is None.
        debug: bool, optional
            Denote if this code should perform sanity checks on inputs to see if they're expected. Default is False

        Attributes
        ----------
        aerosol_properties: np.ndarray
            The input aerosol_properties
        particle_size_grid: np.ndarray or None
            The input particle_size_grid
        wavelength_grid: np.ndarray or None
            The input wavelength_grid
        c_extinction_grid: np.ndarray
            1D array of the extinction coefficient contained within aerosol_properties
        c_scattering_grid: np.ndarray
            1D array of the scattering coefficient contained within aerosol_properties
        g_grid: np.ndarray
            1D array of the asymmetry parameter contained within aerosol_properties

        Notes
        ----
        For an MxNx3 aerosol_properties, both particle_size_grid and wavelength_grid must be included. For an Nx3
        aerosol_properties, one and only one of particle_size_grid and wavelength_grid must be included.
        """
        self.aerosol_properties = aerosol_properties
        self.particle_size_grid = particle_size_grid
        self.wavelength_grid = wavelength_grid
        self.particle_none = self.__check_if_input_is_none(self.particle_size_grid)
        self.wavelength_none = self.__check_if_input_is_none(self.wavelength_grid)

        if debug:
            self.__array_dimensions = self.__get_properties_dimensions()
            self.__array_shape = self.__get_properties_shape()
            self.__check_properties_and_grids()

        self.__c_ext_ind = 0    # I define these 3 because I may loosen the ordering restriction in the future
        self.__c_sca_ind = 1
        self.__g_ind = 2
        self.c_extinction_grid, self.c_scattering_grid, self.g_grid = self.__read_aerosol_file()

        if debug:
            self.__check_radiative_properties()

    @staticmethod
    def __check_if_input_is_none(ndarray):
        return True if ndarray is None else False

    def __get_properties_dimensions(self):
        return np.ndim(self.aerosol_properties)

    def __get_properties_shape(self):
        return np.shape(self.aerosol_properties)

    def __check_properties_and_grids(self):
        self.__check_properties_are_plausible()
        self.__check_grid_is_plausible(self.particle_size_grid, 'particle_size_grid')
        self.__check_grid_is_plausible(self.wavelength_grid, 'wavelength_grid')
        self.__check_properties_dimensions()
        self.__check_grids_match_2d_properties()
        self.__check_grids_match_3d_properties()

    def __check_properties_are_plausible(self):
        properties_checker = ArrayChecker(self.aerosol_properties, 'aerosol_properties')
        properties_checker.check_object_is_array()
        properties_checker.check_ndarray_is_numeric()
        properties_checker.check_ndarray_is_positive_finite()

    @staticmethod
    def __check_grid_is_plausible(grid, grid_name):
        grid_checker = ArrayChecker(grid, grid_name)
        grid_checker.check_object_is_array()
        grid_checker.check_ndarray_is_numeric()
        grid_checker.check_ndarray_is_positive_finite()
        grid_checker.check_ndarray_is_1d()
        grid_checker.check_1d_array_is_monotonically_increasing()

    def __check_properties_dimensions(self):
        if self.__array_dimensions not in [2, 3]:
            raise IndexError('aerosol_properties must be a 2D or 3D array')
        if self.__array_shape[-1] != 3:
            raise IndexError('aerosol_properties must be an (M)xNx3 array for the 3 parameters')

    def __check_grids_match_2d_properties(self):
        if self.__array_dimensions != 2:
            return
        if not (self.particle_size_grid is None) ^ (self.wavelength_grid is None):
            raise TypeError(
                'For 2D aerosol_properties, provide one and only one of particle_size_grid and wavelength_grid')
        if not self.particle_none:
            if self.__array_shape[0] != len(self.particle_size_grid):
                raise IndexError(
                    'For 2D files, aerosol_properties\' first dimension must be the same length as particle_size_grid')
        if not self.wavelength_none:
            if self.__array_shape[0] != len(self.wavelength_grid):
                raise IndexError(
                    'For 2D files, aerosol_properties\' first dimension must be the same length as wavelength_grid')

    def __check_grids_match_3d_properties(self):
        if self.__array_dimensions != 3:
            return
        if (self.particle_size_grid is None) or (self.wavelength_grid is None):
            raise TypeError('You need to include both particle_size_grid and wavelength_grid')
        if self.__array_shape[0] != len(self.particle_size_grid):
            raise IndexError(
                'For 3D files, aerosol_properties\' first dimension must be the same length as particle_size_grid')
        if self.__array_shape[1] != len(self.wavelength_grid):
            raise IndexError(
                    'For 3D files, aerosol_properties\' second dimension must be the same length as wavelength_grid')

    def __read_aerosol_file(self):
        c_extinction = np.take(self.aerosol_properties, self.__c_ext_ind, axis=-1)
        c_scattering = np.take(self.aerosol_properties, self.__c_sca_ind, axis=-1)
        g = np.take(self.aerosol_properties, self.__g_ind, axis=-1)
        return c_extinction, c_scattering, g

    def __check_radiative_properties(self):
        self.__check_radiative_property_is_plausible(self.c_extinction_grid, 'c_extinction')
        self.__check_radiative_property_is_plausible(self.c_scattering_grid, 'c_scattering')
        self.__check_radiative_property_is_plausible(self.g_grid, 'g')

    @staticmethod
    def __check_radiative_property_is_plausible(radiative_property, property_name):
        grid_checker = ArrayChecker(radiative_property, property_name)
        grid_checker.check_ndarray_is_numeric()
        grid_checker.check_ndarray_is_positive_finite()
