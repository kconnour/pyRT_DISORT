# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.preprocessing.utilities.array_checks import CheckArray


class ParameterChecker(CheckArray):
    def __init__(self, array):
        super().__init__(array)
        self.nonetype = self.__check_if_input_is_none()

    def __check_if_input_is_none(self):
        return True if self.ndarray is None else False

    def check(self):
        if not self.nonetype:
            self.check_object_is_array()
            self.check_ndarray_is_numeric()
            self.check_ndarray_is_positive_finite()
            self.check_ndarray_is_1d()


class AerosolPropertiesChecker(CheckArray):
    def __init__(self, array, particle_size_grid, wavelength_grid):
        super().__init__(array)
        self.particle_size_grid = particle_size_grid
        self.wavelength_grid = wavelength_grid
        self.__array_dimensions = self.get_array_dimensions()
        self.__array_shape = self.get_array_shape()
        self.__particle_none = self.__check_if_input_is_none(self.particle_size_grid)
        self.__wavelength_none = self.__check_if_input_is_none(self.wavelength_grid)

    def get_array_dimensions(self):
        return np.ndim(self.ndarray)

    def get_array_shape(self):
        return np.shape(self.ndarray)

    @staticmethod
    def __check_if_input_is_none(ndarray):
        return True if ndarray is None else False

    def check(self):
        self.check_object_is_array()
        self.check_ndarray_is_numeric()
        self.check_ndarray_is_positive_finite()
        self.__check_properties_dimensions()
        self.__check_grids_match_2d_properties()
        self.__check_grids_match_3d_properties()

    def __check_properties_dimensions(self):
        if self.__array_dimensions not in [2, 3]:
            raise IndexError('aerosol_properties must be a 2D or 3D array')
        if self.__array_shape[-1] != 3:
            raise IndexError('aerosol_properties must be an (M)xNx3 array for the 3 parameters')

    def __check_grids_match_2d_properties(self):
        if self.__array_dimensions != 2:
            return
        if not (self.particle_size_grid is None) ^ (self.wavelength_grid is None):
            raise ValueError(
                'For 2D aerosol_properties, provide one and only one of particle_size_grid and wavelength_grid')
        if not self.__particle_none:
            if self.__array_shape[0] != len(self.particle_size_grid):
                raise IndexError(
                    'For 2D files, aerosol_properties\' first dimension must be the same length as particle_size_grid')
        if not self.__wavelength_none:
            if self.__array_shape[0] != len(self.wavelength_grid):
                raise IndexError(
                    'For 2D files, aerosol_properties\' first dimension must be the same length as wavelength_grid')

    def __check_grids_match_3d_properties(self):
        if self.__array_dimensions != 3:
            return
        if (self.particle_size_grid is None) or (self.wavelength_grid is None):
            raise ValueError('You need to include both particle_size_grid and wavelength_grid')
        if self.__array_shape[0] != len(self.particle_size_grid):
            raise IndexError(
                'For 3D files, aerosol_properties\' first dimension must be the same length as particle_size_grid')
        if self.__array_shape[1] != len(self.wavelength_grid):
            raise IndexError(
                    'For 3D files, aerosol_properties\' second dimension must be the same length as wavelength_grid')


class AerosolProperties:
    def __init__(self, aerosol_properties, particle_size_grid=None, wavelength_grid=None):
        self.aerosol_properties = aerosol_properties
        self.particle_size_grid = particle_size_grid
        self.wavelength_grid = wavelength_grid

        self.__check_properties_and_grids()

        self.__c_ext_ind = 0    # I define these 3 because I may loosen the ordering restriction in the future
        self.__c_sca_ind = 1
        self.__g_ind = 2
        self.c_extinction, self.c_scattering, self.g = self.__read_aerosol_file()
        self.__check_aerosol_properties_are_plausible()

    def __check_properties_and_grids(self):
        particle_size_checker = ParameterChecker(self.particle_size_grid)
        particle_size_checker.check()
        wavelength_checker = ParameterChecker(self.wavelength_grid)
        wavelength_checker.check()
        properties_checker = AerosolPropertiesChecker(self.aerosol_properties, self.particle_size_grid,
                                                      self.wavelength_grid)
        properties_checker.check()

    def __read_aerosol_file(self):
        c_extinction = np.take(self.aerosol_properties, self.__c_ext_ind, axis=-1)
        c_scattering = np.take(self.aerosol_properties, self.__c_sca_ind, axis=-1)
        g = np.take(self.aerosol_properties, self.__g_ind, axis=-1)
        return c_extinction, c_scattering, g

    def __check_aerosol_properties_are_plausible(self):
        c_ext_checker = ParameterChecker(self.c_extinction)
        c_ext_checker.check()
        c_sca_checker = ParameterChecker(self.c_scattering)
        c_sca_checker.check()
        g_checker = ParameterChecker(self.g)
        g_checker.check()
        g_checker.check_ndarray_is_in_range(0, 1)


class Aerosol(AerosolProperties):
    """ Create a class to hold all of the information about an aerosol"""
    def __init__(self, aerosol_properties, particle_sizes, wavelengths, reference_wavelengths, particle_size_grid=None,
                 wavelength_grid=None):
        """ Initialize the class to hold all the aerosol's properties

        Parameters
        ----------
        aerosol_properties: np.ndarray
            An array containing the aerosol's properties
        particle_sizes: np.ndarray
            The particle sizes of this aerosol
        wavelengths: np.ndarray
            The wavelengths at which this aerosol was observed
        reference_wavelengths: np.ndarray
            The wavelength at which to scale the wavelengths
        particle_size_grid: np.ndarray
            1D array of the particle sizes corresponding to aerosol_properties. Default is None
        wavelength_grid: np.ndarray
            1D array of the wavelengths corresponding to aerosol_properties. Default is None
        """
        super().__init__(aerosol_properties, particle_size_grid=particle_size_grid, wavelength_grid=wavelength_grid)
        self.particle_sizes = particle_sizes
        self.wavelengths = wavelengths
        self.__aerosol_dimensions = np.ndim(self.aerosol_properties)
        self.__particle_none = self.__check_if_input_is_none(self.particle_size_grid)
        self.__wavelength_none = self.__check_if_input_is_none(self.wavelength_grid)
        self.reference_wavelengths = reference_wavelengths

        self.__check_parameters()
        self.__inform_if_outside_wavelength_range()

        self.spectral_extinction = self.__interpolate_parameter_onto_spectral_grid(self.c_extinction, self.wavelengths)
        self.reference_extinction = self.__interpolate_parameter_onto_spectral_grid(self.c_extinction,
                                                                                    self.reference_wavelengths)
        self.extinction = np.divide.outer(self.spectral_extinction, self.reference_extinction).T
        #self.asymmetry_parameter = self.__make_asymmetry_parameter()
        #self.wavelength_scattering = self.__calculate_scattering(self.wavelengths)
        #self.single_scattering_albedo = self.wavelength_scattering / self.wavelength_extinction
        #self.asymmetry_parameter = self.__calculate_asymmetry_parameter(self.wavelengths)

    @staticmethod
    def __check_if_input_is_none(ndarray):
        return True if ndarray is None else False

    def __check_parameters(self):
        particle_size_checker = ParameterChecker(self.particle_sizes)
        particle_size_checker.check()
        wavelength_checker = ParameterChecker(self.wavelengths)
        wavelength_checker.check()
        reference_wavelength_checker = ParameterChecker(self.reference_wavelengths)
        reference_wavelength_checker.check()
        if np.shape(self.particle_sizes) != np.shape(self.reference_wavelengths):
            raise IndexError('particle_sizes and reference_wavelengths must have the same shape')

    def __inform_if_outside_wavelength_range(self):
        if not self.__wavelength_none:
            if np.size((too_short := self.wavelengths[self.wavelengths < self.wavelength_grid[0]]) != 0):
                print('The following input wavelengths: {} microns are shorter than {:.3f} microns---the shortest '
                      'wavelength in the file. Using properties from that wavelength.'
                      .format(too_short, self.wavelength_grid[0]))
            if np.size((too_long := self.wavelengths[self.wavelengths > self.wavelength_grid[-1]]) != 0):
                print('The following input wavelengths: {} microns are longer than {:.1f} microns---the shortest '
                      'wavelength in the file. Using properties from that wavelength.'
                      .format(too_long, self.wavelength_grid[-1]))

    def __interpolate_parameter_onto_spectral_grid(self, parameter, new_spectral_grid):
        return np.interp(new_spectral_grid, self.wavelength_grid, parameter)

    def __calculate_asymmetry_parameter(self, new_location, grid):
        return np.interp(new_location, grid, self.g)

    def __make_asymmetry_parameter(self):
        if self.__aerosol_dimensions == 2 and self.__particle_none:
            g = self.__calculate_asymmetry_parameter(self.wavelengths, self.wavelength_grid)
            return np.broadcast_to(g, (len(self.particle_sizes), len(self.wavelengths)))
        elif self.__aerosol_dimensions == 2 and self.__wavelength_none:
            g = self.__calculate_asymmetry_parameter(self.particle_sizes, self.particle_size_grid)
            print(g)
            return np.broadcast_to(g, (len(self.particle_sizes), len(self.wavelengths)))
