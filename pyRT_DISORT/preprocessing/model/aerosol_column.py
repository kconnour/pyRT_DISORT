# 3rd-party imports
import numpy as np
from scipy.interpolate import interp2d

# Local imports
from pyRT_DISORT.preprocessing.model.atmosphere import ModelGrid
from pyRT_DISORT.preprocessing.utilities.array_checks import ArrayChecker


class AerosolProperties:
    def __init__(self, aerosol_properties, particle_size_grid=None, wavelength_grid=None, debug=False):
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

    def __get_properties_dimensions(self):
        return np.ndim(self.aerosol_properties)

    def __get_properties_shape(self):
        return np.shape(self.aerosol_properties)

    @staticmethod
    def __check_if_input_is_none(ndarray):
        return True if ndarray is None else False

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
            raise TypeError('You need to include both particle_size_grid and wavelength_grid')
        if self.__array_shape[0] != len(self.particle_size_grid):
            raise IndexError(
                'For 3D files, aerosol_properties\' first dimension must be the same length as particle_size_grid')
        if self.__array_shape[1] != len(self.wavelength_grid):
            raise IndexError(
                    'For 3D files, aerosol_properties\' second dimension must be the same length as wavelength_grid')

    def __check_radiative_properties(self):
        self.__check_radiative_property_is_plausible(self.c_extinction_grid, 'c_extinction')
        self.__check_radiative_property_is_plausible(self.c_scattering_grid, 'c_scattering')
        self.__check_radiative_property_is_plausible(self.g_grid, 'g')

    @staticmethod
    def __check_radiative_property_is_plausible(radiative_property, property_name):
        grid_checker = ArrayChecker(radiative_property, property_name)
        grid_checker.check_ndarray_is_numeric()
        grid_checker.check_ndarray_is_positive_finite()

    def __read_aerosol_file(self):
        c_extinction = np.take(self.aerosol_properties, self.__c_ext_ind, axis=-1)
        c_scattering = np.take(self.aerosol_properties, self.__c_sca_ind, axis=-1)
        g = np.take(self.aerosol_properties, self.__g_ind, axis=-1)
        return c_extinction, c_scattering, g


class Column:
    def __init__(self, aerosol_properties, model_grid, mixing_ratio_profile, particle_size_profile, wavelengths,
                 reference_wavelength, column_integrated_optical_depth, debug=False):
        self.aerosol_properties = aerosol_properties
        self.model_grid = model_grid
        self.mixing_ratio_profile = mixing_ratio_profile
        self.particle_size_profile = particle_size_profile
        self.wavelengths = wavelengths
        self.reference_wavelength = reference_wavelength
        self.OD = column_integrated_optical_depth
        self.__particle_none = self.aerosol_properties.particle_none
        self.__wavelength_none = self.aerosol_properties.wavelength_none

        if debug:
            self.__check_inputs_are_feasible()

        self.c_extinction, self.c_scattering, self.g = self.__make_radiative_properties()
        self.single_scattering_albedo = self.c_scattering / self.c_extinction
        self.extinction_profile = self.__make_extinction_profile()
        self.extinction = self.__make_extinction()
        self.total_optical_depth = self.__calculate_total_optical_depth()
        self.scatting_optical_depth = self.__calculate_scattering_optical_depth()

    def __check_inputs_are_feasible(self):
        self.__check_properties()
        self.__check_model_grid()
        self.__check_mixing_ratio_profile()
        self.__check_particle_size_profile()
        self.__check_wavelengths()
        self.__check_reference_wavelength()
        self.__check_column_integrated_optical_depth()

    def __check_properties(self):
        if not isinstance(self.aerosol_properties, AerosolProperties):
            raise TypeError('aerosol_properties needs to be an instance of AerosolProperties.')

    def __check_model_grid(self):
        if not isinstance(self.model_grid, ModelGrid):
            raise TypeError('model_grid needs to be an instance of ModelGrid.')

    def __check_mixing_ratio_profile(self):
        mixing_ratio_checker = ArrayChecker(self.mixing_ratio_profile, 'mixing_ratio_profile')
        mixing_ratio_checker.check_object_is_array()
        mixing_ratio_checker.check_ndarray_is_numeric()
        mixing_ratio_checker.check_ndarray_is_non_negative()
        mixing_ratio_checker.check_ndarray_is_finite()
        mixing_ratio_checker.check_ndarray_is_1d()

    def __check_particle_size_profile(self):
        size_checker = ArrayChecker(self.particle_size_profile, 'particle_size_profile')
        size_checker.check_object_is_array()
        size_checker.check_ndarray_is_numeric()
        size_checker.check_ndarray_is_positive_finite()
        size_checker.check_ndarray_is_1d()
        if self.model_grid.column_density_layers.shape != self.mixing_ratio_profile.shape:
            raise IndexError('mixing_ratio_profile must be the same length as the number of layers in the model')
        if self.mixing_ratio_profile.shape != self.particle_size_profile.shape:
            raise IndexError('mixing_ratio_profile must be the same shape as particle_size_profile')

    def __check_wavelengths(self):
        wavelength_checker = ArrayChecker(self.wavelengths, 'wavelengths')
        wavelength_checker.check_object_is_array()
        wavelength_checker.check_ndarray_is_numeric()
        wavelength_checker.check_ndarray_is_positive_finite()
        wavelength_checker.check_ndarray_is_1d()

    def __check_reference_wavelength(self):
        if not isinstance(self.reference_wavelength, (int, float)):
            raise TypeError('reference_wavelength must be an int or float')
        if not np.isfinite(self.reference_wavelength):
            raise ValueError('reference_wavelength must be finite')
        if self.reference_wavelength <= 0:
            raise ValueError('reference_wavelength must be positive')

    def __check_column_integrated_optical_depth(self):
        if not isinstance(self.OD, (int, float)):
            raise TypeError('column_integrated_optical_depth must be an int or float')
        if not np.isfinite(self.reference_wavelength):
            raise ValueError('column_integrated_optical_depth must be finite')
        if self.OD <= 0:
            raise ValueError('column_integrated_optical_depth must be positive')

    def __make_radiative_properties(self):
        c_extinction = self.__construct_radiative_properties_grid(self.aerosol_properties.c_extinction_grid)
        c_scattering = self.__construct_radiative_properties_grid(self.aerosol_properties.c_scattering_grid)
        g = self.__construct_radiative_properties_grid(self.aerosol_properties.g_grid)
        return c_extinction, c_scattering, g

    def __construct_radiative_properties_grid(self, radiative_property):
        if self.__wavelength_none:
            interp_grid = np.interp(self.particle_size_profile, self.aerosol_properties.particle_size_grid,
                                    radiative_property)
            return np.broadcast_to(interp_grid, (len(self.wavelengths), len(self.particle_size_profile))).T
        elif self.__particle_none:
            interp_grid = np.interp(self.wavelengths, self.aerosol_properties.wavelength_grid, radiative_property)
            return np.broadcast_to(interp_grid, (len(self.particle_size_profile), len(self.wavelengths)))
        else:
            f = interp2d(self.aerosol_properties.particle_size_grid, self.aerosol_properties.wavelength_grid,
                         radiative_property.T)
            return f(self.particle_size_profile, self.wavelengths).T

    def __make_extinction_profile(self):
        if self.__wavelength_none:
            return np.interp(self.particle_size_profile, self.aerosol_properties.particle_size_grid,
                                    self.aerosol_properties.c_extinction_grid)
        elif self.__particle_none:
            print('aerosol_properties has no particle size info. Using the same value for all points in extinction')
            return np.ones(len(self.particle_size_profile))
        else:
            f = interp2d(self.aerosol_properties.particle_size_grid, self.aerosol_properties.wavelength_grid,
                         self.aerosol_properties.c_extinction_grid.T)
            return np.squeeze(f(self.particle_size_profile, self.reference_wavelength).T)

    def __make_extinction(self):
        return (self.c_extinction.T / self.extinction_profile).T

    def __calculate_total_optical_depth(self):
        normalization = np.sum(self.mixing_ratio_profile * self.model_grid.column_density_layers)
        od_profile = self.mixing_ratio_profile * self.model_grid.column_density_layers * self.OD / normalization
        return (od_profile * self.extinction.T).T

    def __calculate_scattering_optical_depth(self):
        return self.total_optical_depth * self.single_scattering_albedo
