# 3rd-party imports
import numpy as np
from scipy.interpolate import interp2d

# Local imports
from pyRT_DISORT.preprocessing.model.aerosol import AerosolProperties
from pyRT_DISORT.preprocessing.model.atmosphere import ModelGrid
from pyRT_DISORT.preprocessing.model.new_phase_function import LegendreCoefficients, HenyeyGreenstein, TabularLegendreCoefficients
from pyRT_DISORT.preprocessing.utilities.array_checks import ArrayChecker


class Column:
    """ A Column object creates a single column from an input vertical profile """
    def __init__(self, aerosol_properties, model_grid, mixing_ratio_profile, particle_size_profile, wavelengths,
                 reference_wavelength, column_integrated_optical_depth, legendre_coefficients, debug=False):
        """
        Parameters
        ----------
        aerosol_properties: AerosolProperties
            An instance of AerosolProperties to use for this column
        model_grid: ModelGrid
            An instance of ModelGrid to set the structure of the model
        mixing_ratio_profile: np.ndarray
            The vertical mixing ratio profile to constrain Column
        particle_size_profile: np.ndarray
            The vertical profile of particle sizes in each layer
        wavelengths: np.ndarray
            The wavelengths where to construct this column
        reference_wavelength: int or float
            The reference wavelength
        column_integrated_optical_depth: int or float
            The total column integrated optical depth of this column
        legendre_coefficients: LegendreCoefficients
            A class derived from LegendreCoefficients
        debug: bool, optional
            Denote if this code should perform sanity checks on inputs to see if they're expected. Default is False

        Attributes
        ----------
        aerosol_properties: AerosolProperties
            The input aerosol_properties
        model_grid: ModelGrid
            The input model_grid
        mixing_ratio_profile: np.ndarray
            The input mixing_ratio_profile
        particle_size_profile: np.ndarray
            The input particle_size_profile
        wavelengths: np.ndarray
            The input wavelengths
        reference_wavelength: int or float
            The input reference_wavelength
        OD: int or float
            The input column_integrated_optical_depth
        c_extinction: np.ndarray
            2D array of the extinction coefficient at the altitudes and wavelengths of the model
        c_scattering: np.ndarray
            2D array of the scattering coefficient at the altitudes and wavelengths of the model
        g: np.ndarray
            2D array of the asymmetry parameter at the altitudes and wavelengths of the model
        single_scattering_albedo: np.ndarray
            2D array of c_scattering / c_extinction
        extinction_profile: np.ndarray
            1D array of the extinction coefficient at the model altitudes and reference wavelength
        extinction: np.ndarray
            2D array of c_extinction / extinction_profile
        total_optical_depths: np.ndarray
            2D array of the total optical depths in each layer at each wavelength
        scattering_optical_depth: np.ndarray
            2D array of the scattering optical depths in each layer at each wavelength
        """
        self.aerosol_properties = aerosol_properties
        self.model_grid = model_grid
        self.mixing_ratio_profile = mixing_ratio_profile
        self.particle_size_profile = particle_size_profile
        self.wavelengths = wavelengths
        self.reference_wavelength = reference_wavelength
        self.legendre_coefficients = legendre_coefficients
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
        self.phase_function = self.__make_phase_function()

    def __check_inputs_are_feasible(self):
        self.__check_properties()
        self.__check_model_grid()
        self.__check_mixing_ratio_profile()
        self.__check_particle_size_profile()
        self.__check_wavelengths()
        self.__check_reference_wavelength()
        self.__check_column_integrated_optical_depth()
        self.__check_legendre_coefficients()

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
        self.__inform_if_outside_wavelength_range()

    def __inform_if_outside_wavelength_range(self):
        shortest_wavelength = self.aerosol_properties.wavelength_grid[0]
        longest_wavelength = self.aerosol_properties.wavelength_grid[-1]
        if not self.__wavelength_none:
            if np.size((too_short := self.wavelengths[self.wavelengths < shortest_wavelength]) != 0):
                print(
                    f'The following input wavelengths: {too_short} microns are shorter than {shortest_wavelength:.3f} '
                    f'microns---the shortest wavelength in the file. Using properties from that wavelength.')
            if np.size((too_long := self.wavelengths[self.wavelengths > longest_wavelength]) != 0):
                print(
                    f'The following input wavelengths: {too_long} microns are longer than {longest_wavelength:.1f} '
                    f'microns---the longest wavelength in the file. Using properties from that wavelength.')

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

    def __check_legendre_coefficients(self):
        if not isinstance(self.legendre_coefficients, LegendreCoefficients):
            raise TypeError('legendre_coefficients must be an instance of a derived class of LegendreCoefficients')

    def __make_radiative_properties(self):
        c_extinction = self.__interpolate_radiative_properties__to_model_grid(self.aerosol_properties.c_extinction_grid)
        c_scattering = self.__interpolate_radiative_properties__to_model_grid(self.aerosol_properties.c_scattering_grid)
        g = self.__interpolate_radiative_properties__to_model_grid(self.aerosol_properties.g_grid)
        return c_extinction, c_scattering, g

    def __interpolate_radiative_properties__to_model_grid(self, radiative_property):
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
            return np.ones(len(self.particle_size_profile)) * np.interp(self.reference_wavelength,
                                                                        self.aerosol_properties.wavelength_grid,
                                                                        self.aerosol_properties.c_extinction_grid)
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

    def __make_phase_function(self):
        if isinstance(self.legendre_coefficients, HenyeyGreenstein):
            unnormalized_coefficients = self.legendre_coefficients.make_legendre_coefficients(self.g)
            return self.__make_normalized_coefficients(unnormalized_coefficients)
        elif isinstance(self.legendre_coefficients, TabularLegendreCoefficients):
            if self.legendre_coefficients.coefficients_dimensions == 1:
                unnormalized_coefficients = np.broadcast_to(self.legendre_coefficients.tabulated_coefficients,
                                                            (len(self.wavelengths), self.model_grid.n_layers,
                                                             len(self.legendre_coefficients.tabulated_coefficients))).T
                return self.__make_normalized_coefficients(unnormalized_coefficients)
            elif self.legendre_coefficients.coefficients_dimensions == 2:
                pass
            elif self.legendre_coefficients.coefficients_dimensions == 3:
                pass
        else:
            raise TypeError('I have no idea how to handle this type of phase function yet...')

    @staticmethod
    def __make_normalized_coefficients(unnormalized_coefficients):
        n_moments = unnormalized_coefficients.shape[0]
        normalization = np.linspace(0, n_moments-1, num=n_moments)*2 + 1
        return (unnormalized_coefficients.T / normalization).T


class SporadicParticleSizes:
    """ This class interpolates particle sizes on a sporadic altitude grid to the model altitude grid"""
    def __init__(self, z_size_grid, model_grid, debug=False):
        """
        Parameters
        ----------
        z_size_grid: np.ndarray
            An Nx2 array where the first column are altitudes and the second column are the associated particle sizes
        model_grid: ModelGrid
            Model structure to interpolate particle sizes on to
        debug: bool, optional
            Denote if this code should perform sanity checks on inputs to see if they're expected. Default is False

        Attributes
        ----------
        z_size_grid: np.ndarray
            The input z_size_grid
        model_grid: ModelGrid
            The input model_grid
        regridded_particle_sizes: np.ndarray
            The particle sizes regridded to match the number of layers in model_grid
        """
        self.z_size_grid = z_size_grid
        self.model_grid = model_grid

        if debug:
            self.__check_grids()

        self.regridded_particle_sizes = self.__interp_to_new_grid()

    def __check_grids(self):
        z_size_checker = ArrayChecker(self.z_size_grid, 'z_size_grid')
        z_size_checker.check_object_is_array()
        z_size_checker.check_ndarray_is_numeric()
        z_size_checker.check_ndarray_is_non_negative()
        z_size_checker.check_ndarray_is_finite()
        z_size_checker.check_ndarray_is_2d()
        altitude_checker = ArrayChecker(self.z_size_grid[:, 0], 'z_size_grid')
        altitude_checker.check_1d_array_is_monotonically_decreasing()
        if self.z_size_grid.shape[-1] != 2:
            raise IndexError('The second dimension of z_size_grid must be 2')
        if not isinstance(self.model_grid, ModelGrid):
            raise TypeError('model_grid must be an instance of ModelGrid')

    def __interp_to_new_grid(self):
        return np.interp(self.model_grid.altitude_layers, np.flip(self.z_size_grid[:, 0]),
                                 np.flip(self.z_size_grid[:, 1]))
