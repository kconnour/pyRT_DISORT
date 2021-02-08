# 3rd-party imports
import numpy as np
from scipy.interpolate import interp2d

# Local imports
from pyRT_DISORT.untested import ForwardScatteringPropertyCollection
from pyRT_DISORT.untested import ModelGrid
from pyRT_DISORT.untested import LegendreCoefficients, HenyeyGreenstein, \
    TabularLegendreCoefficients
from pyRT_DISORT.untested_utils.utilities import ArrayChecker


class Column:
    """ A Column object creates a single column from an input vertical profile """
    def __init__(self, forward_scattering_properties, model_grid, mixing_ratio_profile, particle_size_profile,
                 wavelengths, reference_wavelength, column_integrated_optical_depth, legendre_coefficients):
        """
        Parameters
        ----------
        forward_scattering_properties: ForwardScatteringPropertyCollection
            An aerosol's forward scattering to use for this column
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
            Any instance of a class derived from LegendreCoefficients

        Attributes
        ----------
        forward_scattering_properties: ForwardScatteringPropertyCollection
            The input forward_scattering_properties
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
        self.forward_scattering_properties = forward_scattering_properties
        self.model_grid = model_grid
        self.mixing_ratio_profile = mixing_ratio_profile
        self.particle_size_profile = particle_size_profile
        self.wavelengths = wavelengths
        self.reference_wavelength = reference_wavelength
        self.OD = column_integrated_optical_depth
        self.legendre_coefficients = legendre_coefficients

        self.check_inputs_are_physical()

        self.c_scattering = self.__interpolate_radiative_property_to_model_grid(
            self.forward_scattering_properties.c_scattering)
        self.c_extinction = self.__interpolate_radiative_property_to_model_grid(
            self.forward_scattering_properties.c_extinction)
        self.single_scattering_albedo = self.c_scattering / self.c_extinction
        self.extinction_profile = self.__make_extinction_profile()
        self.extinction = self.__make_extinction()
        self.total_optical_depth = self.__calculate_total_optical_depth()
        self.scattering_optical_depth = self.__calculate_scattering_optical_depth()
        self.phase_function = self.__make_phase_function()

    def check_inputs_are_physical(self):
        self.__check_forward_scattering_properties_is_ForwardScatteringPropertyCollection()
        self.__check_forward_scattering_properties_have_expected_attributes()
        self.__check_model_grid_is_ModelGrid()
        self.__check_mixing_ratio_profile_is_physical()
        self.__check_particle_size_profile_is_physical()
        self.__check_profiles_match_layers()
        self.__check_wavelengths_are_physical()
        self.__check_reference_wavelength_is_physical()
        self.__check_column_integrated_optical_depth_is_physical()
        self.__check_legendre_coefficients_is_LegendreCoefficients()

    def __check_forward_scattering_properties_is_ForwardScatteringPropertyCollection(self):
        if not isinstance(self.forward_scattering_properties, ForwardScatteringPropertyCollection):
            raise TypeError('forward_scattering_properties must be an instance of ForwardScatteringPropertyCollection.')

    def __check_forward_scattering_properties_have_expected_attributes(self):
        if not hasattr(self.forward_scattering_properties, 'c_scattering'):
            raise AttributeError('forward_scattering_properties must have attribute c_scattering')
        if not hasattr(self.forward_scattering_properties, 'c_extinction'):
            raise AttributeError('forward_scattering_properties must have attribute c_extinction')

    def __check_model_grid_is_ModelGrid(self):
        if not isinstance(self.model_grid, ModelGrid):
            raise TypeError('model_grid must be an instance of ModelGrid.')

    def __check_mixing_ratio_profile_is_physical(self):
        mixing_ratio_checker = ArrayChecker(self.mixing_ratio_profile, 'mixing_ratio_profile')
        mixing_ratio_checker.check_object_is_array()
        mixing_ratio_checker.check_ndarray_is_numeric()
        mixing_ratio_checker.check_ndarray_is_non_negative()
        mixing_ratio_checker.check_ndarray_is_finite()
        mixing_ratio_checker.check_ndarray_is_1d()

    def __check_particle_size_profile_is_physical(self):
        size_checker = ArrayChecker(self.particle_size_profile, 'particle_size_profile')
        size_checker.check_object_is_array()
        size_checker.check_ndarray_is_numeric()
        size_checker.check_ndarray_is_positive_finite()
        size_checker.check_ndarray_is_1d()

    def __check_profiles_match_layers(self):
        if self.model_grid.column_density_layers.shape != self.mixing_ratio_profile.shape:
            raise IndexError('mixing_ratio_profile must be the same length as the number of layers in the model')
        if self.model_grid.column_density_layers.shape != self.particle_size_profile.shape:
            raise IndexError('particle_size_profile must be the same length as the number of layers in the model')

    def __check_wavelengths_are_physical(self):
        wavelength_checker = ArrayChecker(self.wavelengths, 'wavelengths')
        wavelength_checker.check_object_is_array()
        wavelength_checker.check_ndarray_is_numeric()
        wavelength_checker.check_ndarray_is_positive_finite()
        wavelength_checker.check_ndarray_is_1d()

    def __check_reference_wavelength_is_physical(self):
        if not isinstance(self.reference_wavelength, (int, float)):
            raise TypeError('reference_wavelength must be an int or float')
        if not np.isfinite(self.reference_wavelength):
            raise ValueError('reference_wavelength must be finite')
        if self.reference_wavelength <= 0:
            raise ValueError('reference_wavelength must be positive')

    def __check_column_integrated_optical_depth_is_physical(self):
        if not isinstance(self.OD, (int, float)):
            raise TypeError('column_integrated_optical_depth must be an int or float')
        if not np.isfinite(self.reference_wavelength):
            raise ValueError('column_integrated_optical_depth must be finite')
        if self.OD <= 0:
            raise ValueError('column_integrated_optical_depth must be positive')

    def __check_legendre_coefficients_is_LegendreCoefficients(self):
        if not isinstance(self.legendre_coefficients, LegendreCoefficients):
            raise TypeError('legendre_coefficients must be a derived instance of LegendreCoefficients')

    def __interpolate_radiative_property_to_model_grid(self, forward_scattering_property):
        if len(forward_scattering_property.property_values) == 1:
            return np.ones((len(self.particle_size_profile), len(self.wavelengths))) * \
                   forward_scattering_property.property_values
        elif forward_scattering_property.wavelength_grid is None:
            interp_grid = np.interp(self.particle_size_profile, forward_scattering_property.particle_size_grid,
                                    forward_scattering_property.property_values)
            return np.broadcast_to(interp_grid, (len(self.wavelengths), len(self.particle_size_profile))).T
        elif forward_scattering_property.particle_size_grid is None:
            interp_grid = np.interp(self.wavelengths, forward_scattering_property.wavelength_grid,
                                    forward_scattering_property.property_values)
            return np.broadcast_to(interp_grid, (len(self.particle_size_profile), len(self.wavelengths)))
        else:
            f = interp2d(forward_scattering_property.particle_size_grid, forward_scattering_property.wavelength_grid,
                         forward_scattering_property.property_values.T)
            return f(self.particle_size_profile, self.wavelengths).T

    def __make_extinction_profile(self):
        c_extinction = self.forward_scattering_properties.c_extinction
        if c_extinction.scattering_property_is_singleton:
            return np.ones(len(self.particle_size_profile)) * c_extinction.property_values
        elif c_extinction.wavelength_grid is None:
            return np.interp(self.particle_size_profile, c_extinction.particle_size_grid, c_extinction.property_values)
        elif c_extinction.particle_size_grid is None:
            return np.interp(self.reference_wavelength, c_extinction.wavelength_grid, c_extinction.property_values) * \
                   np.ones(len(self.particle_size_profile))
        else:
            f = interp2d(c_extinction.particle_size_grid, c_extinction.wavelength_grid, c_extinction.property_values.T)
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
            return self.__make_hg_phase_function()
        elif isinstance(self.legendre_coefficients, TabularLegendreCoefficients):
            return self.__make_tabular_phase_function()
        else:
            raise TypeError('I have no idea how to handle this type of phase function yet...')

    def __make_hg_phase_function(self):
        if not hasattr(self.forward_scattering_properties, 'g'):
            raise AttributeError('forward_scattering_properties has no attribute "g"')
        unnormalized_coefficients = self.legendre_coefficients.make_legendre_coefficients(
            self.forward_scattering_properties.g.property_values)
        return self.__make_normalized_coefficients(unnormalized_coefficients)

    @staticmethod
    def __make_normalized_coefficients(unnormalized_coefficients):
        n_moments = unnormalized_coefficients.shape[0]
        normalization = np.linspace(0, n_moments-1, num=n_moments)*2 + 1
        return (unnormalized_coefficients.T / normalization).T

    def __make_tabular_phase_function(self):
        if self.legendre_coefficients.coefficients_dimensions == 1:
            unnormalized_coefficients = np.broadcast_to(self.legendre_coefficients.tabulated_coefficients,
                                                        (len(self.wavelengths), self.model_grid.n_layers,
                                                         len(self.legendre_coefficients.tabulated_coefficients))).T
            return self.__make_normalized_coefficients(unnormalized_coefficients)
        elif self.legendre_coefficients.coefficients_dimensions == 2:
            shape = self.legendre_coefficients.tabulated_coefficients.shape
            if self.legendre_coefficients.wavelengths is None:
                expanded_coeff = np.broadcast_to(self.legendre_coefficients.tabulated_coefficients.T,
                                                 (len(self.wavelengths), shape[1], shape[0])).T
            elif self.legendre_coefficients.particle_sizes is None:
                expanded_coeff = np.moveaxis(np.broadcast_to(self.legendre_coefficients.tabulated_coefficients,
                                                             (len(self.wavelengths), shape[0], shape[1])), 0, 1)
            unnormalized_coefficients = self.__get_nearest_neighbor_phase_functions(expanded_coeff)
            return self.__make_normalized_coefficients(unnormalized_coefficients)
        elif self.legendre_coefficients.coefficients_dimensions == 3:
            unnormalized_coefficients = self.__get_nearest_neighbor_phase_functions(
                self.legendre_coefficients.tabulated_coefficients)
            return self.__make_normalized_coefficients(unnormalized_coefficients)

    def __get_nearest_neighbor_phase_functions(self, coefficients):
        radius_indices = self.__get_nearest_indices(self.particle_size_profile,
                                                    self.legendre_coefficients.particle_sizes)
        wavelength_indices = self.__get_nearest_indices(self.wavelengths, self.legendre_coefficients.wavelengths)
        return coefficients[:, radius_indices, :][:, :, wavelength_indices]

    @staticmethod
    def __get_nearest_indices(values, array):
        diff = (values.reshape(1, -1) - array.reshape(-1, 1))
        indices = np.abs(diff).argmin(axis=0)
        return indices
