# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.preprocessing.model.atmosphere import ModelAtmosphere
from pyRT_DISORT.preprocessing.utilities.array_checks import CheckArray


class VerticalProfile:
    def __init__(self, model_atmosphere):
        self.model_atmosphere = model_atmosphere
        self.__check_input_is_atmosphere()

    def __check_input_is_atmosphere(self):
        if not isinstance(self.model_atmosphere, ModelAtmosphere):
            raise TypeError('model_atmosphere must be an instance of ModelAtmosphere')


class Conrath(VerticalProfile):
    """ Construct a Conrath profile"""
    def __init__(self, model_atmosphere, aerosol_scale_height, conrath_nu):
        """

        Parameters
        ----------
        model_atmosphere: ModelAtmosphere
            The model atmosphere
        aerosol_scale_height: np.ndarray
            1D array of the aerosols' scale heights used in the Conrath parameterization [km]
        conrath_nu: np.ndarray
            1D array of the aerosols' nu parameters used in the Conrath parameterization. Must be the same length as
            aerosol_scale_height
        """
        super().__init__(model_atmosphere)
        self.H = aerosol_scale_height
        self.nu = conrath_nu
        self.__check_parameters_are_plausible()
        self.profile = self.__make_conrath_profile()

    def __check_parameters_are_plausible(self):
        self.__check_scale_height_is_plausible()
        self.__check_conrath_nu_is_plausible()

    def __check_scale_height_is_plausible(self):
        scale_height_checker = CheckArray(self.H, 'scale_height')
        scale_height_checker.check_object_is_array()
        scale_height_checker.check_ndarray_is_numeric()
        scale_height_checker.check_ndarray_is_positive_finite()
        scale_height_checker.check_ndarray_is_1d()

    def __check_conrath_nu_is_plausible(self):
        scale_height_checker = CheckArray(self.nu, 'conrath_nu')
        scale_height_checker.check_object_is_array()
        scale_height_checker.check_ndarray_is_numeric()
        scale_height_checker.check_ndarray_is_positive_finite()
        scale_height_checker.check_ndarray_is_1d()

    def __make_conrath_profile(self):
        """Calculate the vertical dust distribution assuming a Conrath profile, i.e.
        q(z) / q(0) = exp( nu * (1 - exp(z / H)))
        where q is the mass mixing ratio

        Returns
        -------
        fractional_mixing_ratio: np.ndarray (len(altitude_layer))
            The fraction of the mass mixing ratio at the midpoint altitudes
        """
        altitude_scale = np.divide.outer(self.model_atmosphere.altitude_layers, self.H)
        return np.exp(self.nu * (1 - np.exp(altitude_scale)))


class Uniform(VerticalProfile):
    """ Construct N uniform profiles between two altitudes"""
    def __init__(self, model_atmosphere, altitude_bottom, altitude_top):
        """

        Parameters
        ----------
        model_atmosphere: ModelAtmosphere
            The model atmosphere
        altitude_bottom: np.ndarray
            1D array of bottom altitudes. Will create N profiles, where N = len(altitude_bottom)
        altitude_top: np.ndarray
            1D array of top altitudes. Must be the same length as altitude_bottom.
        """
        super().__init__(model_atmosphere)
        self.altitude_bottom = altitude_bottom
        self.altitude_top = altitude_top
        self.__check_input_altitudes_are_plausible()
        self.profile = self.__make_profiles()

    def __check_input_altitudes_are_plausible(self):
        self.__check_altitude_bottom()
        self.__check_altitude_top()
        self.__check_altitude_relationships()

    def __check_altitude_bottom(self):
        bottom_altitude_checker = CheckArray(self.altitude_bottom, 'altitude_bottom')
        bottom_altitude_checker.check_object_is_array()
        bottom_altitude_checker.check_ndarray_is_numeric()
        bottom_altitude_checker.check_ndarray_is_finite()
        bottom_altitude_checker.check_ndarray_is_non_negative()
        bottom_altitude_checker.check_ndarray_is_1d()

    def __check_altitude_top(self):
        top_altitude_checker = CheckArray(self.altitude_top, 'altitude_top')
        top_altitude_checker.check_object_is_array()
        top_altitude_checker.check_ndarray_is_numeric()
        top_altitude_checker.check_ndarray_is_positive_finite()
        top_altitude_checker.check_ndarray_is_1d()

    def __check_altitude_relationships(self):
        if not self.altitude_bottom.shape == self.altitude_top.shape:
            raise ValueError('altitude_bottom and altitude_top must have the same shape')
        if not np.all(self.altitude_top > self.altitude_bottom):
            raise ValueError('altitude_top must be greater than altitude_bottom in each index')

    def __make_profiles(self):
        # SHIELD YOUR EYES AND LOOK AWAY FROM THIS MONSTROSITY OF A METHOD
        expanded_altitude_layers = np.broadcast_to(self.model_atmosphere.model_altitudes,
                                                   (len(self.altitude_bottom), self.model_atmosphere.n_boundaries)).T

        # Get the difference between the altitudes and the target bottom altitudes
        boundary_bottoms = np.flipud(expanded_altitude_layers - self.altitude_bottom)
        # Get the first index where the altitude grid surpasses the bottom altitude
        bottom_indices = np.argmax(boundary_bottoms > 0, axis=0)
        # Get the fraction of the grid that's within the target value
        bottom_frac = (np.flip(self.model_atmosphere.model_altitudes)[bottom_indices] - self.altitude_bottom) / \
                       np.diff(np.flip(self.model_atmosphere.model_altitudes), axis=0)[bottom_indices]

        boundary_tops = np.flipud(expanded_altitude_layers - self.altitude_top)
        top_indices = np.argmax(boundary_tops > 0, axis=0)
        top_frac = np.abs((np.flip(self.model_atmosphere.model_altitudes)[top_indices] - self.altitude_top) /
                      np.diff(np.flip(self.model_atmosphere.model_altitudes), axis=0)[top_indices] - 1)

        # The profile is constant (1) except where it's 0 or fractional
        profiles = np.ones((self.model_atmosphere.n_layers, len(self.altitude_bottom)))
        for i in range(len(self.altitude_bottom)):
            profiles[:bottom_indices[i]-1, i] = 0
            profiles[top_indices[i]-1:, i] = 0
            profiles[bottom_indices[i]-1, i] = bottom_frac[i]
            profiles[top_indices[i]-1, i] = top_frac[i]

        # Check the profile cannot be out of range...
        profiles = np.where(np.abs(profiles) > 1, 1, profiles)

        return np.flipud(profiles)


class Layers(VerticalProfile):
    """ Construct atmospheric layers from the layer indices"""
    def __init__(self, model_atmosphere, layer_indices):
        """

        Parameters
        ----------
        model_atmosphere: ModelAtmosphere
            The model atmospheric
        layer_indices: float or np.ndarray
            The index or indices where to create a uniform profile. If input is not a float, this class will expect
            a np.ndarray
        """
        super().__init__(model_atmosphere)
        self.layer_indices = layer_indices
        self.__check_indices()
        self.profile = self.__make_profile()

    def __check_indices(self):
        if isinstance(self.layer_indices, int):
            return
        else:
            indices_checker = CheckArray(self.layer_indices, 'layer_indices')
            indices_checker.check_object_is_array()
            indices_checker.check_ndarray_contains_only_ints()
            indices_checker.check_ndarray_is_in_range(-self.model_atmosphere.n_layers, self.model_atmosphere.n_layers)
            indices_checker.check_ndarray_is_1d()
            indices_checker.check_1d_array_is_no_longer_than(self.model_atmosphere.n_layers)

    def __make_profile(self):
        profile = np.zeros(self.model_atmosphere.n_layers)
        profile[self.layer_indices] = 1
        return profile


class ProfileHolder:
    """ Make a class to hold on to vertical profiles created elsewhere """
    def __init__(self):
        self.__profiles = []
        self.profile = None

    def add_profile(self, profile):
        """ Add a profile to the object

        Parameters
        ----------
        profile: np.ndarray
            A 1D or 2D array of vertical profiles. All profiles added with this method should have the same first
            dimension

        Returns
        -------
        None
        """
        checked_profile = self.__make_arrays_2d(profile)
        self.__profiles.append(checked_profile)

    @staticmethod
    def __make_arrays_2d(profile):
        if np.ndim(profile) >= 3:
            raise ValueError('Whoa there, I dunno what to do with a profile that\'s greater than 2D')
        elif np.ndim(profile) == 1:
            return profile[:, np.newaxis]
        else:
            return profile

    def stack_profiles(self):
        """ Stack all profiles into a single 2D array. You can use this at any time but it's only intended to be used
        after all profiles have been added via .add_profile()

        Returns
        -------
        None
        """
        self.profile = np.concatenate(self.__profiles, axis=-1)
