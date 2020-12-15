# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.model_atmosphere.atmosphere_grid import ModelGrid
from pyRT_DISORT.utilities.array_checks import ArrayChecker


class VerticalProfile:
    """A VerticalProfile object is an abstract class to perform basic checks on vertical profiles"""
    def __init__(self, model_grid):
        self.model_grid = model_grid
        self.__check_input_is_model_grid()

    def __check_input_is_model_grid(self):
        if not isinstance(self.model_grid, ModelGrid):
            raise TypeError('model_grid must be an instance of ModelGrid')


class Conrath(VerticalProfile):
    """A Conrath object constructs a Conrath vertical profile"""
    def __init__(self, model_grid, scale_height, conrath_nu):
        """
        Parameters
        ----------
        model_grid: ModelGrid
            The model atmosphere
        scale_height: np.ndarray
            1D array of the aerosols' scale heights used in the Conrath parameterization [km]
        conrath_nu: np.ndarray
            1D array of the aerosols' nu parameters used in the Conrath parameterization. Must be the same length as
            aerosol_scale_height

        Attributes
        ----------
        model_grid: ModelGrid
            The input model_grid
        scale_height: int or float
            The input aerosol_scale_height
        conrath_nu: float
            The input conrath_nu
        profile: np.ndarray
            The mixing ratio vertical profile
        """
        super().__init__(model_grid)
        self.H = scale_height
        self.nu = conrath_nu
        self.__check_parameters_are_plausible()
        self.profile = self.__make_conrath_profile()

    def __check_parameters_are_plausible(self):
        self.__check_scale_height_is_plausible()
        self.__check_conrath_nu_is_plausible()
        self.__check_parameters_have_same_shapes()

    def __check_scale_height_is_plausible(self):
        scale_height_checker = ArrayChecker(self.H, 'scale_height')
        scale_height_checker.check_object_is_array()
        scale_height_checker.check_ndarray_is_numeric()
        scale_height_checker.check_ndarray_is_positive_finite()
        scale_height_checker.check_ndarray_is_1d()

    def __check_conrath_nu_is_plausible(self):
        scale_height_checker = ArrayChecker(self.nu, 'conrath_nu')
        scale_height_checker.check_object_is_array()
        scale_height_checker.check_ndarray_is_numeric()
        scale_height_checker.check_ndarray_is_positive_finite()
        scale_height_checker.check_ndarray_is_1d()

    def __check_parameters_have_same_shapes(self):
        if self.H.shape != self.nu.shape:
            raise ValueError('scale_height and conrath_nu must have the same shapes')

    def __make_conrath_profile(self):
        """Calculate the vertical dust distribution assuming a Conrath profile, i.e.
        q(z) / q(0) = exp( nu * (1 - exp(z / H)))
        where q is the mass mixing ratio

        Returns
        -------
        fractional_mixing_ratio: np.ndarray (len(altitude_layer))
            The fraction of the mass mixing ratio at the midpoint altitudes
        """
        altitude_scale = np.divide.outer(self.model_grid.layer_altitudes, self.H)
        return np.exp(self.nu * (1 - np.exp(altitude_scale)))


class Uniform(VerticalProfile):
    """A Uniform object constructs a profile with uniform mixing ratio between two altitudes"""
    def __init__(self, model_grid, altitude_bottom, altitude_top):
        """
        Parameters
        ----------
        model_grid: ModelGrid
            The model atmosphere
        altitude_bottom: int or float
            The bottom altitude of this profile [km]
        altitude_top: int or float
            The top altitude of this profile [km]
        """
        super().__init__(model_grid)
        self.altitude_bottom = altitude_bottom
        self.altitude_top = altitude_top
        self.__check_input_altitudes_are_plausible()
        self.profile = self.__make_profiles()

    def __check_input_altitudes_are_plausible(self):
        if not isinstance(self.altitude_bottom, (int, float)):
            raise ValueError('altitude_bottom must be an int or float')
        if not isinstance(self.altitude_top, (int, float)):
            raise ValueError('altitude_top must be an int or float')
        if self.altitude_bottom >= self.altitude_top:
            raise ValueError('altitude_top must be greater than altitude_bottom')

    def __check_altitude_bottom(self):
        bottom_altitude_checker = ArrayChecker(self.altitude_bottom, 'altitude_bottom')
        bottom_altitude_checker.check_object_is_array()
        bottom_altitude_checker.check_ndarray_is_numeric()
        bottom_altitude_checker.check_ndarray_is_finite()
        bottom_altitude_checker.check_ndarray_is_non_negative()
        bottom_altitude_checker.check_ndarray_is_1d()

    def __check_altitude_top(self):
        top_altitude_checker = ArrayChecker(self.altitude_top, 'altitude_top')
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
            indices_checker = ArrayChecker(self.layer_indices, 'layer_indices')
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
