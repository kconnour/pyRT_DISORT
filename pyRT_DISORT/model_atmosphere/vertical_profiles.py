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
        print('this is an unfinished class right now! It doesn\'t do anything...')

    def __check_input_altitudes_are_plausible(self):
        if not isinstance(self.altitude_bottom, (int, float)):
            raise ValueError('altitude_bottom must be an int or float')
        if not isinstance(self.altitude_top, (int, float)):
            raise ValueError('altitude_top must be an int or float')
        if self.altitude_bottom >= self.altitude_top:
            raise ValueError('altitude_top must be greater than altitude_bottom')
