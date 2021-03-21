"""The vertical profile module contains structures to create vertical profiles
for use in DISORT.
"""
import numpy as np
from pyRT_DISORT.eos import Altitude


class Conrath:
    r"""A structure to compute a Conrath profile(s).

    Conrath creates Conrath profile(s) on an input grid of altitudes given a
    volumetric mixing ratio and Conrath parameters. The Conrath profile is
    defined as

    .. math::

       q(z) = q_0 * e^{\nu(1 - e^{z/H})}

    where :math:`q` is a volumetric mixing ratio, :math:`z` is the altitude,
    :math:`\nu` is the Conrath nu parameter, and :math:`H` is the scale height.
    """

    def __init__(self, altitude: np.ndarray, q0: np.ndarray,
                 scale_height: np.ndarray, nu: np.ndarray) -> None:
        r"""
        Parameters
        ----------
        altitude
            The altitude at which to construct a Conrath profile (probably the
            midpoint altitude)
        q0
            The surface mixing ratio for each of the Conrath profiles.
        scale_height
            The scale height of each of the Conrath profiles.
        nu
            The nu parameter of each of the Conrath profiles.

        Raises
        ------
        TypeError
            Raised if any of the inputs are not a numpy.ndarray.
        ValueError
            Raised if any of the inputs do not have the same shape, or if they
            contain physically or mathematically unrealistic values.

        Notes
        -----
        For an MxN array of pixels, :code:`q0`, :code:`scale_height`, and
        :code:`nu` should be of shape MxN. :code:`altitude` should have shape
        ZxMxN, where Z is the number of altitudes. Additionally, the units of
        :code:`altitude` and :code:`scale_height` should be the same.

        """
        Altitude(altitude)
        ConrathParameterValidator(q0, 'q0', 0, np.inf)
        ConrathParameterValidator(scale_height, 'scale_height', 0, np.inf)
        ConrathParameterValidator(nu, 'nu', 0, np.inf)

        self.__altitude = altitude
        self.__q0 = q0
        self.__scale_height = scale_height
        self.__nu = nu

        self.__raise_value_error_if_inputs_have_differing_shapes()

        self.__profile = self.__make_profile()

    def __raise_value_error_if_inputs_have_differing_shapes(self) -> None:
        if not (self.__altitude.shape[1:] == self.__q0.shape ==
                self.__scale_height.shape == self.__nu.shape):
            message = 'All inputs must have the same shape along the pixel ' \
                      'dimension.'
            raise ValueError(message)

    def __make_profile(self) -> np.ndarray:
        altitude_scaling = self.__altitude / self.__scale_height
        return self.__q0 * np.exp(self.__nu * (1 - np.exp(altitude_scaling)))

    @property
    def profile(self) -> np.ndarray:
        """Get the Conrath profile(s). This will have the same shape as
        :code:`altitude`.

        """
        return self.__profile


class ConrathParameterValidator:
    def __init__(self, parameter: np.ndarray, name: str,
                 low: float, high: float) -> None:
        self.__param = parameter
        self.__name = name
        self.__low = low
        self.__high = high

        self.__raise_error_if_param_is_bad()

    def __raise_error_if_param_is_bad(self) -> None:
        self.__raise_type_error_if_not_ndarray()
        self.__raise_value_error_if_not_in_range()

    def __raise_type_error_if_not_ndarray(self) -> None:
        if not isinstance(self.__param, np.ndarray):
            message = f'{self.__name} must be a numpy.ndarray.'
            raise TypeError(message)

    def __raise_value_error_if_not_in_range(self) -> None:
        if not (np.all(self.__low <= self.__param) and
                (np.all(self.__param <= self.__high))):
            message = f'{self.__name} must be between {self.__low} and ' \
                      f'{self.__high}.'
            raise ValueError(message)


class Uniform:
    """A structure to create a uniform profile(s).

    Uniform creates uniform volumetric mixing ratio profile(s) on an input
    grid of altitudes given a set of top and bottom altitudes.

    """

    def __init__(self, altitude: np.ndarray, bottom: np.ndarray,
                 top: np.ndarray) -> None:
        """
        Parameters
        ----------
        altitude
            The altitude *boundaries* at which to construct uniform profile(s).
            These are assumed to be decreasing to keep with DISORT's convention.
        bottom
            The bottom altitudes of each of the profiles.
        top
            The top altitudes of each of the profiles.

        Raises
        ------
        TypeError
            Raised if any inputs are not a numpy.ndarray.
        ValueError
            Raised if a number of things...

        """
        self.__altitude = altitude
        self.__bottom = bottom
        self.__top = top

        Altitude(altitude)
        UniformParameterValidator(bottom, 'bottom')
        UniformParameterValidator(top, 'top')

        self.__profile = self.__make_profile()

    def __raise_error_if_inputs_are_bad(self) -> None:
        self.__raise_value_error_if_inputs_have_differing_shapes()
        self.__raise_value_error_if_top_is_not_larger()

    def __raise_value_error_if_inputs_have_differing_shapes(self) -> None:
        if not (self.__altitude.shape[1:] == self.__top.shape ==
                self.__bottom.shape):
            message = 'All inputs must have the same shape along the pixel ' \
                      'dimension.'
            raise ValueError(message)

    def __raise_value_error_if_top_is_not_larger(self) -> None:
        if not np.all(self.__top > self.__bottom):
            message = 'All values in top must be larger than the ' \
                      'corresponding values in bottom.'
            raise ValueError(message)

    def __make_profile(self) -> np.ndarray:
        alt_dif = np.diff(self.__altitude, axis=0)
        top_prof = np.clip((self.__top - self.__altitude[1:]) /
                           np.abs(alt_dif), 0, 1)
        bottom_prof = np.clip((self.__altitude[:-1] - self.__bottom) /
                              np.abs(alt_dif), 0, 1)
        return top_prof + bottom_prof - 1

    @property
    def profile(self) -> np.ndarray:
        """Get the uniform profile(s). This will have the same shape as
        :code:`altitude`.

        """
        return self.__profile


class UniformParameterValidator:
    def __init__(self, parameter: np.ndarray, name: str) -> None:
        self.__param = parameter
        self.__name = name

        self.__raise_error_if_param_is_bad()

    def __raise_error_if_param_is_bad(self) -> None:
        self.__raise_type_error_if_not_ndarray()

    def __raise_type_error_if_not_ndarray(self) -> None:
        if not isinstance(self.__param, np.ndarray):
            message = f'{self.__name} must be a numpy.ndarray.'
            raise TypeError(message)
