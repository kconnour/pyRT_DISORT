"""This module contains code for creating a generic Column object.
"""
import numpy as np
from numpy.typing import ArrayLike


class Column:
    """A data class to hold column information and interact with other column
    objects.

    Parameters
    ----------
    optical_depth: ArrayLike
        N-dimensional array of optical depths.
    single_scattering_albedo: ArrayLike
        N-dimensional array of single-scattering albedos. Must be the same
        shape as `optical_depth`.
    legendre_coefficients: ArrayLike
        N-dimensional array of Legendre coefficients. Axis 0 can have any
        length but the remaining axes must have the same shape as
        `optical_depth`. These get divided by 2k + 1 to keep with DISORT's
        convention.

    Examples
    --------
    Suppose you have some dust aerosols that you want to use in a 15-layer
    RT model. You can group these properties together in a Column object.

    >>> import numpy as np
    >>> import pyrt
    >>> dust_optical_depth = np.linspace(0.1, 1, num=15)
    >>> dust_single_scattering_albedo = np.ones((15,)) * 0.7
    >>> dust_legendre_coefficients = np.ones((128, 15))
    >>> dust_column = Column(dust_optical_depth, dust_single_scattering_albedo, dust_legendre_coefficients)

    You can access these via this object's properties. The optical depth and
    single scattering albedo are unchanged, but the Legendre coefficients
    are divided by 2k + 1 to match what DISORT wants.

    >>> dust_column.single_scattering_albedo
    array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
           0.7, 0.7])

    Suppose you also want to add some ice aerosols to your model. You can create
    a separate Column for these.

    >>> ice_optical_depth = np.linspace(0, 0.5, num=15)
    >>> ice_single_scattering_albedo = np.ones((15,))
    >>> ice_legendre_coefficients = np.ones((128, 15))
    >>> ice_column = Column(ice_optical_depth, ice_single_scattering_albedo, ice_legendre_coefficients)

    If these are all the aerosols that make up your model, you can add them
    to get a new Column that represents these properties for the combined
    atmosphere.

    >>> total_column = dust_column + ice_column
    >>> total_column.optical_depth
    array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2, 1.3,
           1.4, 1.5])
    >>> total_column.single_scattering_albedo
    array([0.7       , 0.75357143, 0.77142857, 0.78035714, 0.78571429,
           0.78928571, 0.79183673, 0.79375   , 0.7952381 , 0.79642857,
           0.7974026 , 0.79821429, 0.7989011 , 0.7994898 , 0.8       ])

    """
    def __init__(self, optical_depth: ArrayLike,
                 single_scattering_albedo: ArrayLike,
                 legendre_coefficients: ArrayLike):
        self.optical_depth = optical_depth
        self.single_scattering_albedo = single_scattering_albedo
        self.legendre_coefficients = legendre_coefficients

        self._raise_value_error_if_inputs_have_incompatible_shapes()

    def _raise_value_error_if_inputs_have_incompatible_shapes(self):
        if not (self.optical_depth.shape ==
                self.single_scattering_albedo.shape ==
                self.legendre_coefficients.shape[1:]):
            message = 'The inputs have incompatible shapes.'
            raise ValueError(message)

    @property
    def optical_depth(self):
        return self._optical_depth

    @optical_depth.setter
    def optical_depth(self, value):
        def _make_array(val):
            try:
                array = np.asarray(val).astype(float)
            except TypeError as te:
                message = 'The optical depth must be ArrayLike.'
                raise TypeError(message) from te
            except ValueError as ve:
                message = 'The optical depth must be numeric.'
                raise ValueError(message) from ve
            return array

        def _validate(array):
            if not np.all((0 <= array) & (array <= np.inf)):
                message = 'The optical depth must be positive finite.'
                raise ValueError(message)

        arr = _make_array(value)
        _validate(arr)
        self._optical_depth = arr

    @property
    def single_scattering_albedo(self):
        return self._single_scattering_albedo

    @single_scattering_albedo.setter
    def single_scattering_albedo(self, value):
        def _make_array(val):
            try:
                array = np.asarray(val).astype(float)
            except TypeError as te:
                message = 'The single-scattering albedo must be ArrayLike.'
                raise TypeError(message) from te
            except ValueError as ve:
                message = 'The single-scattering albedo must be numeric.'
                raise ValueError(message) from ve
            return array

        def _validate(array):
            if not np.all((0 <= array) & (array <= 1)):
                message = 'The single-scattering albedo must be between 0 and 1.'
                raise ValueError(message)

        arr = _make_array(value)
        _validate(arr)
        self._single_scattering_albedo = arr

    @property
    def legendre_coefficients(self):
        return self._legendre_coefficients

    @legendre_coefficients.setter
    def legendre_coefficients(self, value):
        def _make_array(val):
            try:
                array = np.asarray(val).astype(float)
            except TypeError as te:
                message = 'The Legendre coefficients must be ArrayLike.'
                raise TypeError(message) from te
            except ValueError as ve:
                message = 'The Legendre coefficients must be numeric.'
                raise ValueError(message) from ve
            return array

        def _validate(array):
            if not np.all(np.isfinite(array)):
                message = 'The Legendre coefficients must be finite.'
                raise ValueError(message)

        arr = _make_array(value)
        _validate(arr)
        self._legendre_coefficients = arr
        self._normalize_legendre_coefficients()

    def _normalize_legendre_coefficients(self) -> None:
        weight = np.arange(self._legendre_coefficients.shape[0]) * 2 + 1
        self._legendre_coefficients = (self._legendre_coefficients.T / weight).T

    def _denormalize_legendre_coefficients(self) -> None:
        weight = np.arange(self._legendre_coefficients.shape[0]) * 2 + 1
        self._legendre_coefficients = (self._legendre_coefficients.T * weight).T

    def __add__(self, obj):
        self._raise_type_error_if_input_is_not_a_column(obj)
        self._raise_value_error_if_objects_have_incompatible_shapes(obj)
        self._match_moments(obj)

        od = self._calculate_total_optical_depth(obj)
        ssa = self._calculate_single_scattering_albedo(obj)
        pmom = self._calculate_legendre_coefficients(obj)
        col = Column(od, ssa, pmom)
        col._denormalize_legendre_coefficients()
        return col

    @staticmethod
    def _raise_type_error_if_input_is_not_a_column(obj):
        if not isinstance(obj, Column):
            message = f'The input must be a Column, not a {type(obj)}.'
            raise TypeError(message)

    def _raise_value_error_if_objects_have_incompatible_shapes(self, obj):
        if self.optical_depth.shape != obj.optical_depth.shape:
            message = 'The objects cannot be added because their data do ' \
                      'not have the same shapes.'
            raise ValueError(message)

    def _match_moments(self, other):
        # This function ensures both Column objects have the same number of
        #  Legendre coefficients. For instance, if one has a shape of (128, 15)
        #  and the other is (200, 15), the result should be (200, 15).

        # Strategy: Make 2 arrays of 0s and fill them with each object's
        #  Legendre coefficients. Then replace each object's Legendre
        #  coefficients with these new arrays.
        max_moments = self._get_max_moments(other)

        current_legendre_coefficients = \
            np.zeros((max_moments, *self.optical_depth.shape))
        current_legendre_coefficients[:self.legendre_coefficients.shape[0]] = \
            self.legendre_coefficients
        self._legendre_coefficients = current_legendre_coefficients

        other_legendre_coefficients = \
            np.zeros((max_moments, *self.optical_depth.shape))
        other_legendre_coefficients[:other.legendre_coefficients.shape[0]] = \
            other.legendre_coefficients
        other._legendre_coefficients = other_legendre_coefficients

    def _get_max_moments(self, other):
        return max(self.legendre_coefficients.shape[0],
                   other.legendre_coefficients.shape[0])

    def _calculate_total_optical_depth(self, other) -> np.ndarray:
        return self.optical_depth + other.optical_depth

    def _calculate_scattering_optical_depth(self, other) -> np.ndarray:
        return self.single_scattering_albedo * self.optical_depth + \
            other.single_scattering_albedo * other.optical_depth

    def _calculate_single_scattering_albedo(self, other) -> np.ndarray:
        scattering_od = self._calculate_scattering_optical_depth(other)
        total_od = self._calculate_total_optical_depth(other)
        return scattering_od / total_od

    def _calculate_legendre_coefficients(self, other) -> np.ndarray:
        weighted_moments = self.optical_depth * \
                           self.single_scattering_albedo * \
                           self.legendre_coefficients + \
                           other.optical_depth * \
                           other.single_scattering_albedo * \
                           other.legendre_coefficients
        return weighted_moments / self._calculate_scattering_optical_depth(other)
