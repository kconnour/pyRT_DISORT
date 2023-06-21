"""This module contains code for creating a generic Column object.
"""
import numpy as np
from numpy.typing import ArrayLike


class Column:
    """A data class to hold column information.

    Its primary utility is adding columns via the `+` operator.

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
        `optical_depth`. These get divided by 2k + 1.

    """
    def __init__(self, optical_depth: ArrayLike,
                 single_scattering_albedo: ArrayLike,
                 legendre_coefficients: ArrayLike):
        self.optical_depth = optical_depth
        self.single_scattering_albedo = single_scattering_albedo
        self.legendre_coefficients = legendre_coefficients

        self._raise_value_error_if_inputs_have_incompatible_shapes()

    def _raise_value_error_if_inputs_have_incompatible_shapes(self):
        if not (self.optical_depth.shape == self.single_scattering_albedo.shape == self.legendre_coefficients.shape[1:]):
            print(self.optical_depth.shape, self.single_scattering_albedo.shape, self.legendre_coefficients.shape[1:])
            message = 'The inputs have incompatible shapes.'
            raise ValueError(message)

    @property
    def optical_depth(self):
        return self._optical_depth

    @optical_depth.setter
    def optical_depth(self, value):
        def _make_array(val):
            try:
                array = np.asarray(val)
                array.astype(float)
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
        return self._ssa

    @single_scattering_albedo.setter
    def single_scattering_albedo(self, value):
        def _make_array(val):
            try:
                array = np.asarray(val)
                array.astype(float)
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
        self._ssa = arr

    @property
    def legendre_coefficients(self):
        return self._pmom

    @legendre_coefficients.setter
    def legendre_coefficients(self, value):
        def _make_array(val):
            try:
                array = np.asarray(val)
                array.astype(float)
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
        self._pmom = arr
        self._normalize_legendre_coefficients()

    def __add__(self, obj):
        self._raise_value_error_if_objects_have_incompatible_shapes(obj)
        self._match_moments(obj)

        od = self._calculate_total_optical_depth(obj)
        ssa = self._calculate_single_scattering_albedo(obj)
        pmom = self._calculate_legendre_coefficients(obj)
        col = Column(od, ssa, pmom)
        col._denormalize_legendre_coefficients()
        return col

    def _raise_value_error_if_objects_have_incompatible_shapes(self, obj):
        if self.optical_depth.shape != obj.optical_depth.shape:
            message = 'The objects cannot be added because their data do ' \
                      'not have the same shapes.'
            raise ValueError(message)

    def _match_moments(self, other):
        max_moments = self._get_max_moments(other)
        current_pmom = np.zeros((max_moments, *self.optical_depth.shape))
        other_pmom = np.copy(current_pmom)
        current_pmom[:self.legendre_coefficients.shape[0]] = self.legendre_coefficients
        other_pmom[:other.legendre_coefficients.shape[0]] = other.legendre_coefficients
        self._pmom = current_pmom
        other._pmom = other_pmom

    def _get_max_moments(self, other):
        return max(self.legendre_coefficients.shape[0], other.legendre_coefficients.shape[0])

    def _calculate_total_optical_depth(self, other) -> np.ndarray:
        return self.optical_depth + other.optical_depth

    def _calculate_scattering_optical_depth(self, other) -> np.ndarray:
        return self.single_scattering_albedo * self.optical_depth + other.single_scattering_albedo * other.optical_depth

    def _calculate_single_scattering_albedo(self, other) -> np.ndarray:
        scattering_od = self._calculate_scattering_optical_depth(other)
        total_od = self._calculate_total_optical_depth(other)
        return scattering_od / total_od

    def _calculate_legendre_coefficients(self, other) -> np.ndarray:
        weighted_moments = self.optical_depth * self.single_scattering_albedo * self.legendre_coefficients + \
                           other.optical_depth * other.single_scattering_albedo * other.legendre_coefficients
        scattering_od = self._calculate_scattering_optical_depth(other)
        return weighted_moments / scattering_od

    def _normalize_legendre_coefficients(self) -> None:
        weight = np.arange(self._pmom.shape[0]) * 2 + 1
        self._pmom = (self._pmom.T / weight).T

    def _denormalize_legendre_coefficients(self) -> None:
        weight = np.arange(self._pmom.shape[0]) * 2 + 1
        self._pmom = (self._pmom.T * weight).T
