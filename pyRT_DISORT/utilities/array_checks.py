# 3rd-party imports
import numpy as np


# TODO: docstrings throughout?
class ArrayChecker:
    """An ArrayChecker object can perform common checks on np.ndarrays.

    The ArrayChecker class holds methods commonly performed on np.ndarrays
    throughout pyRT_DISORT to determine if the array fits some criteria.

    """

    def __init__(self, array):
        self.__array = array
        self.__raise_type_error_if_object_is_not_array()

    def __raise_type_error_if_object_is_not_array(self):
        if not self.__object_is_array():
            raise TypeError('The input object is not a np.ndarray.')

    def __object_is_array(self):
        return isinstance(self.__array, np.ndarray)

    def determine_if_array_is_numeric(self):
        return np.issubdtype(self.__array.dtype, np.number)

    def determine_if_array_contains_only_ints(self):
        return np.issubdtype(self.__array.dtype, np.int)

    def determine_if_array_is_finite(self):
        return np.all(np.isfinite(self.__array))

    def determine_if_array_is_positive(self):
        return np.all(self.__array > 0)

    def determine_if_array_is_non_negative(self):
        return np.all(self.__array >= 0)

    def determine_if_array_is_positive_finite(self):
        return self.determine_if_array_is_positive() and \
               self.determine_if_array_is_finite()

    def determine_if_array_is_in_range(self, low, high):
        more_than_low = np.all(self.__array >= low)
        less_than_high = np.all(self.__array <= high)
        return more_than_low and less_than_high

    def determine_if_array_matches_dimensionality(self, dimension):
        return np.ndim(self.__array) == dimension

    def determine_if_array_is_1d(self):
        return self.determine_if_array_matches_dimensionality(1)

    def determine_if_array_is_2d(self):
        return self.determine_if_array_matches_dimensionality(2)

    def determine_if_array_is_monotonically_decreasing(self):
        return np.all(np.diff(self.__array) < 0)

    def determine_if_array_is_monotonically_increasing(self):
        return np.all(np.diff(self.__array) > 0)

    # TODO: cleanup
    def check_1d_array_is_no_longer_than(self, length):
        if len(self.some_object) > length:
            raise ValueError(f'{self.array_name} must be no longer than {length} elements long')

    # TODO: cleanup
    def check_1d_array_is_at_least(self, length):
        if len(self.some_object) < length:
            raise ValueError(f'{self.array_name} must be at least {length} elements long')
