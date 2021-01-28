"""array_checks.py contains ArrayChecker, an object to perform a variety of
array checks on an input array.
"""
import numpy as np


class ArrayChecker:
    """An ArrayChecker object can perform common checks on np.ndarrays.

    The ArrayChecker class holds methods commonly performed on np.ndarrays
    throughout pyRT_DISORT to determine if the array fits some criteria.

    """

    def __init__(self, array) -> None:
        """
        Parameters
        ----------
        array: np.ndarray
            Any array to perform checks on.

        Raises
        ------
        TypeError
            Raised if the input is not a np.ndarray.

        """
        self.__array = array
        self.__raise_type_error_if_object_is_not_array()

    def __raise_type_error_if_object_is_not_array(self) -> None:
        if not self.__object_is_array():
            raise TypeError('The input object is not a np.ndarray.')

    def __object_is_array(self) -> bool:
        return isinstance(self.__array, np.ndarray)

    def determine_if_array_contains_only_ints(self) -> bool:
        """Determine if the array contains only integers.

        Returns
        -------
        bool
            True if the contents are all integers; False otherwise.

        """
        return np.issubdtype(self.__array.dtype, np.int)

    def determine_if_array_is_1d(self) -> bool:
        """Determine if the array is 1D.

        Returns
        -------
        bool
            True if the array is 1D; False otherwise.

        """
        return self.determine_if_array_matches_dimensionality(1)

    def determine_if_array_is_2d(self) -> bool:
        """Determine if the array is 2D.

        Returns
        -------
        bool
            True if the array is 2D; False otherwise.

        """
        return self.determine_if_array_matches_dimensionality(2)

    def determine_if_array_is_finite(self) -> bool:
        """Determine if the array contains only finite values.

        Returns
        -------
        bool
            True if the contents are all finite; False otherwise.

        """
        return np.all(np.isfinite(self.__array))

    def determine_if_array_is_in_range(self, low, high) -> bool:
        """Determine if the array contains only values within a range.

        Returns
        -------
        bool
            True if the contents are all within the input range; False
            otherwise.

        """
        return np.all(self.__array >= low) and np.all(self.__array <= high)

    def determine_if_array_is_monotonically_decreasing(self) -> bool:
        """Determine if the array is monotonically decreasing.

        Returns
        -------
        bool
            True if the array is monotonically decreasing; False otherwise.

        """
        return np.all(np.diff(self.__array) < 0)

    def determine_if_array_is_monotonically_increasing(self) -> bool:
        """Determine if the array is monotonically increasing.

        Returns
        -------
        bool
            True if the array is monotonically increasing; False otherwise.

        """
        return np.all(np.diff(self.__array) > 0)

    def determine_if_array_is_non_negative(self) -> bool:
        """Determine if the array contains only non-negative values.

        Returns
        -------
        bool
            True if the contents are all non-negative; False otherwise.

        """
        return np.all(self.__array >= 0)

    def determine_if_array_is_numeric(self) -> bool:
        """Determine if the array contains numeric values.

        Returns
        -------
        bool
            True if contents are numeric; False otherwise.

        """
        return np.issubdtype(self.__array.dtype, np.number)

    def determine_if_array_is_positive(self) -> bool:
        """Determine if the array contains only positive values.

        Returns
        -------
        bool
            True if the contents are all positive; False otherwise.

        """
        return np.all(self.__array > 0)

    def determine_if_array_is_positive_finite(self) -> bool:
        """Determine if the array contains only positive, finite values.

        Returns
        -------
        bool
            True if the contents are all positive finite; False otherwise.

        """
        return self.determine_if_array_is_positive() and \
            self.determine_if_array_is_finite()

    def determine_if_array_matches_dimensionality(self, dimension) -> bool:
        """Determine if the array matches an input number of dimensions.

        Returns
        -------
        bool
            True if the array has the input number of dimensions; False
            otherwise.

        """
        return np.ndim(self.__array) == dimension
