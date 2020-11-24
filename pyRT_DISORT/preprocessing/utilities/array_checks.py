import numpy as np


class CheckArray:
    def __init__(self, ndarray, array_name):
        self.array_name = array_name
        self.ndarray = ndarray

    def check_object_is_array(self):
        if not isinstance(self.ndarray, np.ndarray):
            raise TypeError(f'{self.array_name} must be a np.ndarray')

    def check_ndarray_is_numeric(self):
        if not np.issubdtype(self.ndarray.dtype, np.number):
            raise ValueError(f'{self.array_name} must contain only numbers')

    def check_ndarray_contains_only_ints(self):
        if not np.issubdtype(self.ndarray.dtype, np.int):
            raise ValueError(f'{self.array_name} must contain only integers')

    def check_ndarray_is_finite(self):
        if not np.all(np.isfinite(self.ndarray)):
            raise ValueError(f'{self.array_name} must contain all finite values')

    def check_ndarray_is_positive(self):
        if not np.all(self.ndarray > 0):
            raise ValueError(f'{self.array_name} must contain all positive values')

    def check_ndarray_is_non_negative(self):
        if not np.all(self.ndarray >= 0):
            raise ValueError(f'{self.array_name} must contain all non-negative values')

    def check_ndarray_is_positive_finite(self):
        self.check_ndarray_is_finite()
        self.check_ndarray_is_positive()

    def check_ndarray_is_in_range(self, low, high):
        if not(np.all(self.ndarray >= low) and np.all(self.ndarray <= high)):
            raise ValueError(f'{self.array_name} must be in range [{low}, {high}]')

    def check_ndarray_dimension(self, dimension):
        return np.ndim(self.ndarray) == dimension

    def check_ndarray_is_1d(self):
        if not self.check_ndarray_dimension(1):
            raise IndexError(f'{self.array_name} must be a 1D array')

    def check_ndarray_is_2d(self):
        if not self.check_ndarray_dimension(2):
            raise IndexError(f'{self.array_name} must be a 2D array')

    def check_1d_array_is_monotonically_decreasing(self):
        diff = np.diff(self.ndarray)
        if not np.all(diff < 0):
            raise ValueError(f'{self.array_name} must be monotonically decreasing')

    def check_1d_array_is_no_longer_than(self, length):
        if len(self.ndarray) > length:
            raise ValueError(f'{self.array_name} must be no longer than {length}')
