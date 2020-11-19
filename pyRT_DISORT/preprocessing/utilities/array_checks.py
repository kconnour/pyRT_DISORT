import numpy as np


class CheckArray:
    def __init__(self, ndarray):
        self.ndarray = ndarray
        self.array_name = f'{self.ndarray=}'.split('=')[0]

    def check_object_is_array(self):
        if not isinstance(self.ndarray, np.ndarray):
            raise TypeError(f'{self.array_name} must be a np.ndarray')

    def check_ndarray_is_numeric(self):
        if not np.issubdtype(self.ndarray.dtype, np.number):
            raise TypeError(f'{self.array_name} must contain only numbers')

    def check_ndarray_is_finite(self):
        if not np.all(np.isfinite(self.ndarray)):
            raise TypeError(f'{self.array_name} must contain all finite values')

    def check_ndarray_is_positive(self):
        if not np.all(self.ndarray > 0):
            raise TypeError(f'{self.array_name} must contain all positive values')

    def check_ndarray_is_positive_finite(self):
        self.check_ndarray_is_finite()
        self.check_ndarray_is_positive()

    def check_ndarray_is_in_range(self, low, high):
        if not(np.all(self.ndarray >= low) and np.all(self.ndarray <= high)):
            raise ValueError(f'{self.array_name} must be in range [{low}--{high}]')

    def check_ndarray_dimension(self, dimension):
        return np.ndim(self.ndarray) == dimension

    def check_ndarray_is_1d(self):
        if not self.check_ndarray_dimension(1):
            raise IndexError(f'{self.array_name} must be a 1D array')
