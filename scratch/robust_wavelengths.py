from pyRT_DISORT.utilities.array_checks import ArrayChecker
import numpy as np


class Wavelengths:
    def __init__(self, short_wavelength, long_wavelength):
        self.short_wavelength = short_wavelength
        self.long_wavelength = long_wavelength

    @property
    def short_wavelength(self):
        return self.__short_wavelength

    @short_wavelength.setter
    def short_wavelength(self, val):
        short_wav_checker = ArrayChecker(val, 'short_wavelength')
        short_wav_checker.check_object_is_array()
        short_wav_checker.check_ndarray_is_numeric()
        short_wav_checker.check_ndarray_is_positive_finite()
        short_wav_checker.check_ndarray_is_1d()
        if hasattr(self, 'long_wavelength'):
            self.__check_short_wavelength_has_same_shape_as_long_wavelength(val)
            self.__check_short_wavelength_is_shorter_than_long_wavelength(val)
        self.__short_wavelength = val

    def __check_short_wavelength_has_same_shape_as_long_wavelength(self, val):
        if val.shape != self.long_wavelength.shape:
            raise ValueError('short_wavelength and long_wavelength must have the same shape')

    def __check_short_wavelength_is_shorter_than_long_wavelength(self, val):
        if not np.all(self.long_wavelength > val):
            raise ValueError('long_wavelength must always be larger than the corresponding short_wavelength')

    @property
    def long_wavelength(self):
        return self.__long_wavelength

    @long_wavelength.setter
    def long_wavelength(self, val):
        long_wav_checker = ArrayChecker(val, 'long_wavelength')
        long_wav_checker.check_object_is_array()
        long_wav_checker.check_ndarray_is_numeric()
        long_wav_checker.check_ndarray_is_positive_finite()
        long_wav_checker.check_ndarray_is_1d()
        if hasattr(self, 'short_wavelength'):
            self.__check_long_wavelength_has_same_shape_as_short_wavelength(val)
            self.__check_log_wavelength_is_longer_than_short_wavelength(val)
        self.__long_wavelength = val

    def __check_long_wavelength_has_same_shape_as_short_wavelength(self, val):
        if self.short_wavelength.shape != val.shape:
            raise ValueError('short_wavelength and long_wavelength must have the same shape')

    def __check_log_wavelength_is_longer_than_short_wavelength(self, val):
        if not np.all(val > self.short_wavelength):
            raise ValueError('long_wavelength must always be larger than the corresponding short_wavelength')
