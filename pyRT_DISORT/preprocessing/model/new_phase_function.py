import numpy as np
from pyRT_DISORT.preprocessing.utilities.array_checks import ArrayChecker


class LegendreCoefficients:
    def __init__(self, n_moments):
        self.n_moments = n_moments
        self.__check_moments()

    def __check_moments(self):
        if not isinstance(self.n_moments, int):
            raise ValueError('n_moments must be an int')


class HenyeyGreenstein(LegendreCoefficients):
    def __init__(self, n_moments=200):
        super().__init__(n_moments=n_moments)

    def make_legendre_coefficients(self, asymmetry_parameter):
        moments = np.linspace(0, self.n_moments - 1, num=self.n_moments)
        return np.moveaxis((2 * moments + 1) * np.power.outer(asymmetry_parameter, moments), -1, 0)


class TabularLegendreCoefficients(LegendreCoefficients):
    def __init__(self, tabulated_coefficients, phase_function_particle_sizes=None, phase_function_wavelengths=None,
                 max_moments=None, debug=False):
        self.tabulated_coefficients = tabulated_coefficients
        self.particle_sizes = phase_function_particle_sizes
        self.wavelengths = phase_function_wavelengths
        self.max_moments = max_moments
        self.coefficients_dimensions = self.__get_coefficients_dimensions()
        if max_moments is None:
            super().__init__(self.__get_n_moments())
        else:
            super().__init__(min(self.max_moments, self.__get_n_moments()))

        if debug:
            self.__particle_none = self.__check_if_input_is_none(self.particle_sizes)
            self.__wavelength_none = self.__check_if_input_is_none(self.wavelengths)

            self.__coefficients_shape = self.__get_coefficients_shape()
            self.__check_coefficients_and_grids()

    def __get_n_moments(self):
        return self.tabulated_coefficients.shape[0]

    @staticmethod
    def __check_if_input_is_none(ndarray):
        return True if ndarray is None else False

    def __get_coefficients_dimensions(self):
        return np.ndim(self.tabulated_coefficients)

    def __get_coefficients_shape(self):
        return np.shape(self.tabulated_coefficients)

    def __check_coefficients_and_grids(self):
        self.__check_coefficients_are_plausible()
        self.__check_grids_are_plausible()
        self.__check_coefficients_dimensions()
        self.__check_grids_match_1d_coefficients()
        self.__check_grids_match_2d_coefficients()
        self.__check_grids_match_3d_coefficients()

    def __check_coefficients_are_plausible(self):
        coefficient_checker = ArrayChecker(self.tabulated_coefficients, 'tabulated_coefficients')
        coefficient_checker.check_object_is_array()
        coefficient_checker.check_ndarray_is_numeric()
        coefficient_checker.check_ndarray_is_positive_finite()

    def __check_grids_are_plausible(self):
        if not self.__particle_none:
            particle_checker = ArrayChecker(self.particle_sizes, 'phase_function_particle_sizes')
            particle_checker.check_object_is_array()
            particle_checker.check_ndarray_is_numeric()
            particle_checker.check_ndarray_is_positive_finite()
            particle_checker.check_ndarray_is_1d()
        if not self.__wavelength_none:
            wavelength_checker = ArrayChecker(self.wavelengths, 'phase_function_wavelengths')
            wavelength_checker.check_object_is_array()
            wavelength_checker.check_ndarray_is_numeric()
            wavelength_checker.check_ndarray_is_positive_finite()
            wavelength_checker.check_ndarray_is_1d()

    def __check_coefficients_dimensions(self):
        if self.coefficients_dimensions not in [1, 2, 3]:
            raise IndexError('tabulated_coefficients must be a 1D, 2D, or 3D array')

    def __check_grids_match_1d_coefficients(self):
        if self.coefficients_dimensions != 1:
            return
        else:
            if not (self.__particle_none and self.__wavelength_none):
                raise ValueError('1D tabulated_coefficients should not include any particle size or wavelength info')

    def __check_grids_match_2d_coefficients(self):
        if self.coefficients_dimensions != 2:
            return
        if not self.__particle_none ^ self.__wavelength_none:
            raise TypeError(
                'For 2D tabulated_coefficients, provide one and only one of particle size / wavelength info')
        if not self.__particle_none:
            if self.__coefficients_shape[0] != len(self.particle_sizes):
                raise IndexError(
                    '2D tabulated_coefficients\' first dimension must be the same length as particle_size_grid')
        if not self.__wavelength_none:
            if self.__coefficients_shape[0] != len(self.wavelengths):
                raise IndexError(
                    '2D tabulated_coefficients\' first dimension must be the same length as wavelength_grid')

    def __check_grids_match_3d_coefficients(self):
        if self.coefficients_dimensions != 3:
            return
        if self.__particle_none or self.__wavelength_none:
            raise TypeError('You need to include both particle size and wavelength info')
        if self.__coefficients_shape[0] != len(self.particle_sizes):
            raise IndexError(
                '3D tabulated_coefficients\' first dimension must be the same length as particle sizes')
        if self.__coefficients_shape[1] != len(self.wavelengths):
            raise IndexError(
                    '3D tabulated_coefficients\' second dimension must be the same length as wavelengths')
