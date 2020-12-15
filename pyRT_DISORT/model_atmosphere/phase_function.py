# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.utilities.array_checks import ArrayChecker


class LegendreCoefficients:
    """A LegendreCoefficints object is an abstract class to hold basic properties of Legendre coefficients"""
    def __init__(self, max_moments):
        self.max_moments = max_moments
        self.__check_moments_is_int()

    def __check_moments_is_int(self):
        if not isinstance(self.max_moments, int):
            raise ValueError('n_moments must be an int')


class HenyeyGreenstein(LegendreCoefficients):
    """A HenyeyGreenstein object can create the Legendre coefficients for a Henyey-Greenstein phase function"""
    def __init__(self, max_moments=200):
        """
        Parameters
        ----------
        max_moments: int
            The maximum number of moments to create. Default is 200
        """
        super().__init__(max_moments=max_moments)

    def make_legendre_coefficients(self, asymmetry_parameter):
        moments = np.linspace(0, self.max_moments - 1, num=self.max_moments)
        return np.moveaxis((2 * moments + 1) * np.power.outer(asymmetry_parameter, moments), -1, 0)


class TabularLegendreCoefficients(LegendreCoefficients):
    """A TabularLegendreCoefficients object associates particle sizes and wavelengths with an input phase function"""
    def __init__(self, tabulated_coefficients, particle_sizes=None, wavelengths=None, max_moments=None):
        """
        Parameters
        ----------
        tabulated_coefficients: np.ndarray
            3D, 2D, or 1D array of Legendre coefficients. If the array is 3D, both particle_sizes and wavelengths
            must be included. The array is assumed to have dimensions (n_moments, n_particle_sizes, n_wavelengths).
            If the array is 2D, one and only one of particle_sizes and wavelengths must be included. The array is
            assumed to have dimensions (n_moments, n_particle_sizes or n_wavelengths), whichever is included. If the
            array is 1D, neither particle_sizes nor wavelengths can be included and the array is assumed to have
            dimension (n_moments)
        particle_sizes: np.ndarray
            1D array of particle sizes associated with tabulated_coefficients, if they exist. Default is None
        wavelengths: np.ndarray
            1D array of wavelengths associated with tabulated_coefficients, if they exist. Default is None
        max_moments: int
            The maximum number of moments from tabulated_coefficients to use. If max_moments is None, all included
            moments will be used. Default is None.

        Attributes
        ----------
        tabulated_coefficients: np.ndarray
            The input tabulated_coefficients
        particle_sizes: np.ndarray
            The input particle_sizes
        wavelengths: np.ndarray
            The input wavelengths
        """
        self.tabulated_coefficients = tabulated_coefficients
        self.particle_sizes = particle_sizes
        self.wavelengths = wavelengths
        self.__max_moments = max_moments

        self.__coefficients_dimensions = self.__get_coefficients_dimensions()
        self.__coefficients_shape = self.__get_coefficients_shape()
        self.__check_coefficients_and_grids_are_physical()
        self.__check_max_moments_is_int_or_none()

        self.__included_moments = self.__get_n_moments()
        if self.__max_moments is None:
            super().__init__(self.__included_moments)
        else:
            super().__init__(min(self.__max_moments, self.__included_moments))

    def __get_coefficients_dimensions(self):
        return np.ndim(self.tabulated_coefficients)

    def __get_coefficients_shape(self):
        return np.shape(self.tabulated_coefficients)

    def __check_coefficients_and_grids_are_physical(self):
        self.__check_coefficients_are_physical()
        self.__check_particle_size_grid_is_physical()
        self.__check_grids_match_1d_coefficients()
        self.__check_grids_match_2d_coefficients()
        self.__check_grids_match_3d_coefficients()

    def __check_coefficients_are_physical(self):
        coefficient_checker = ArrayChecker(self.tabulated_coefficients, 'tabulated_coefficients')
        coefficient_checker.check_object_is_array()
        coefficient_checker.check_ndarray_is_numeric()
        coefficient_checker.check_ndarray_is_positive_finite()
        self.__check_coefficients_dimensions()

    def __check_coefficients_dimensions(self):
        if self.__coefficients_dimensions not in [1, 2, 3]:
            raise IndexError('tabulated_coefficients must be a 1D, 2D, or 3D array')

    def __check_particle_size_grid_is_physical(self):
        if self.particle_sizes is not None:
            particle_checker = ArrayChecker(self.particle_sizes, 'particle_sizes')
            particle_checker.check_object_is_array()
            particle_checker.check_ndarray_is_numeric()
            particle_checker.check_ndarray_is_positive_finite()
            particle_checker.check_ndarray_is_1d()

    def __check_wavelength_grid_is_physical(self):
        if self.wavelengths is not None:
            wavelength_checker = ArrayChecker(self.wavelengths, 'phase_function_wavelengths')
            wavelength_checker.check_object_is_array()
            wavelength_checker.check_ndarray_is_numeric()
            wavelength_checker.check_ndarray_is_positive_finite()
            wavelength_checker.check_ndarray_is_1d()

    def __check_grids_match_1d_coefficients(self):
        if self.__coefficients_dimensions != 1:
            return
        if self.particle_sizes is not None or self.wavelengths is not None:
            raise ValueError('1D tabulated_coefficients should not include any particle size or wavelength info')

    def __check_grids_match_2d_coefficients(self):
        if self.__coefficients_dimensions != 2:
            return
        if not (self.particle_sizes is None) ^ (self.wavelengths is None):
            raise TypeError('For 2D tabulated_coefficients, provide one and only one of particle_sizes and wavelengths')
        if self.particle_sizes is not None:
            if self.__coefficients_shape[1] != len(self.particle_sizes):
                raise IndexError(
                    '2D tabulated_coefficients\' second dimension must be the same length as particle_size_grid')
        if self.wavelengths is not None:
            if self.__coefficients_shape[1] != len(self.wavelengths):
                raise IndexError(
                    '2D tabulated_coefficients\' second dimension must be the same length as wavelength_grid')

    def __check_grids_match_3d_coefficients(self):
        if self.__coefficients_dimensions != 3:
            return
        if self.particle_sizes is None or self.wavelengths is None:
            raise TypeError('You need to include both particle_sizes and wavelengths')
        if self.__coefficients_shape[1] != len(self.particle_sizes):
            raise IndexError(
                '3D tabulated_coefficients\' second dimension must be the same length as particle sizes')
        if self.__coefficients_shape[2] != len(self.wavelengths):
            raise IndexError(
                    '3D tabulated_coefficients\' third dimension must be the same length as wavelengths')

    def __check_max_moments_is_int_or_none(self):
        if not isinstance(self.max_moments, (int, type(None))):
            raise TypeError('max_moments must be an int or None')

    def __get_n_moments(self):
        return self.tabulated_coefficients.shape[0]
