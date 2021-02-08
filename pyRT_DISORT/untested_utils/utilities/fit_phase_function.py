# 3rd-party imports
import numpy as np
from scipy import integrate, interpolate

# Local imports
from pyRT_DISORT.untested_utils.utilities import ArrayChecker


class PhaseFunction:
    """A PhaseFunction object holds input phase function and can create its Legendre decomposition"""
    def __init__(self, empirical_phase_function, angles):
        """
        Parameters
        ----------
        empirical_phase_function: np.ndarray
            1D array of the empirical phase function
        angles: np.ndarray
            1D array of the angles [radians] at which empirical_phase_function is defined

        Attributes
        ----------
        empirical_phase_function: np.ndarray
            The input empirical phase function
        angles: np.ndarray
            The input angles
        mu: np.ndarray
            The cosine of angles
        n_angles: int
            The length of angles
        """
        self.empirical_phase_function = empirical_phase_function
        self.angles = angles

        self.__check_inputs_are_physical()

        self.mu = np.cos(self.angles)
        self.n_angles = len(self.angles)

    def __check_inputs_are_physical(self):
        self.__check_empirical_phase_function_is_physical()
        self.__check_angles_are_physical()
        self.__check_epf_matches_angles_shape()

    def __check_empirical_phase_function_is_physical(self):
        epf_checker = ArrayChecker(self.empirical_phase_function, 'empirical_phase_function')
        epf_checker.check_object_is_array()
        epf_checker.check_ndarray_is_numeric()
        epf_checker.check_ndarray_is_positive_finite()
        epf_checker.check_ndarray_is_1d()

    def __check_angles_are_physical(self):
        angles_checker = ArrayChecker(self.angles, 'angles')
        angles_checker.check_object_is_array()
        angles_checker.check_ndarray_is_numeric()
        angles_checker.check_ndarray_is_in_range(0, np.pi)
        angles_checker.check_ndarray_is_1d()

    def __check_epf_matches_angles_shape(self):
        if self.empirical_phase_function.shape != self.angles.shape:
            raise ValueError('empirical_phase_function and angles must have the same shape')

    def create_legendre_coefficients(self, n_moments, n_samples):
        """Create the Legendre coefficient decomposition for the input phase function

        Parameters
        ----------
        n_moments: int
            The desired number of moments to find a solution for
        n_samples: int
            Resample the input phase function to n_samples. Must be >= n_moments

        Returns
        -------
        fit_coefficients: np.ndarray
            A 1D array of fitted coefficients of length n_moments
        """
        self.__check_inputs_are_expected(n_moments, n_samples)
        resampled_angles = self.__resample_angles(n_samples)
        resampled_phase_function = self.__resample_phase_function(resampled_angles)
        norm_resampled_phase_function = self.__normalize_phase_function(resampled_phase_function, resampled_angles)

        # Since we're forcing c0 = 1 and fitting p = c0 + c1 * L1 + ..., p - 1 = c1 * L1 + ..., which is what we want!
        phase_function_to_fit = norm_resampled_phase_function - 1

        legendre_polynomials = self.__make_legendre_polynomials(n_moments, n_samples, resampled_angles)
        normal_matrix = self.__make_normal_matrix(legendre_polynomials, phase_function_to_fit)
        normal_vector = self.__make_normal_vector(legendre_polynomials, phase_function_to_fit)
        cholesky_factorization = self.__cholesky_decomposition(normal_matrix)
        first_solution = self.__solve_first_system(cholesky_factorization, normal_vector)
        second_solution = self.__solve_second_system(cholesky_factorization, first_solution)
        fit_coefficients = self.__filter_negative_coefficients(second_solution)
        return fit_coefficients

    def __check_inputs_are_expected(self, n_moments, n_samples):
        self.__check_input_is_int(n_moments, 'n_moments')
        self.__check_input_is_int(n_samples, 'n_samples')
        self.__check_samples_greater_than_moments(n_moments, n_samples)

    @staticmethod
    def __check_input_is_int(input_variable, variable_name):
        if not isinstance(input_variable, int):
            raise TypeError(f'{variable_name} must be an int')

    @staticmethod
    def __check_samples_greater_than_moments(n_moments, n_samples):
        if not n_samples >= n_moments:
            raise ValueError('n_samples must be >= n_moments')

    def __resample_angles(self, n_samples):
        return np.linspace(0, self.angles[-1], num=n_samples)

    def __resample_phase_function(self, resampled_theta):
        f = interpolate.interp1d(self.mu, self.empirical_phase_function)
        resampled_mu = np.cos(resampled_theta)
        return f(resampled_mu)

    @staticmethod
    def __normalize_phase_function(interp_phase_function, resampled_theta):
        resampled_norm = np.abs(integrate.simps(interp_phase_function, np.cos(resampled_theta)))
        return 2 * interp_phase_function / resampled_norm

    @staticmethod
    def __make_legendre_polynomials(n_moments, n_samples, resampled_theta):
        # Note: This returns a 2D array. The 0th index is the i+1 polynomial and the 1st index is the angle.
        # So index [2, 6] will be the 3rd Legendre polynomial (L3) evaluated at the 6th angle

        # Make a 2D array with 1s on the diagonal to pick out only the desired Legendre polynomial
        diagonal = np.diag(np.ones(n_moments))
        dummy_coeff = np.zeros((n_moments, n_samples))
        dummy_coeff[:, :n_moments] = diagonal

        # Evaluate the polynomials at the input angles. I have no idea why legval resizes the output array,
        # so slice off the 0th moment (which we aren't fitting), and only get up to n_moments
        return np.polynomial.legendre.legval(np.cos(resampled_theta), dummy_coeff)[1:n_moments, :]

    @staticmethod
    def __make_normal_matrix(legendre_coefficients, phase_function_to_fit):
        return np.sum(legendre_coefficients[:, None, :] * legendre_coefficients[None, :, :] / phase_function_to_fit**2,
                      axis=-1)

    @staticmethod
    def __make_normal_vector(legendre_coefficients, phase_function_to_fit):
        return np.sum(legendre_coefficients / phase_function_to_fit, axis=-1)

    @staticmethod
    def __cholesky_decomposition(normal_matrix):
        return np.linalg.cholesky(normal_matrix)

    @staticmethod
    def __solve_first_system(cholesky_factorization, normal_vector):
        return np.linalg.solve(cholesky_factorization, normal_vector)

    @staticmethod
    def __solve_second_system(cholesky_factorization, first_solution):
        return np.linalg.solve(cholesky_factorization.T, first_solution)

    @staticmethod
    def __filter_negative_coefficients(second_solution):
        coefficients = np.concatenate((np.array([1]), second_solution))
        first_negative_index = np.argmax(coefficients < 0)
        if first_negative_index:
            print(f"Setting coefficients to zero starting with coefficient {first_negative_index}")
            coefficients[first_negative_index:] = 0
        return coefficients
