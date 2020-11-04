# 3rd-party imports
import numpy as np
from scipy import integrate, interpolate
import time

# Local imports
from pyRT_DISORT.preprocessing.utilities.utilities import ExternalFile


class PhaseFunction:
    """ Do stuff with a phase function"""
    def __init__(self, input_phase_function, degrees=True):
        """
        Parameters
        ----------
        input_phase_function: np.ndarray
            A 2D array where the first column is the scattering angle and the second is the (possibly unnormalized)
            phase function
        degrees: bool
            Denote if the first column of phase_function is in degrees. Default is True
        """
        self.input_phase_function = input_phase_function
        self.degrees = degrees

        if self.degrees:
            self.input_theta_degrees = self.input_phase_function[:, 0]
            self.input_theta_radians = np.radians(self.input_theta_degrees)
        else:
            self.input_theta_radians = self.input_phase_function[:, 0]

        self.mu = np.cos(self.input_theta_radians)
        self.phase_function = self.input_phase_function[:, 1]
        self.n_angles = len(self.mu)

    def create_legendre_coefficients(self, n_moments, n_samples):
        """ Create the Legendre coefficients for this phase function

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
        self.__check_create_coefficient_inputs(n_moments, n_samples)
        self.__check_samples_greater_than_moments(n_moments, n_samples)

        resampled_theta = self.__resample_angles(n_samples)
        norm_resampled_phase_function = self.__normalize_phase_function(self.__resample_phase_function(resampled_theta),
                                                                        resampled_theta)

        # Since we're forcing c0 = 1 and fitting p = c0 + c1 * L1 + ..., p - 1 = c1 * L1 + ..., which is what we want!
        phase_function_to_fit = norm_resampled_phase_function - 1

        legendre_polynomials = self.__make_legendre_polynomials(n_moments, n_samples, resampled_theta)
        normal_matrix = self.__make_normal_matrix(legendre_polynomials, phase_function_to_fit)
        normal_vector = self.__make_normal_vector(legendre_polynomials, phase_function_to_fit)
        cholesky_factorization = self.__cholesky_decomposition(normal_matrix)
        first_solution = self.__solve_first_system(cholesky_factorization, normal_vector)
        second_solution = self.__solve_second_system(cholesky_factorization, first_solution)
        fit_coefficients = self.__filter_negative_coefficients(second_solution)
        return fit_coefficients

    @staticmethod
    def __check_create_coefficient_inputs(n_moments, n_samples):
        assert isinstance(n_moments, int), 'n_moments must be an int.'
        assert isinstance(n_samples, int), 'n_samples must be an int.'

    @staticmethod
    def __check_samples_greater_than_moments(n_moments, n_samples):
        assert n_samples >= n_moments, 'n_samples must be >= n_moments'

    def __resample_angles(self, n_samples):
        return np.linspace(0, self.input_theta_radians[-1], num=n_samples)

    def __resample_phase_function(self, resampled_theta):
        f = interpolate.interp1d(self.mu, self.phase_function)
        resampled_mu = np.cos(resampled_theta)
        return f(resampled_mu)

    @staticmethod
    def __normalize_phase_function(interp_phase_function, resampled_theta):
        resampled_norm = np.abs(integrate.simps(interp_phase_function, np.cos(resampled_theta)))
        return 2 * interp_phase_function / resampled_norm

    @staticmethod
    def __make_legendre_polynomials(n_moments, n_samples, resampled_theta):
        """ Evaulate Legendre polynomials

        Parameters
        ----------
        n_moments
        n_samples
        resampled_theta

        Returns
        -------
        np.ndarray
            A 2D array. The 0th index is the i+1 polynomial and the 1st index is the angle. So [2, 6] will be the
            3rd Legendre polynomial (L3) evaluated at the 7th angle
        """
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


t0 = time.time()
for i in range(10):
    print(i)
    file = ExternalFile('/home/kyle/Downloads/ice_shape001_r030_00321.dat', header_lines=3, text1d=False)
    pf = PhaseFunction(file.array)
    a = pf.create_legendre_coefficients(128, 361)

t1 = time.time()
print(t1 - t0)
print(a)
