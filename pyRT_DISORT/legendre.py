"""The legendre module provides functions for making Legendre decompositions."""
import numpy as np
from scipy import integrate, interpolate


class _PhaseFunction(np.ndarray):
    """Designate an array as representing a phase function.

    Parameters
    ----------
    array
        Any phase function.

    Raises
    ------
    ValueError
        Raised if the input array is not 1D or if it contains negative,
        infinite, or NaN values.

    """
    def __new__(cls, array: np.ndarray):
        obj = np.asarray(array).view(cls)
        return obj

    def __array_finalize__(self, obj: np.ndarray):
        self.__raise_value_error_if_array_is_not_positive_finite(obj)
        self.__raise_value_error_if_array_is_not_1d(obj)

    @staticmethod
    def __raise_value_error_if_array_is_not_positive_finite(obj: np.ndarray):
        if ((obj < 0) | ~np.isfinite(obj)).any():
            message = 'Some values in the phase function are negative or ' \
                      'not finite.'
            raise ValueError(message)

    @staticmethod
    def __raise_value_error_if_array_is_not_1d(obj: np.ndarray):
        if obj.ndim != 1:
            message = 'The phase function should be 1D.'
            raise ValueError(message)


class _PhaseFunctionAngles(np.ndarray):
    """Designate an array as representing the angles where the phase function is
    defined.

    Parameters
    ----------
    array
        The angles where the phase function is defined.

    Raises
    ------
    ValueError
        Raised if the input array is not 1D; if contains negative, infinite,
        or NaN values; or if it is not monotonically increasing.

    """

    def __new__(cls, array: np.ndarray):
        obj = np.asarray(array).view(cls)
        return obj

    def __array_finalize__(self, obj: np.ndarray):
        self.__raise_value_error_if_array_is_not_between_0_and_pi(obj)
        self.__raise_value_error_if_array_is_not_1d(obj)
        self.__raise_value_error_if_array_is_not_monotonically_inc(obj)

    @staticmethod
    def __raise_value_error_if_array_is_not_between_0_and_pi(obj: np.ndarray):
        if ((obj < 0) | (obj > np.pi)).any():
            message = 'Some values in the phase function angles are not ' \
                      'between 0 and pi.'
            raise ValueError(message)

    @staticmethod
    def __raise_value_error_if_array_is_not_1d(obj: np.ndarray):
        if obj.ndim != 1:
            message = 'The phase function angles should be 1D.'
            raise ValueError(message)

    @staticmethod
    def __raise_value_error_if_array_is_not_monotonically_inc(obj: np.ndarray):
        if ~np.all(np.diff(obj) > 0):
            message = 'The phase function angles must be monotonically ' \
                      'increasing.'
            raise ValueError(message)


class _PhaseFunctionBundle:
    """Bundle together the phase function and its associated angles.

    Parameters
    ----------
    phase_function
        The phase function.
    angles
        The angles where the phase function is defined.

    Raises
    ------
    ValueError
        Raised if either :code:`phase_function` or :code:`angles` are unphysical
        or if they do not have the same shape.

    """
    def __init__(self, phase_function: np.ndarray, angles: np.ndarray):
        self.pf = _PhaseFunction(phase_function)
        self.angles = _PhaseFunctionAngles(angles)

        self.__raise_value_error_if_inputs_are_not_same_size()

    def __raise_value_error_if_inputs_are_not_same_size(self):
        if self.pf.shape != self.angles.shape:
            message = 'The phase function and its associated angles must ' \
                      'have the same shape.'
            raise ValueError(message)

    def resample_phase_function(self, n_samples) -> np.ndarray:
        return np.interp(self.resample_angles(n_samples), self.angles, self.pf)

    def resample_angles(self, n_samples) -> np.ndarray:
        return np.linspace(self.angles[0], self.angles[-1], num=n_samples)

    def normalize_resampled_phase_function(self, n_samples):
        resampled_pf = self.resample_phase_function(n_samples)
        resampled_angles = self.resample_angles(n_samples)
        resampled_norm = np.abs(integrate.simps(resampled_pf,
                                                np.cos(resampled_angles)))
        return 2 * resampled_pf / resampled_norm


class Samples(int):
    """Designate that a number represents the number of samples.

    Parameters
    ----------
    value
        The number of samples to use.

    Raises
    ------
    TypeError
        Raised if the input cannot be converted into an int.
    ValueError
        Raised if the number of samples is not positive.
    """

    def __new__(cls, value: int, *args, **kwargs):
        if value <= 0:
            raise ValueError("The number of samples must be positive.")
        return super(cls, cls).__new__(cls, value)


class _LegendreDecomposer:
    def __init__(self, phase_function: np.ndarray, angles: np.ndarray, n_moments: int, n_samples: int):
        self.bundle = _PhaseFunctionBundle(phase_function, angles)
        self._n_moments = n_moments
        self._n_samples = n_samples

        self.resamp_norm_pf = self.bundle.normalize_resampled_phase_function(n_samples) - 1   # due to not fitting c0 = 1
        self.lpoly = self._make_legendre_polynomials()

    def _make_legendre_polynomials(self):
        """

        Returns
        -------

        Notes
        -----
        This returns a 2D array. The 0th index is the i+1 polynomial and the
        1st index is the angle. So index [2, 6] will be the 3rd Legendre
        polynomial (L3) evaluated at the 6th angle

        """
        resampled_theta = self.bundle.resample_angles(self._n_samples)
        ones = np.ones((self._n_moments, self._n_samples))

        # This creates an MxN array with 1s on the diagonal; 0s elsewhere
        diag_mask = np.triu(ones) + np.tril(ones) - 1

        # Evaluate the polynomials at the input angles. I don't know why
        return np.polynomial.legendre.legval(np.cos(resampled_theta),
                                             diag_mask)[1:self._n_moments, :]

    def decompose(self) -> np.ndarray:
        """

        Returns
        -------

        """
        normal_matrix = self._make_normal_matrix()
        normal_vector = self.__make_normal_vector()
        cholesky_factorization = self._cholesky_decomposition(normal_matrix)
        first_solution = self._solve_first_system(cholesky_factorization, normal_vector)
        second_solution = self._solve_second_system(cholesky_factorization, first_solution)
        fit_coefficients = self._filter_negative_coefficients(second_solution)
        return fit_coefficients

    def _make_normal_matrix(self) -> np.ndarray:
        return np.sum(self.lpoly[:, None, :] * self.lpoly[None, :, :] /
                      self.resamp_norm_pf**2, axis=-1)

    def __make_normal_vector(self) -> np.ndarray:
        return np.sum(self.lpoly / self.resamp_norm_pf, axis=-1)

    @staticmethod
    def _cholesky_decomposition(normal_matrix: np.ndarray) -> np.ndarray:
        return np.linalg.cholesky(normal_matrix)

    @staticmethod
    def _solve_first_system(cholesky_factorization, normal_vector) -> np.ndarray:
        return np.linalg.solve(cholesky_factorization, normal_vector)

    @staticmethod
    def _solve_second_system(cholesky_factorization, first_solution) -> np.ndarray:
        return np.linalg.solve(cholesky_factorization.T, first_solution)

    @staticmethod
    def _filter_negative_coefficients(second_solution) -> np.ndarray:
        coefficients = np.concatenate((np.array([1]), second_solution))
        first_negative_index = np.argmax(coefficients < 0)
        if first_negative_index:
            print(f"Setting coefficients to zero starting with coefficient {first_negative_index}")
            coefficients[first_negative_index:] = 0
        return coefficients


def decompose_phase_function(
        empirical_phase_function: np.ndarray, angles: np.ndarray,
        n_moments: int, n_samples: int) -> np.ndarray:
    """Decompose a phase function into its Legendre moments.

    Parameters
    ----------
    empirical_phase_function
    angles
    n_moments
    n_samples

    """
    ld = _LegendreDecomposer(empirical_phase_function, angles, n_moments, n_samples)
    return ld.decompose()


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
        self.__check_epf_matches_angles_shape()

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


if __name__ == '__main__':
    s = Samples('foo')
    t = s + 56
    print(t, type(t))

    raise SystemExit(9)

    import time
    from scipy.interpolate import interp2d
    from scipy.optimize import minimize
    # 0. Load files
    lut_loc = '/home/kyle/dustssa/kyle_iuvs_2/'

    cext_lut = np.load(f'{lut_loc}cext_lut.npy')
    csca_lut = np.load(f'{lut_loc}csca_lut.npy')  # shape: (3, 4, 8)
    z11_lut = np.load(f'{lut_loc}z11_lut.npy')  # shape: (181, 3, 4, 8)
    ssa_lut = csca_lut / cext_lut

    lut_wavs = np.array([230, 260, 300])
    lut_reff = np.array([1.4, 1.6, 1.8, 2])
    lut_k = np.array([1, 10, 30, 60, 100, 200, 300, 500])

    retrieval15 = np.genfromtxt('/home/kyle/dustssa/1-5microns.csv',
                                delimiter=',')
    retrieval20 = np.genfromtxt('/home/kyle/dustssa/2-0microns.csv',
                                delimiter=',')

    # 1. Turn SSA into k
    k_spectra = np.zeros((4, 19))


    def fit_k(k_guess, wavelength, retr_ssa):
        return (interp(wavelength, k_guess) - retr_ssa) ** 2


    for reff in range(lut_reff.shape[0]):
        interp = interp2d(lut_wavs, lut_k,
                          ssa_lut[:, reff, :].T)  # ssa = f(wavelength, k)

        # Invert: knowing ssa and wavelength, get k
        for i in range(retrieval15[:, 0].shape[0]):
            if reff < 2:
                m = minimize(fit_k, 150,
                             args=(retrieval15[i, 0], retrieval15[i, 1])).x[0]
                # m = np.interp()
            else:
                m = minimize(fit_k, 150,
                             args=(retrieval20[i, 0], retrieval20[i, 1])).x[0]
            k_spectra[reff, i] = m

    # 2. Get c_sca, c_ext, and z11 at k
    new_csca = np.zeros((4, 19))
    new_cext = np.zeros((4, 19))

    for reff in range(lut_reff.shape[0]):
        f = interp2d(lut_wavs, lut_k, csca_lut[:, reff, :].T)
        g = interp2d(lut_wavs, lut_k, cext_lut[:, reff, :].T)
        for w in range(19):
            new_csca[reff, w] = f(retrieval15[w, 0], k_spectra[reff, w])[0]
            new_cext[reff, w] = g(retrieval15[w, 0], k_spectra[reff, w])[0]

    # Get z11 at k
    new_z11 = np.zeros((181, 4, 19))

    for ang in range(181):
        for reff in range(lut_reff.shape[0]):
            f = interp2d(lut_wavs, lut_k, z11_lut[ang, :, reff, :].T)
            for w in range(19):
                new_z11[ang, reff, w] = \
                f(retrieval15[w, 0], k_spectra[reff, w])[0]

    # 3. Make P(theta)
    p = 4 * np.pi * new_z11 / new_csca   # 181, 4, 19

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
    testp = p[:, 0, 0]
    angles = np.radians(np.linspace(0, 180, num=181))

    pf = PhaseFunction(testp, angles)
    a = pf.create_legendre_coefficients(65, 360)

    b = decompose_phase_function(testp, angles, 65, 360)

    print(np.amax((b-a)**2))
