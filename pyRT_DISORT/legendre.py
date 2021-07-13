"""The legendre module provides functions for making Legendre decompositions."""
from warnings import warn
import numpy as np
from scipy import integrate


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


class NMoments(int):
    """Designate that a number represents the number of moments.

    Parameters
    ----------
    value
        The number of moments to use.

    Raises
    ------
    TypeError
        Raised if the input cannot be converted into an int.
    ValueError
        Raised if the number of samples is not positive.
    """

    def __new__(cls, value: int, *args, **kwargs):
        if value <= 0:
            raise ValueError("The number of moments must be positive.")
        return super(cls, cls).__new__(cls, value)


class NSamples(int):
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
    """ A collection of methods for decomposing a phase function into
    polynomials.

    This class can decompose a phase function into Legendre polynomials. In
    principle it can decompose any function into those polynomials but it's set
    up specifically for that task.

    Parameters
    ----------
    phase_function
    angles
    n_moments
    n_samples

    Raises
    ------
    TypeError
        Raised if any of the inputs cannot be cast to the correct shape.
    ValueError
        Raised in any of the inputs are unphysical.

    """
    def __init__(self, phase_function: np.ndarray, angles: np.ndarray,
                 n_moments: int, n_samples: int):
        self.bundle = _PhaseFunctionBundle(phase_function, angles)
        self.n_moments = NMoments(n_moments)
        self.n_samples = NSamples(n_samples)

        # Fit P(x) = c0 + c1*L1(x) + ... where I force c0 = 1 for DISORT
        self.resamp_norm_pf = \
            self.bundle.normalize_resampled_phase_function(self.n_samples) - 1
        self.lpoly = self._make_legendre_polynomials()

    def _make_legendre_polynomials(self) -> np.ndarray:
        """Make an array of Legendre polynomials at the input angles.

        Notes
        -----
        This returns a 2D array. The 0th index is the i+1 polynomial and the
        1st index is the angle. So index [2, 6] will be the 3rd Legendre
        polynomial (L3) evaluated at the 6th angle

        """
        resampled_theta = self.bundle.resample_angles(self.n_samples)
        ones = np.ones((self.n_moments, self.n_samples))

        # This creates an MxN array with 1s on the diagonal and 0s elsewhere
        diag_mask = np.triu(ones) + np.tril(ones) - 1

        # Evaluate the polynomials at the input angles. I don't know why
        return np.polynomial.legendre.legval(
            np.cos(resampled_theta), diag_mask)[1:self.n_moments, :]

    def decompose(self) -> np.ndarray:
        """Decompose the phase function into its Legendre moments.

        """
        normal_matrix = self._make_normal_matrix()
        normal_vector = self.__make_normal_vector()
        cholesky = np.linalg.cholesky(normal_matrix)
        first_solution = np.linalg.solve(cholesky, normal_vector)
        second_solution = np.linalg.solve(cholesky.T, first_solution)
        coeff = np.concatenate((np.array([1]), second_solution))
        self._warn_if_negative_coefficients(coeff)
        return coeff

    def _make_normal_matrix(self) -> np.ndarray:
        return np.sum(self.lpoly[:, None, :] * self.lpoly[None, :, :] /
                      self.resamp_norm_pf**2, axis=-1)

    def __make_normal_vector(self) -> np.ndarray:
        return np.sum(self.lpoly / self.resamp_norm_pf, axis=-1)

    def _warn_if_negative_coefficients(self, coeff):
        first_negative_index = self._get_first_negative_coefficient_index(coeff)
        if first_negative_index:
            message = f'Coefficient {first_negative_index} is negative.'
            warn(message)

    @staticmethod
    def _get_first_negative_coefficient_index(coeff: np.ndarray) -> np.ndarray:
        return np.argmax(coeff < 0)

    def filter_negative_coefficients(self, coeff: np.ndarray) -> np.ndarray:
        """Set all coefficients at the first negative one to 0.

        Parameters
        ----------
        coeff
            The Legendre coefficients.

        """
        first_negative_index = self._get_first_negative_coefficient_index(coeff)
        if first_negative_index:
            coeff[first_negative_index:] = 0
        return coeff


def decompose_phase_function(
        phase_function: np.ndarray, angles: np.ndarray,
        n_moments: int, n_samples: int) -> np.ndarray:
    """Decompose a phase function into its Legendre moments.

    Parameters
    ----------
    phase_function
        The phase function to decompose.
    angles
        The angles where the phase function is defined.
    n_moments
        The number of Legendre moments to decompose the phase function into.
    n_samples
        The number of samples to use for the resampling. Must be >= the number
        of moments.

    Examples
    --------
    Let's suppose we have a Henyey-Greenstein phase function with asymmetry
    parameter of 0.6. Let's create its first 65 moments along with the angles
    over which it's defined.

    >>> from pyRT_DISORT.aerosol import HenyeyGreenstein
    >>> analytic_moments = HenyeyGreenstein(0.6).legendre_decomposition(65)
    >>> angles = np.radians(np.linspace(0, 180, num=181))

    We can convert it to a phase fuction with the following:

    >>> import numpy as np
    >>> phase = np.polynomial.legendre.legval(np.cos(angles), analytic_moments)

    If we decompose the phase fuction into Legendre coefficients, we should get
    what we started with. Let's decompose the phase fuction back into 65 moments
    and use 360 samples to make sure the resolution is good.

    >>> from pyRT_DISORT.legendre import decompose_phase_function
    >>> decomp_moments = decompose_phase_function(phase, angles, 65, 360)
    >>> np.amax((analytic_moments - decomp_moments)**2)
    1.6443715589020334e-07

    We've recovered the original moments pretty well!

    """
    ld = _LegendreDecomposer(phase_function, angles, n_moments, n_samples)
    return ld.decompose()
