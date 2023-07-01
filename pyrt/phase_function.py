import warnings

import numpy as np
from numpy.typing import ArrayLike


def decompose(phase_function: ArrayLike, scattering_angles: ArrayLike,
              n_moments: int) -> np.ndarray:
    """Decompose a phase function into Legendre coefficients.

    Parameters
    ----------
    phase_function: ArrayLike
        1-dimensional array of the phase function to decompose.
    scattering_angles: ArrayLike
        1-dimensional array of the scattering angles [degrees]. This array must
        have the same shape as ``phase_function`` and the angles should be
        monotonically increasing.
    n_moments: int
        The number of moments to decompose the phase function into. This
        value must be smaller than the number of points in the phase
        function.

    Returns
    -------
    np.ndarray
        1-dimensional array of Legendre coefficients of the decomposed phase
        function with a shape of ``(moments,)``.

    Examples
    --------
    Construct a Henyey-Greenstein phase function and decompose it into 129
    moments.

    >>> import numpy as np
    >>> import pyrt
    >>> scattering_angles = np.arange(181)
    >>> pf = pyrt.construct_henyey_greenstein(0.5, scattering_angles) * 4 * np.pi
    >>> coeff = pyrt.decompose(pf, scattering_angles, 129)

    Compare the result of this general decomposition to the result of the
    formula for the Henyey-Greenstein decomposition.

    >>> coeff0 = pyrt.henyey_greenstein_legendre_coefficients(0.5, 129)
    >>> np.amax(np.abs(coeff - coeff0)) < 1e-10
    True

    """
    # Why does the last test check a boolean and not a number? When I do CI,
    #  I find that it's OS dependent regarding what answer that test spits out
    #  and therefore, my tests pass or fail depending on what the exact OS is.
    #  So, I made it a bool so all OSs can agree the number is small.
    def _make_legendre_polynomials(scat_angles, n_mom) -> np.ndarray:
        """Make an array of Legendre polynomials at the scattering angles.

        Notes
        -----
        This returns a 2D array. The 0th index is the i+1 polynomial and the
        1st index is the angle. So index [2, 6] will be the 3rd Legendre
        polynomial (L3) evaluated at the 6th angle

        """
        ones = np.ones((n_mom, scat_angles.shape[0]))

        # This creates an MxN array with 1s on the diagonal and 0s elsewhere
        diag_mask = np.triu(ones) + np.tril(ones) - 1

        # Evaluate the polynomials at the input angles. I don't know why
        return np.polynomial.legendre.legval(
            np.cos(scat_angles), diag_mask)[1:n_mom, :]

    def _make_normal_matrix(phase_func, legendre_poly: np.ndarray) -> np.ndarray:
        return np.sum(
            legendre_poly[:, None, :] * legendre_poly[None, :, :] / phase_func ** 2,
            axis=-1)

    def _make_normal_vector(phase_func, legendre_poly: np.ndarray) -> np.ndarray:
        return np.sum(legendre_poly / phase_func, axis=-1)

    pf = np.asarray(phase_function)
    sa = np.asarray(scattering_angles)
    sa = np.radians(sa)
    try:
        # Subtract 1 since I'm forcing c0 = 1 in the equation
        # P(x) = c0 + c1*L1(x) + ... for DISORT
        pf -= 1
        lpoly = _make_legendre_polynomials(sa, n_moments)
        normal_matrix = _make_normal_matrix(pf, lpoly)
        normal_vector = _make_normal_vector(pf, lpoly)
        cholesky = np.linalg.cholesky(normal_matrix)
        first_solution = np.linalg.solve(cholesky, normal_vector)
        second_solution = np.linalg.solve(cholesky.T, first_solution)
        coeff = np.concatenate((np.array([1]), second_solution))
        return coeff
    except np.linalg.LinAlgError as lae:
        message = 'The inputs did not make a positive definite matrix.'
        raise ValueError(message) from lae


def fit_asymmetry_parameter(phase_function: ArrayLike,
                            scattering_angles: ArrayLike) -> float:
    r"""Fit an asymmetry parameter to a phase function.

    Parameters
    ----------
    phase_function: ArrayLike
        1-dimensional phase function.
    scattering_angles: ArrayLike
        1-dimensional scattering angles [degrees] corresponding to each value
        in ``phase_function``.

    Returns
    -------
    float
        N-dimensional array of asymmetry parameters with a shape of
        ``phase_function.shape[1:]``.

    Notes
    -----
    The asymmetry parameter is defined as

    .. math::
       g \equiv \frac{1}{4 \pi} \int_{4\pi} p(\theta)\cos(\theta) d\Omega

    where :math:`g` is the asymmetry parameter, :math:`p` is the phase
    function, :math:`\theta` is the scattering angle, and :math:`\Omega` is
    the solid angle. This is essentially the expectation value of the phase
    function.

    Examples
    --------
    Make a phase function from a known asymmetry parameter, then see how well
    this function recovers the asymmetry parameter.

    >>> import numpy as np
    >>> import pyrt
    >>> g = 0.8
    >>> sa = np.linspace(0, 180, num=181)
    >>> pf = pyrt.construct_henyey_greenstein(g, sa) * 4 * np.pi  # 4 pi is a normalization factor
    >>> fit_g = pyrt.fit_asymmetry_parameter(pf, sa)
    >>> round(fit_g, 5)
    0.83473

    It's not completely terrible but not particularly inspiring. The error is
    due to the coarse resolution. Increasing the number of points in the
    phase function can reduce the error.

    >>> sa = np.linspace(0, 180, num=18100)
    >>> pf = pyrt.construct_henyey_greenstein(g, sa) * 4 * np.pi
    >>> fit_g = pyrt.fit_asymmetry_parameter(pf, sa)
    >>> round(fit_g, 5)
    0.80034

    """
    pf = np.asarray(phase_function)
    sa = np.asarray(scattering_angles)

    cos_sa = np.array(np.cos(np.radians(sa)))
    mid_sa = cos_sa[:-1] + np.diff(cos_sa)
    mid_pf = np.interp(np.flip(mid_sa), np.flip(cos_sa), pf)

    expectation_pf = mid_pf.T * mid_sa
    # Divide by 2 because g = 1/(4*pi) * integral but the azimuth angle
    #  integral = 2*pi so the factor becomes 1/2
    return np.sum((expectation_pf * np.abs(np.diff(cos_sa))).T / 2, axis=0)


def set_negative_coefficients_to_0(coefficients: ArrayLike) -> np.ndarray:
    """Set all Legendre coefficients to 0 after the first negative coefficient.

    Parameters
    ----------
    coefficients: ArrayLike
        1-dimensional array of Legendre coefficients.

    Returns
    -------
    np.ndarray
        1-dimensional array zeroed coefficients. If no coefficients are
        negative, this is identical to the input array.

    """
    coeff = np.copy(np.asarray(coefficients))
    if not np.any(coeff < 0):
        return coeff

    idx = np.argmax(coeff < 0)
    coeff[idx:] = 0
    return coeff


def construct_henyey_greenstein(asymmetry_parameter: float,
                                scattering_angles: ArrayLike) -> np.ndarray:
    r"""Construct a Henyey-Greenstein phase function.

    Parameters
    ----------
    asymmetry_parameter: float
        The Henyey-Greenstein asymmetry parameter. Must be between -1 and 1.
    scattering_angles: ArrayLike
        1-dimensional array of scattering angles [degrees].

    Returns
    -------
    np.ndarray
        1-dimensiona phase function corresponding to each value in
        ``scattering_angles``.

    Notes
    -----
    The Henyey-Greenstein phase function (per solid angle) is defined as

    .. math::

       p(\theta) = \frac{1}{4\pi} \frac{1 - g^2}
                    {[1 + g^2 - 2g \cos(\theta)]^\frac{3}{2}}

    where :math:`p` is the phase function, :math:`\theta` is the scattering
    angle, and :math:`g` is the asymemtry parameter.

    .. warning::
       The normalization for the Henyey-Greenstein phase function is not the
       same as for a regular phase function. For this phase function,

       .. math::
          \int_{4\pi} p(\theta) = 1

       *not* 4 :math:`\pi`! To normalize it simply multiply the output by
       4 :math:`\pi`.

    Examples
    --------
    Construct a Henyey-Greenstein phase function.

    >>> import numpy as np
    >>> import pyrt
    >>> scattering_angles = np.arange(181)
    >>> g = 0.5
    >>> hg_pf = pyrt.construct_henyey_greenstein(g, scattering_angles)
    >>> hg_pf.shape
    (181,)

    """
    scattering_angles = np.asarray(scattering_angles)
    asymmetry_parameter = np.asarray(asymmetry_parameter)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        denominator = (1 + asymmetry_parameter ** 2 -
                       2 * asymmetry_parameter *
                       np.cos(np.radians(scattering_angles))) ** (3 / 2)
        return 1 / (4 * np.pi) * (1 - asymmetry_parameter ** 2) / denominator


def henyey_greenstein_legendre_coefficients(
        asymmetry_parameter: float, n_coefficients: int) -> np.ndarray:
    r"""Get the Legendre coefficients of a Henyey-Greenstein phase function.

    Parameters
    ----------
    asymmetry_parameter: float
        The Henyey-Greenstein asymmetry parameter. Must be between -1 and 1.
    n_coefficients: int
        The number of coefficients to keep.

    Returns
    -------
    np.ndarray
        1-dimensional arrray of Legendre coefficients up to the specificed
        number of coefficients.

    Notes
    -----
    The Henyey-Greenstein phase function can be decomposed as follows:

    .. math::
       p(\mu) = \sum_{n=0}^{\infty} (2n + 1)g^n P_n(\mu)

    where :math:`p` is the phase function, :math:`\mu` is the cosine of the
    scattering angle, :math:`n` is the moment number, :math:`g` is the
    asymmetry parameter, and :math:`P_n(\mu)` is the :math:`n`:sup:`th`
    Legendre polynomial.

    Examples
    --------
    Get the first 129 coefficients of the Henyey-Greenstein phase function for
    a given asymmetry parameter.

    >>> import numpy as np
    >>> import pyrt
    >>> g = 0.5
    >>> coeff = pyrt.henyey_greenstein_legendre_coefficients(g, 129)
    >>> coeff.shape
    (129,)

    Construct a Henyey-Greenstein phase function, decompose it, and see how
    this result compares to the analytic decomposition performed above.

    >>> ang = np.linspace(0, 180, num=181)
    >>> pf = pyrt.construct_henyey_greenstein(g, ang) * 4 * np.pi  # normalize it
    >>> lc = pyrt.decompose(pf, ang, 129)
    >>> np.amax(np.abs(lc - coeff)) < 1e-10
    True

    """
    asymmetry_parameter = np.asarray(asymmetry_parameter)
    coeff = np.arange(n_coefficients)
    return (2 * coeff + 1) * asymmetry_parameter ** coeff
