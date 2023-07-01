"""This module provides functions for creating geometric angles.
"""
import numpy as np
from numpy.typing import ArrayLike


def azimuth(incidence_angles: ArrayLike, emission_angles: ArrayLike,
            phase_angles: ArrayLike) -> np.ndarray:
    r"""Construct azimuth angles from a set of incidence, emission, and phase
    angles.

    Parameters
    ----------
    incidence_angles: ArrayLike
        N-dimensional array of incidence (solar zenith) angles [degrees]. All
        values should be between 0 and 90.
    emission_angles: ArrayLike
        N-dimensional array of emission (emergence) angles [degrees]. All
        values should be between 0 and 90.
    phase_angles: ArrayLike
        N-dimensional array of phase angles [degrees]. All values should be
        between 0 and 180.

    Returns
    -------
    np.ndarray
        N-dimensional array of azimuth angles [degrees].

    Notes
    -----
    This function can handle arrays of any shape, but all input arrays should
    have the same shape.

    Examples
    --------
    Create the azimuth angle from a scalar set of angles.

    >>> import numpy as np
    >>> import pyrt
    >>> pyrt.azimuth(10, 15, 20)
    array(75.09711684)

    You can also input an N-dimensional set of angles. This may be useful for
    spacecraft observations, where each pixel in a 2D image has a unique set
    of incidence, emission, and phase angles.

    >>> data_shape = (15, 20)
    >>> incidence = np.ones(data_shape) * 10
    >>> emission = np.ones(data_shape) * 15
    >>> phase = np.ones(data_shape) * 20
    >>> pyrt.azimuth(incidence, emission, phase).shape
    (15, 20)

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        tmp_arg = np.true_divide(
            np.cos(np.radians(phase_angles)) -
            np.cos(np.radians(emission_angles)) *
            np.cos(np.radians(incidence_angles)),
            np.sin(np.radians(emission_angles)) *
            np.sin(np.radians(incidence_angles)))
    tmp_arg = np.asarray(tmp_arg)
    tmp_arg[~np.isfinite(tmp_arg)] = -1
    d_phi = np.arccos(np.clip(tmp_arg, -1, 1))
    return np.array(180 - np.degrees(d_phi))
