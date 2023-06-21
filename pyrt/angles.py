"""This module provides functions for creating geometric angles.
"""
import numpy as np
from numpy.typing import ArrayLike


def azimuth(incidence_angles: ArrayLike,
            emission_angles: ArrayLike,
            phase_angles: ArrayLike) \
        -> np.ndarray:
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

    Examples
    --------
    Create the azimuth angles from a set of angles.

    >>> import numpy as np
    >>> import pyrt
    >>> incidence = np.array([10, 20])
    >>> emission = np.array([15, 25])
    >>> phase = np.array([20, 30])
    >>> pyrt.azimuth(incidence, emission, phase)
    array([75.09711684, 95.70740729])

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        tmp_arg = np.true_divide(
            np.cos(np.radians(phase_angles)) - np.cos(np.radians(emission_angles)) *
            np.cos(np.radians(incidence_angles)),
            np.sin(np.radians(emission_angles)) * np.sin(np.radians(incidence_angles)))
        tmp_arg = np.asarray(tmp_arg)
        tmp_arg[~np.isfinite(tmp_arg)] = -1
        d_phi = np.arccos(np.clip(tmp_arg, -1, 1))
    return np.array(180 - np.degrees(d_phi))
