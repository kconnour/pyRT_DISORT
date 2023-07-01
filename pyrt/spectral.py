"""This module provides functions for converting between spectral scales.
"""
import numpy as np
from numpy.typing import ArrayLike


def wavenumber(wavelengths: ArrayLike) -> np.ndarray:
    r"""Convert wavelengths to wavenumbers.

    Parameters
    ----------
    wavelengths: ArrayLike
        N-dimensional array of wavelengths [microns].

    Returns
    -------
    np.ndarray
        N-dimensional array of wavenumbers [:math:`\frac{1}{\text{cm}}`].

    Examples
    --------
    Convert a wavelength to a wavenumber

    >>> import numpy as np
    >>> import pyrt
    >>> pyrt.wavenumber(3)
    array(3333.33333333)

    This function can handle arrays of any shape.

    >>> wavs = np.ones((10, 20, 30))
    >>> pyrt.wavenumber(wavs).shape
    (10, 20, 30)

    """
    return np.array(10 ** 4 / np.asarray(wavelengths))
