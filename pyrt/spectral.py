import numpy as np
from numpy.typing import ArrayLike


def wavenumber(wavelength: ArrayLike) -> np.ndarray:
    r"""Convert wavelengths [microns] to wavenumber
    [:math:`\frac{1}{\text{cm}}`].

    Parameters
    ----------
    wavelength: ArrayLike
        N-dimensional array of wavelengths.

    Returns
    -------
    np.ndarray
        N-dimensional array of wavenumbers.

    Examples
    --------
    Convert wavelengths to wavenumbers.

    >>> import numpy as np
    >>> import pyrt
    >>> wavs = [1, 2, 3]
    >>> pyrt.wavenumber(wavs)
    array([10000.        ,  5000.        ,  3333.33333333])

    This function can handle arrays of any shape.

    >>> wavs = np.ones((10, 20, 30))
    >>> pyrt.wavenumber(wavs).shape
    (10, 20, 30)

    """
    return 10 ** 4 / np.asarray(wavelength)
