import numpy as np


def conrath(altitude: np.ndarray, q0: float, scale_height: np.ndarray, nu: float) -> float:
    """Construct a Conrath profile.

    Parameters
    ----------
    altitude: np.ndarray
        The altitudes [m] at which to construct the profile.
    q0: float
        The surface volumetric mixing ratio.
    scale_height: np.ndarray
        The atmospheric scale height [m] at each altitude.
    nu: float
        The nu parameter.

    Returns
    -------
    np.ndarray
        The Conrath profile at each altitude.

    Examples
    --------
    Construct a Conrath profile from a set of input parameters.

    >>> import numpy as np
    >>> import pyrt
    >>> altitude = np.linspace(100000, 0, num=15)
    >>> profile = conrath(altitude, 1, 10000, 0.1)
    >>> profile.shape
    (15,)

    """
    return q0 * np.exp(nu * (1 - np.exp(altitude / scale_height)))
