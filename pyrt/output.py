import numpy as np


def empty_albedo_medium(n_polar: int) -> np.ndarray:
    """Make empty albedo of the medium array.

    Parameters
    ----------
    n_polar: int
        The number of polar angles.

    Returns
    -------
    np.ndarray
        An empty array.

    """
    return np.zeros((n_polar,))


def empty_diffuse_up_flux(n_user_levels: int) -> np.ndarray:
    """Make empty diffuse up flux array.

    Parameters
    ----------
    n_user_levels: int
        The number of user levels.

    Returns
    -------
    np.ndarray
        An empty array.

    """
    return np.zeros((n_user_levels,))


def empty_diffuse_down_flux(n_user_levels: int) -> np.ndarray:
    """Make empty diffuse downward flux array.

    Parameters
    ----------
    n_user_levels: int
        The number of user levels.

    Returns
    -------
    np.ndarray
        An empty array.

    """
    return np.zeros((n_user_levels,))


def empty_direct_beam_flux(n_user_levels: int) -> np.ndarray:
    """Make empty direct beam flux array.

    Parameters
    ----------
    n_user_levels: int
        The number of user levels.

    Returns
    -------
    np.ndarray
        An empty array.

    """
    return np.zeros((n_user_levels,))


def empty_flux_divergence(n_user_levels: int) -> np.ndarray:
    """Make empty flux divergence array.

    Parameters
    ----------
    n_user_levels: int
        The number of user levels.

    Returns
    -------
    np.ndarray
        An empty array.

    """
    return np.zeros((n_user_levels,))


def empty_intensity(n_polar: int, n_user_levels: int, n_azimuth: int) -> np.ndarray:
    """Make empty intensity array.

    Parameters
    ----------
    n_polar: int
        The number of polar angles.
    n_user_levels: int
        The number of user levels.
    n_azimuth: int
        The number of azimuth angles.

    Returns
    -------
    np.ndarray
        An empty array.

    """
    return np.zeros((n_polar, n_user_levels, n_azimuth))


def empty_mean_intensity(n_user_levels: int) -> np.ndarray:
    """Make empty mean intensity array.

    Parameters
    ----------
    n_user_levels: int
        The number of user levels.

    Returns
    -------
    np.ndarray
        An empty array.

    """
    return np.zeros((n_user_levels,))


def empty_transmissivity_medium(n_polar: int) -> np.ndarray:
    """Make empty transmissivity of the medium array.

    Parameters
    ----------
    n_polar: int
        The number of polar angles.

    Returns
    -------
    np.ndarray
        An empty array.

    """
    return np.zeros((n_polar,))
