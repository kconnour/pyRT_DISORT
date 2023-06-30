import numpy as np


def empty_albedo_medium(n_polar: int) -> np.ndarray:
    return np.zeros((n_polar,))


def empty_diffuse_up_flux(n_user_levels: int) -> np.ndarray:
    return np.zeros((n_user_levels,))


def empty_diffuse_down_flux(n_user_levels: int) -> np.ndarray:
    return np.zeros((n_user_levels,))


def empty_direct_beam_flux(n_user_levels: int) -> np.ndarray:
    return np.zeros((n_user_levels,))


def empty_flux_divergence(n_user_levels: int) -> np.ndarray:
    return np.zeros((n_user_levels,))


def empty_intensity(n_polar: int, n_user_levels: int, n_azimuth: int) -> np.ndarray:
    return np.zeros((n_polar, n_user_levels, n_azimuth))


def empty_mean_intensity(n_user_levels: int) -> np.ndarray:
    return np.zeros((n_user_levels,))


def empty_transmissivity_medium(n_polar: int) -> np.ndarray:
    return np.zeros((n_polar,))
