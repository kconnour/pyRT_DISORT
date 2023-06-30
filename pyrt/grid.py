import numpy as np
from numpy.typing import ArrayLike


def regrid(array: np.ndarray, particle_size_grid: np.ndarray, wavelength_grid,
        particle_sizes: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Regrid the input array onto a new particle size and wavelength grid
    using nearest-neighbor interpolation.

    Parameters
    ----------
    array: np.ndarray
        The array to regrid.
    particle_size_grid: np.ndarray
        The particle size grid.
    wavelength_grid: np.ndarray
        The wavelength grid.
    particle_sizes: np.ndarray
        The particle sizes to regrid the array on to.
    wavelengths: np.ndarray
        The wavelengths to regrid the array on to.

    Returns
    -------
    np.ndarray
        Regridded array of shape (..., particle_sizes, wavelengths)

    """

    reff_indices = _get_nearest_indices(particle_size_grid, particle_sizes)
    wav_indices = _get_nearest_indices(wavelength_grid, wavelengths)
    return np.take(np.take(array, reff_indices, axis=-2),
                   wav_indices, axis=-1)


def _get_nearest_indices(grid: np.ndarray, values: np.ndarray) \
        -> np.ndarray:
    # grid should be 1D; values can be ND
    return np.abs(np.subtract.outer(grid, values)).argmin(0)
