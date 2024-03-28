import numpy as np

from pyrt.grid import regrid


def extinction_ratio(extinction_cross_section: np.ndarray,
                     particle_size_grid: np.ndarray,
                     wavelength_grid: np.ndarray,
                     wavelength_reference: float) -> np.ndarray:
    """Make a grid of extinction cross-section ratios.

    This is the extinction cross-section at the input wavelengths divided by
    the extinction cross-section at the reference wavelength.

    Parameters
    ----------
    extinction_cross_section: np.ndarray
        2-dimensional array of extinction cross-sections.
    particle_size_grid: np.ndarray
        1-dimensional array of particle sizes corresponding to the first axis
        of extinction_cross_section.
    wavelength_grid: np.ndarray
        1-dimensional array of wavelengths [microns] corresponding to the second
        axis of ``extinction_cross_section``.
    wavelength_reference: np.ndarray
        The wavelength [microns] to scale everything to.

    Returns
    -------
    np.ndarray
        Array of extinction cross-section ratios. This will retain the shape
        of the original array.

    """
    cext_slice = np.squeeze(regrid(
        extinction_cross_section, particle_size_grid, wavelength_grid,
        particle_size_grid, wavelength_reference))
    return (extinction_cross_section.T / cext_slice).T


def optical_depth(q_profile: np.ndarray, column_density: np.ndarray,
                  extinction_ratio: np.ndarray,
                  column_integrated_od: float) -> np.ndarray:
    """Make the optical depth in each model layer.

    Parameters
    ----------
    q_profile: np.ndarray
        1-dimensional array of volumetric mixing ratios.
    column_density: np.ndarray
        1-dimensional array of column densities
        [:math:`\frac{particles}{\text{m^2}}`].
    extinction_ratio: np.ndarray
        2-dimensional array of extinction ratios.
    column_integrated_od: float
        The column integrated optical depth.

    Returns
    -------
    np.ndarray
        2-dimensional array of the optical depth in each model layer at each
        wavelength.

    """
    normalization = np.sum(q_profile * column_density)
    profile = q_profile * column_density * column_integrated_od / normalization
    return (profile * extinction_ratio.T).T
