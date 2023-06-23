"""This module provides functions for working with Rayleigh scattering.
"""
import numpy as np
from numpy.typing import ArrayLike

from pyrt.spectral import wavenumber
from pyrt.column import Column


def rayleigh_legendre(n_layers: int, n_wavelengths: int) -> np.ndarray:
    r"""Make the generic Legendre decomposition of the Rayleigh scattering
    phase function.

    Parameters
    ----------
    n_layers: int
        The number of layers to make the phase function for.
    n_wavelengths: int
        The number of wavelengths to make

    Returns
    -------
    np.ndarray
        3-dimensional array of the Legendre decomposition of the phase
        function with a shape of ``(3, n_layers, n_wavelengths)``.

    Notes
    -----
    Moment 0 is always 1 and moment 2 is always 0.5. The Rayleigh scattering
    phase function is given by

    .. math::
       P(\theta) = \frac{3}{4} (1 + \cos^2(\theta)).

    Since :math:`P_0(x) = 1` and :math:`P_2(x) = \frac{3x^2 - 1}{2},
    :math:`P(\mu) = P_0(\mu) + 0.5 P_2(\mu).

    Examples
    --------
    Make the Rayleigh scattering phase function for a model with 15 layers
    and 5 wavelengths.

    >>> import pyrt
    >>> rayleigh_pf = pyrt.rayleigh_legendre(15, 5)
    >>> rayleigh_pf.shape
    (3, 15, 5)

    """
    rayleigh_phase_function = np.zeros((3, n_layers, n_wavelengths))
    rayleigh_phase_function[0] = 1
    rayleigh_phase_function[2] = 0.5
    return rayleigh_phase_function


def rayleigh_co2(column_density: ArrayLike, wavelength: ArrayLike) -> Column:
    r"""Compute the Rayleigh CO :sub:`2` Column.

    Parameters
    ----------
    column_density: ArrayLike
        1-dimensional array of the column density in each layer.
    wavelength: ArrayLike
        1-dimensional array of the wavelengths [microns] to compute the optical
        depth at.

    Returns
    -------
    Column
        A Rayleigh column.

    Notes
    -----
    This algorithm computes the optical depth in each layer i by using

    .. math:
       \tau_i = N_i * \sigma_i.

    The molecular cross-section is given by laboratory measurements from
    `Sneep and Ubachs, 2005 <https://doi.org/10.1016/j.jqsrt.2004.07.025>`_.

    Examples
    --------
    Compute the Rayleigh scattering optical depth from (very simplified)
    column densities and wavelengths.

    >>> import numpy as np
    >>> import pyrt
    >>> column_density = np.linspace(10**26, 10**27, num=15)
    >>> wavs = np.linspace(0.2, 1, num=5)
    >>> rayleigh_column = pyrt.rayleigh_co2(column_density, wavs)
    >>> rayleigh_column.optical_depth.shape
    (15, 5)

    Simply sum over the layers to get the column integrated optical depth due
    to Rayleigh scattering.

    >>> np.sum(rayleigh_column.optical_depth, axis=0)
    array([0.78456184, 0.03590366, 0.00672394, 0.00208873, 0.00084833])

    """
    def _molecular_cross_section(wave_number: np.ndarray):
        number_density = 25.47 * 10 ** 18  # laboratory molecules / cm**3
        king_factor = 1.1364 + 25.3 * 10 ** -12 * wave_number ** 2
        index_of_refraction = _co2_index_of_refraction(wave_number)
        return _co2_cross_section(
            number_density, wave_number, king_factor, index_of_refraction) * \
            10 ** -4

    def _co2_index_of_refraction(wave_number: np.ndarray) -> np.ndarray:
        n = 1 + 1.1427 * 10 ** 3 * (
                5799.25 / (128908.9 ** 2 - wave_number ** 2) +
                120.05 / (89223.8 ** 2 - wave_number ** 2) +
                5.3334 / (75037.5 ** 2 - wave_number ** 2) +
                4.3244 / (67837.7 ** 2 - wave_number ** 2) +
                0.00001218145 / (2418.136 ** 2 - wave_number ** 2))
        return n

    def _co2_cross_section(number_density: float, wave_number: np.ndarray,
                           king_factor: np.ndarray,
                           index_of_refraction: np.ndarray) -> np.ndarray:
        coefficient = 24 * np.pi ** 3 * wave_number ** 4 / number_density ** 2
        middle_term = ((index_of_refraction ** 2 - 1) /
                       (index_of_refraction ** 2 + 2)) ** 2
        return coefficient * middle_term * king_factor  # cm**2 / molecule

    wavenum = wavenumber(wavelength)
    column_density = np.asarray(column_density)[:, None]
    mol_cs = _molecular_cross_section(wavenum)[:, None]
    scattering_od = np.multiply(column_density[:, None, :], mol_cs[None, :])
    scattering_od = np.squeeze(scattering_od)
    rayleigh_ssa = np.ones(scattering_od.shape)

    return Column(scattering_od, rayleigh_ssa,
                  rayleigh_legendre(scattering_od.shape[0], wavenum.shape[0]))
