"""The rayleigh module contains structures for computing Rayleigh scattering.
"""
import numpy as np


class Rayleigh:
    """An abstract base class for Rayleigh scattering.

    Rayleigh creates the Legendre coefficient phase function array and holds
    the wavenumbers at which scattering was observed. This is an abstract base
    class from which all other Rayleigh classes are derived; it is not meant
    to be instantiated.

    """

    def __init__(self, altitude_grid: np.ndarray, wavenumbers: np.ndarray) \
            -> None:
        """
        Parameters
        ----------
        altitude_grid
            1D array of layer altitudes where the model is defined.
        wavenumbers
            Array of wavenumbers for the observation. These can be created in
            :class:`.observation.Wavelengths`.

        """
        self._wavenumbers = wavenumbers
        self._grid = altitude_grid

        self.__phase_function = self.__construct_phase_function()

    def __construct_phase_function(self) -> np.ndarray:
        pf = np.zeros((3, len(self._grid), self._wavenumbers.size))
        pf[0, :, :] = 1
        pf[2, :, :] = 0.1
        return pf.reshape(pf.shape[:-1] + self._wavenumbers.shape)

    @property
    def phase_function(self) -> np.ndarray:
        """Get the Legendre decomposition of the phase function.

        Returns
        -------
        np.ndarray
            The decomposed phase function.

        Notes
        -----
        The shape of this array is determined by the inputs. In general, the
        shape will be [n_moments, n_layers, (n_pixels)] (here, n_moments will
        always be 3 since the 0th and 2nd terms are the only terms in the
        expansion). For example, ff altitude_grid has shape [20], and
        wavenumbers has shape [50, 100, 45], this array will have shape
        [3, 20, 50, 100, 45].

        """
        return self.__phase_function


class RayleighCO2(Rayleigh):
    """Construct an object to hold arrays related to CO2 Rayleigh scattering.

    Rayleigh creates the Legendre coefficient phase function array and holds
    the wavenumbers at which scattering was observed. This is an abstract base
    class from which all other Rayleigh classes are derived; it is not meant
    to be instantiated.

    """

    def __init__(self, altitude_grid: np.ndarray, wavenumbers: np.ndarray,
                 column_density_layers: np.ndarray) -> None:
        """
        Parameters
        ----------
        altitude_grid
            1D array of altitudes where the model is defined.
        wavenumbers
            Array of wavenumbers for the observation. These can be created in
            :class:`.observation.Wavelengths`.
        column_density_layers
            1D array of column densities (particles / m**2) for each layer of
            the model. This should be the same length as altitude_grid to be
            useful.

        Notes
        -----
        The values used here are from `Sneep and Ubachs 2005
        <https://doi.org/10.1016/j.jqsrt.2004.07.025>`_

        Due to a typo in the paper, I changed the coefficient to 10**3 when
        using equation 13 for computing the index of refraction

        """
        super().__init__(altitude_grid, wavenumbers)
        self.__scattering_od = \
            self.__calculate_scattering_optical_depths(column_density_layers)

    def __calculate_scattering_optical_depths(
            self, column_density_layers: np.ndarray) -> np.ndarray:
        return np.multiply.outer(column_density_layers,
                                 self.__molecular_cross_section())

    def __molecular_cross_section(self):
        number_density = 25.47 * 10 ** 18  # laboratory molecules / cm**3
        king_factor = 1.1364 + 25.3 * 10 ** -12 * self._wavenumbers ** 2
        index_of_refraction = self.__index_of_refraction()
        return self.__cross_section(
            number_density, king_factor, index_of_refraction) * 10 ** -4

    def __index_of_refraction(self) -> np.ndarray:
        n = 1 + 1.1427 * 10 ** 3 * (
                    5799.25 / (128908.9 ** 2 - self._wavenumbers ** 2) +
                    120.05 / (89223.8 ** 2 - self._wavenumbers ** 2) +
                    5.3334 / (75037.5 ** 2 - self._wavenumbers ** 2) +
                    4.3244 / (67837.7 ** 2 - self._wavenumbers ** 2) +
                    0.00001218145 / (2418.136 ** 2 - self._wavenumbers ** 2))
        return n

    def __cross_section(self, number_density: float, king_factor: np.ndarray,
                        index_of_refraction: np.ndarray) -> np.ndarray:
        coefficient = 24 * np.pi**3 * self._wavenumbers**4 / number_density**2
        middle_term = ((index_of_refraction ** 2 - 1) /
                       (index_of_refraction ** 2 + 2)) ** 2
        return coefficient * middle_term * king_factor   # cm**2 / molecule

    @property
    def scattering_optical_depth(self) -> np.ndarray:
        """Get the Rayleigh scattering optical depth.

        Returns
        -------
        np.ndarray
            The scattering optical depth.

        Notes
        -----
        The shape of this array is determined by the inputs. In general, the
        shape will be [n_layers, (n_pixels)]. For example, if altitude_grid has
        shape [20] and wavenumbers has shape [50, 100, 45], this array will have
        shape [20, 50, 100, 45].

        """
        return self.__scattering_od
