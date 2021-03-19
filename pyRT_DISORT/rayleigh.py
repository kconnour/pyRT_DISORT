"""The rayleigh module contains structures for computing Rayleigh scattering.
"""
import numpy as np
from pyRT_DISORT.eos import ColumnDensity
from pyRT_DISORT.observation import Wavelength


class Rayleigh:
    """An abstract base class for Rayleigh scattering.

    Rayleigh creates the Legendre coefficient phase function array and holds
    the wavenumbers at which scattering was observed. This is an abstract base
    class from which all other Rayleigh classes are derived.

    """

    def __init__(self, n_layers: int, spectral_shape: tuple) -> None:
        """
        Parameters
        ----------
        n_layers
            The number of layers to use in the model.
        spectral_shape

        Raises
        ------
        TypeError
            Raised if :code:`n_layers` is not an int, or if
            :code:`spectral_shape` is not a tuple.
        ValueError
            Raised if the values in :code:`spectral_shape` are not ints.

        """
        self.__n_layers = n_layers
        self.__spectral_shape = spectral_shape

        self.__raise_error_if_inputs_are_bad()

        self.__phase_function = self.__construct_phase_function()

    def __raise_error_if_inputs_are_bad(self) -> None:
        self.__raise_type_error_if_n_layers_is_not_int()
        self.__raise_type_error_if_spectral_shape_is_not_tuple()
        self.__raise_value_error_if_spectral_shape_contains_non_ints()

    def __raise_type_error_if_n_layers_is_not_int(self) -> None:
        if not isinstance(self.__n_layers, int):
            message = 'n_layers must be an int.'
            raise TypeError(message)

    def __raise_type_error_if_spectral_shape_is_not_tuple(self) -> None:
        if not isinstance(self.__spectral_shape, tuple):
            message = 'spectral_shape must be a tuple.'
            raise TypeError(message)

    def __raise_value_error_if_spectral_shape_contains_non_ints(self) -> None:
        for val in self.__spectral_shape:
            if not isinstance(val, int):
                message = 'At least one value in spectral_shape is not an int.'
                raise ValueError(message)

    def __construct_phase_function(self) -> np.ndarray:
        pf = np.zeros((3, self.__n_layers, self.__spectral_shape))
        pf[0, :] = 1
        pf[2, :] = 0.1
        return pf.reshape(pf.shape[:-1] + self.__spectral_shape)

    @property
    def phase_function(self) -> np.ndarray:
        """Get the Legendre decomposition of the phase function.

        Notes
        -----
        The shape of this array is (3, n_layers, (spectral_shape)). The 0th and
        2nd coefficient along the 0th axis will be 1 and 0.1, respectively.

        """
        return self.__phase_function


class RayleighCO2(Rayleigh):
    """A structure to hold arrays related to CO2 Rayleigh scattering.

    RayleighCO2 creates the Legendre coefficient phase function array and the
    optical depths due to Rayleigh scattering by CO2 in each of the layers.

    """

    def __init__(self, wavelength: np.ndarray,
                 column_density: np.ndarray) -> None:
        """
        Parameters
        ----------
        wavelength

        column_density
            1D array of column densities (particles / m**2) for each layer of
            the model. This should be the same length as altitude_grid to be
            useful.

        Raises
        ------
        TypeError
            Raised if :code:`wavelength` is not an instance of numpy.ndarray.
        ValueError
            Raised if any values in :code:`wavelength` are not between 0.1 and
            50 microns (I assume this is the valid range to do retrievals).

        Notes
        -----
        The values used here are from `Sneep and Ubachs 2005
        <https://doi.org/10.1016/j.jqsrt.2004.07.025>`_

        Due to a typo in the paper, I changed the coefficient to 10**3 when
        using equation 13 for computing the index of refraction

        """
        self.__wavelength = Wavelength(wavelength)
        self.__column_density = ColumnDensity(column_density)

    def __raise_error





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
