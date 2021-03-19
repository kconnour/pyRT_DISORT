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
        pf = np.zeros((3, self.__n_layers) + self.__spectral_shape)
        pf[0, :] = 1
        pf[2, :] = 0.1
        return pf

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
            Raised if :code:`wavelength` or :code:`column_density` is not a
            numpy.ndarray.
        ValueError
            Raised if any values in :code:`wavelength` are not between 0.1 and
            50 microns (I assume this is the valid range to do retrievals); if


        Notes
        -----
        The values used here are from `Sneep and Ubachs 2005
        <https://doi.org/10.1016/j.jqsrt.2004.07.025>`_

        Due to a typo in the paper, I changed the coefficient to 10**3 when
        using equation 13 for computing the index of refraction

        """
        self.__wavelength = Wavelength(wavelength)
        self.__wavenumber = self.__wavelength.wavelength_to_wavenumber()
        self.__column_density = ColumnDensity(column_density)

        self.__raise_error_if_inputs_have_incompatible_shapes()

        super().__init__(column_density.shape[0], wavelength.shape)

        self.__scattering_od = \
            self.__calculate_scattering_optical_depths(column_density)

    def __raise_error_if_inputs_have_incompatible_shapes(self) -> None:
        if self.__wavelength.wavelength.shape[1:] != \
                self.__column_density.column_density.shape[1:]:
            message = 'wavelength and column_density must have the same ' \
                      'shape along all dimensions except the 0th.'
            raise ValueError(message)

    def __calculate_scattering_optical_depths(
            self, column_density: np.ndarray) -> np.ndarray:
        return np.multiply(column_density[:, None, :],
                                 self.__molecular_cross_section()[None, :])

    def __molecular_cross_section(self):
        number_density = 25.47 * 10 ** 18  # laboratory molecules / cm**3
        king_factor = 1.1364 + 25.3 * 10 ** -12 * self.__wavenumber ** 2
        index_of_refraction = self.__index_of_refraction()
        return self.__cross_section(
            number_density, king_factor, index_of_refraction) * 10 ** -4

    def __index_of_refraction(self) -> np.ndarray:
        n = 1 + 1.1427 * 10 ** 3 * (
                    5799.25 / (128908.9 ** 2 - self.__wavenumber ** 2) +
                    120.05 / (89223.8 ** 2 - self.__wavenumber ** 2) +
                    5.3334 / (75037.5 ** 2 - self.__wavenumber ** 2) +
                    4.3244 / (67837.7 ** 2 - self.__wavenumber ** 2) +
                    0.00001218145 / (2418.136 ** 2 - self.__wavenumber ** 2))
        return n

    def __cross_section(self, number_density: float, king_factor: np.ndarray,
                        index_of_refraction: np.ndarray) -> np.ndarray:
        coefficient = 24 * np.pi**3 * self.__wavenumber**4 / number_density**2
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


if __name__ == '__main__':
    wav = np.broadcast_to(np.array([1, 2, 3, 4, 40]), (15, 10, 5)).T
    cd = np.array([4.31127724e+20, 6.95971530e+20, 1.13068092e+21,
                       1.75965132e+21, 2.72941205e+21, 4.25424936e+21,
                       6.61806947e+21, 1.02334693e+22, 1.55374107e+22,
                       2.34710798e+22, 3.61121739e+22, 5.54481700e+22,
                       8.47633760e+22, 1.29169149e+23, 1.93867954e+23,
                       2.86145159e+23, 4.15075084e+23, 5.93623734e+23,
                       8.34122953e+23])
    colden = np.broadcast_to(cd, (15, 10, 19)).T
    r = RayleighCO2(wav, colden)
    od = r.scattering_optical_depth
    pf = r.phase_function
    print(pf.shape)
    print(od.shape)
    print(od[:, 4, 0, 0])
