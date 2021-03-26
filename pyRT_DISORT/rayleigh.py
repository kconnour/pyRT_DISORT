"""The rayleigh module contains structures for computing Rayleigh scattering.
"""
import numpy as np
from pyRT_DISORT.eos import _EoSVar
from pyRT_DISORT.observation import _Wavelength


class Rayleigh:
    """An abstract base class for Rayleigh scattering.

    Rayleigh creates the single scattering albedo and Legendre coefficient
    phase function array given the number of layers and the spectral shape. This
    is an abstract base class from which all other Rayleigh classes are derived.

    """

    def __init__(self, n_layers: int, spectral_shape: tuple) -> None:
        """
        Parameters
        ----------
        n_layers
            The number of layers to use in the model.
        spectral_shape
            The pixel shape to construct a phase function.

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

        self.__single_scattering_albedo = self.__make_single_scattering_albedo()
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

    def __make_single_scattering_albedo(self) -> np.ndarray:
        return np.ones((self.__n_layers,) + self.__spectral_shape)

    def __construct_phase_function(self) -> np.ndarray:
        pf = np.zeros((3, self.__n_layers) + self.__spectral_shape)
        pf[0, :] = 1
        pf[2, :] = 0.1
        return pf

    @property
    def single_scattering_albedo(self) -> np.ndarray:
        r"""Get the Rayleigh single scattering albedo.

        Notes
        -----
        The shape of this array is (n_layers, (spectral_shape)). It will be
        filled with all 1s.

        """
        return self.__single_scattering_albedo

    @property
    def phase_function(self) -> np.ndarray:
        r"""Get the Legendre decomposition of the phase function.

        Notes
        -----
        The shape of this array is (3, n_layers, (spectral_shape)). The
        0 :sup:`th` and 2 :sup:`nd` coefficient along the 0 :sup:`th` axis will
        be 1 and 0.1, respectively.

        """
        return self.__phase_function


class RayleighCO2(Rayleigh):
    r"""A structure to compute CO :sub:`2` Rayleigh scattering arrays.

    RayleighCO2 creates the optical depth, single scattering albedo, and
    Legendre coefficient decomposition phase function arrays due to Rayleigh
    scattering by CO :sub:`2` in each of the layers.

    """

    def __init__(self, wavelength: np.ndarray,
                 column_density: np.ndarray) -> None:
        """
        Parameters
        ----------
        wavelength
            Wavelength at which Rayleigh scattering will be computed.
        column_density
            Column density in the model layers.

        Raises
        ------
        TypeError
            Raised if :code:`wavelength` or :code:`column_density` is not a
            numpy.ndarray.
        ValueError
            Raised if any values in :code:`wavelength` or :code:`column_density`
            are unphysical, or if they have incompatible shapes. See the note
            below for more details.

        Notes
        -----
        In the general case of a hyperspectral imager with MxN pixels and W
        wavelengths, :code:`wavelength` can have shape WxMxN. In this case,
        :code:`column_density` should have shape ZxMxN, where Z is the number
        of model layers. The 0 :sup:`th` dimension can have different shapes
        between the arrays but the subsequent dimensions (if any) should have
        the same shape.

        The values used here are from `Sneep and Ubachs 2005
        <https://doi.org/10.1016/j.jqsrt.2004.07.025>`_

        Due to a typo in the paper, I changed the coefficient to 10 :sup:`3`
        when using equation 13 for computing the index of refraction

        """
        self.__wavelength = _Wavelength(wavelength, 'wavelength')
        self.__wavenumber = self.__wavelength.wavelength_to_wavenumber()
        self.__column_density = _EoSVar(column_density, 'cd')

        self.__raise_error_if_inputs_have_incompatible_shapes()

        super().__init__(column_density.shape[0], wavelength.shape)

        self.__scattering_od = \
            self.__calculate_scattering_optical_depths(column_density)

    def __raise_error_if_inputs_have_incompatible_shapes(self) -> None:
        if self.__wavelength.shape[1:] != \
                self.__column_density.val.shape[1:]:
            message = 'wavelength and column_density must have the same ' \
                      'shape along all dimensions except the 0th.'
            raise ValueError(message)

    def __calculate_scattering_optical_depths(
            self, column_density: np.ndarray) -> np.ndarray:
        column_density = column_density[:, None]
        mcs = self.__molecular_cross_section()[:, None]
        scattering_od = np.multiply(column_density[:, None, :], mcs[None, :])
        return np.squeeze(scattering_od)

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
    def optical_depth(self) -> np.ndarray:
        """Get the Rayleigh optical depth.

        """
        return self.__scattering_od
