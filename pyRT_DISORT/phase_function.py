"""The phase function module contains structures to create a variety of phase
function arrays required by DISORT.
"""
import numpy as np


class TabularLegendreCoefficients:
    """An abstract class from which all coefficient classes are derived.

    TabularLegendreCoefficients holds a variety of properties that are needed by
    all derived classes. It is not meant to be instantiated.

    """

    def __init__(self, coefficients: np.ndarray, altitude_grid: np.ndarray,
                 wavelengths: np.ndarray, max_moments: int = None) -> None:
        self._coefficients = coefficients
        self._altitude_grid = altitude_grid
        self._wavelengths = wavelengths
        self._max_moments = max_moments
        self._n_layers = len(self._altitude_grid)
        self._n_moments = coefficients.shape[0] if max_moments is None \
            else max_moments

    def _normalize_coefficients(self, unnormalized_coefficients):
        moment_indices = np.linspace(0, self._n_moments-1, num=self._n_moments)
        normalization = moment_indices * 2 + 1
        return (unnormalized_coefficients.T / normalization).T


class StaticTabularLegendreCoefficients(TabularLegendreCoefficients):
    """Create a grid of static Legendre coefficients.

    StaticTabularLegendreCoefficients accepts a 1D array of Legendre polynomial
    coefficients of the phase function that do not depend on particle size or
    wavelength and casts them to an array of the proper shape for use in DISORT.

    """
    def __init__(self, coefficients: np.ndarray, altitude_grid: np.ndarray,
                 wavelengths: np.ndarray, max_moments: int = None) -> None:
        """
        Parameters
        ----------
        coefficients
            1D array of Legendre coefficients that do not depend on particle
            size or wavelength.
        altitude_grid
            1D array of altitudes where to cast coefficients to.
        wavelengths
            ND array of wavelengths where to cast coefficients to. As this class
            does not depend on wavelength, this is essentially the pixels where
            to define the phase function coefficients.
        max_moments
            The maximum number of coefficients to use in the array. If None,
            all available coefficients are used.

        """
        super().__init__(coefficients, altitude_grid, wavelengths, max_moments)
        self.__phase = self.__make_static_phase_function()

    def __make_static_phase_function(self) -> np.ndarray:
        coefficient_grid = self.__create_coefficient_grid()
        return self._normalize_coefficients(coefficient_grid)

    def __create_coefficient_grid(self) -> np.ndarray:
        coeff_grid = np.broadcast_to(self._coefficients[:self._n_moments],
            (self._wavelengths.size, self._n_layers, self._n_moments)).T
        return coeff_grid.reshape(coeff_grid.shape[:-1] +
                                  self._wavelengths.shape)

    @property
    def phase_function(self):
        """Get the Legendre coefficients cast to the proper grid.

        Returns
        -------
        np.ndarray
            Coefficients on a grid.

        Notes
        -----
        In DISORT, this variable is named "PMOM". In addition,
        The shape of this array is determined by the inputs. In general, the
        shape will be [n_moments, n_layers, (n_pixels)]. For example, if
        coefficients has shape [65], altitude_grid has shape [20], and
        wavelengths has shape [50, 100, 45], this array will have shape
        [65, 20, 50, 100, 45].

        Note that DISORT expects an array of shape [n_moments, n_layers] as the
        PMOM input, so be sure to select the correct pixel when iterating over
        pixels.

        """
        return self.__phase


# TODO: make this class later
'''class RadialTabularLegendreCoefficients:
    def __init__(self, altitude_grid, particle_size_profile, wavelengths,
                 n_moments, particle_size_grid):
        super().__init__(altitude_grid, particle_size_profile, wavelengths,
                         n_moments)
        self.__particle_size_grid = particle_size_grid'''
        
# TODO: make this class later
'''class SpectralTabularLegendreCoefficients:
    def __init__(self, altitude_grid, particle_size_profile, wavelengths,
                 n_moments, wavelength_grid):
        super().__init__(altitude_grid, particle_size_profile, wavelengths,
                         n_moments)
        self.__wavelength_grid = wavelength_grid'''


# TODO: It'd be nice to have a variant that's not nearest neighbor
# TODO: I almost duplicate the docstring of phase function. It'd be nice to put
#  it in the base class
class RadialSpectralTabularLegendreCoefficients(TabularLegendreCoefficients):
    """Create a grid of Legendre coefficients (particle size, wavelength).

    StaticTabularLegendreCoefficients accepts a 3D array of Legendre polynomial
    coefficients of the phase function that depend on particle size and
    wavelength and casts them to an array of the proper shape for use in DISORT
    given a vertical particle size gradient.

    """

    def __init__(self, coefficients: np.ndarray, particle_size_grid: np.ndarray,
                 wavelength_grid: np.ndarray, altitude_grid: np.ndarray,
                 wavelengths: np.ndarray, particle_size_profile: np.ndarray,
                 max_moments: int = None) -> None:
        """
        Parameters
        ----------
        coefficients
            3D array of Legendre coefficients that depend on particle
            size and wavelength. It is assumed to have shape [n_moments,
            particle_size_grid, wavelength_grid]
        particle_size_grid
            1D array of particle sizes associated with the coefficients matrix.
        wavelength_grid
            1D array of wavelengths associated with the coefficients matrix.
        altitude_grid
            1D array of altitudes where to cast coefficients to.
        wavelengths
            ND array of wavelengths where to cast coefficients to.
        particle_size_profile
            1D array of particle sizes.
        max_moments
            The maximum number of coefficients to use in the array. If None,
            all available coefficients are used.

        """
        super().__init__(coefficients, altitude_grid, wavelengths, max_moments)
        self.__particle_size_grid = particle_size_grid
        self.__wavelength_grid = wavelength_grid
        self.__particle_size_profile = particle_size_profile

        self.__phase_function = self.__make_phase_function()

    def __make_phase_function(self) -> np.ndarray:
        regridded_phase_function = self.__get_nearest_neighbor_phase_function()
        return self._normalize_coefficients(regridded_phase_function)

    def __get_nearest_neighbor_phase_function(self) -> np.ndarray:
        size_indices = self.__get_nearest_indices(self.__particle_size_grid,
                                                  self.__particle_size_profile)
        wavelength_indices = self.__get_nearest_indices(self.__wavelength_grid,
                                                        self._wavelengths)
        return self._coefficients[:, size_indices, :][:, :, wavelength_indices]

    @staticmethod
    def __get_nearest_indices(grid: np.ndarray, values: np.ndarray) \
            -> np.ndarray:
        # grid should be 1D; values can be ND
        return np.abs(np.subtract.outer(grid, values)).argmin(0)

    @property
    def phase_function(self) -> np.ndarray:
        """Get the Legendre coefficients cast to the proper grid.

        Returns
        -------
        np.ndarray
            The Legendre coefficients on a grid.

        Notes
        -----
        In DISORT, this variable is named "PMOM". In addition,
        The shape of this array is determined by the inputs. In general, the
        shape will be [n_moments, n_layers, (n_pixels)]. For example, if
        coefficients has shape [65, 10, 300], altitude_grid has shape [20], and
        wavelengths has shape [50, 100, 45], this array will have shape
        [65, 20, 50, 100, 45].

        Note that DISORT expects an array of shape [n_moments, n_layers] as the
        PMOM input, so be sure to select the correct pixel when iterating over
        pixels.

        """
        return self.__phase_function
