# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.eos import ModelEquationOfState


class RayleighCo2:
    """A RayleighCo2 object contains the Rayleigh scattering optical depths in each model layer due to CO2,
    along with the phase function"""
    def __init__(self, wavelengths, model_grid, n_moments):
        """
        Parameters
        ----------
        wavelengths: np.ndarray
            The wavelengths at which to calculate the scattering optical depths
        model_grid: ModelGrid
            The model grid
        n_moments: int
            The number of moments to construct a phase function for

        Attributes
        ----------
        wavelengths: np.ndarray
            The input wavelengths
        model_grid: ModelEquationOfState
            The input model_grid
        n_moments: int
            The input n_moments
        wavenumbers: np.ndarray
            wavelengths converted to wavenumbers
        scattering_optical_depths: np.ndarray
            The optical depths due to Rayleigh scattering in each layer in the model
        phase_function: np.ndarray
            3D phase function of Legendre coefficients in each layer and at each wavelength. This is the phase function
            multiplied by scattering_optical_depths
        """
        self.wavelengths = wavelengths
        self.model_grid = model_grid
        self.n_moments = n_moments

        self.wavenumbers = self.__convert_wavelength_to_wavenumber()
        self.scattering_optical_depths = self.__calculate_hyperspectral_rayleigh_co2_optical_depths()
        self.__phase_function = self.__make_phase_function()
        self.phase_function = self.__make_hyperspectral_layered_phase_function()

    def __convert_wavelength_to_wavenumber(self):
        return 1 / (self.wavelengths * 10 ** -4)

    def __calculate_hyperspectral_rayleigh_co2_optical_depths(self):
        return np.outer(self.model_grid.column_density_layers, self.__calculate_molecular_cross_section())

    def __calculate_molecular_cross_section(self):
        """ Calculate the molecular cross section (m**2 / molecule)
        Equation 2 in Sneep and Ubachs 2005, JQSRT, 92, 293-310
        """
        number_density = 25.47 * 10 ** 18  # molecules / cm**3  used in the paper measurements
        king_factor = 1.1364 + 25.3 * 10 ** -12 * self.wavenumbers ** 2
        index_of_refraction = self.__co2_index_of_refraction()
        return self.__scattering_cross_section(number_density, king_factor, index_of_refraction) * 10 ** -4

    def __co2_index_of_refraction(self):
        """ Calculate the index of refraction for CO2 using equation 13 and changing the coefficient to 10**3"""
        n = 1 + 1.1427 * 10 ** 3 * (
                    5799.25 / (128908.9 ** 2 - self.wavenumbers ** 2) + 120.05 / (89223.8 ** 2 - self.wavenumbers ** 2)
                    + 5.3334 / (75037.5 ** 2 - self.wavenumbers ** 2) + 4.3244 / (67837.7 ** 2 - self.wavenumbers ** 2)
                    + 0.1218145 * 10 ** -4 / (2418.136 ** 2 - self.wavenumbers ** 2))
        return n

    def __scattering_cross_section(self, number_density, king_factor, index_of_refraction):
        coefficient = 24 * np.pi ** 3 * self.wavenumbers ** 4 / number_density ** 2
        middle_term = ((index_of_refraction ** 2 - 1) / (index_of_refraction ** 2 + 2)) ** 2
        return coefficient * middle_term * king_factor   # cm**2 / molecule

    def __make_phase_function(self):
        # Note: if I have other Rayleigh things I can probably put this in a Rayleigh class and everything else
        # into the RayleighCo2 class which extends Rayleigh
        rayleigh_phase_function = np.zeros((self.n_moments, self.model_grid.n_layers, len(self.wavelengths)))
        rayleigh_phase_function[0, :, :] = 1
        rayleigh_phase_function[2, :, :] = 0.1
        return rayleigh_phase_function

    def __make_hyperspectral_layered_phase_function(self):
        return self.scattering_optical_depths * self.__phase_function
