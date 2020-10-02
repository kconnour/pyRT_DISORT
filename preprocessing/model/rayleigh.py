# 3rd-party imports
import numpy as np

# Local imports
#from preprocessing.model.atmosphere import Layers


class RayleighCo2:
    def __init__(self, wavelengths, layers, n_moments):
        self.wavelengths = wavelengths
        self.layers = layers
        self.n_moments = n_moments

        assert isinstance(self.wavelengths, np.ndarray), 'wavelengths must be a numpy array.'
        assert isinstance(self.layers, Layers), 'layers must be an instance of Layers.'
        assert isinstance(self.n_moments, int), 'n_moments must be an int.'

        self.wavenumbers = 1 / (self.wavelengths * 10 ** -4)   # 1/cm
        self.hyperspectral_optical_depths = self.calculate_hyperspectral_rayleigh_co2_optical_depths()
        self.phase_function = self.make_phase_function()
        self.hyperspectral_layered_phase_function = self.make_hyperspectral_layered_phase_function()

    def calculate_hyperspectral_rayleigh_co2_optical_depths(self):
        return np.outer(self.layers.column_density_layers, self.calculate_molecular_cross_section())

    def calculate_molecular_cross_section(self):
        """ Calculate the molecular cross section (m**2 / molecule)
        Equation 2 in Sneep and Ubachs 2005, JQSRT, 92, 293-310
        """

        number_density = 25.47 * 10 ** 18  # molecules / cm**3  used in the paper measurements
        king_factor = 1.1364 + 25.3 * 10 ** -12 * self.wavenumbers ** 2
        index_of_refraction = self.co2_index_of_refraction()
        return self.scattering_cross_section(number_density, king_factor, index_of_refraction) * 10 ** -4

    def co2_index_of_refraction(self):
        """ Calculate the index of refraction for CO2 using equation 13 and changing the coefficient to 10**3"""
        n = 1 + 1.1427 * 10 ** 3 * (
                    5799.25 / (128908.9 ** 2 - self.wavenumbers ** 2) + 120.05 / (89223.8 ** 2 - self.wavenumbers ** 2)
                    + 5.3334 / (75037.5 ** 2 - self.wavenumbers ** 2) + 4.3244 / (67837.7 ** 2 - self.wavenumbers ** 2)
                    + 0.1218145 * 10 ** -4 / (2418.136 ** 2 - self.wavenumbers ** 2))
        return n

    def scattering_cross_section(self, number_density, king_factor, index_of_refraction):
        coefficient = 24 * np.pi ** 3 * self.wavenumbers ** 4 / number_density ** 2
        middle_term = ((index_of_refraction ** 2 - 1) / (index_of_refraction ** 2 + 2)) ** 2
        return coefficient * middle_term * king_factor   # cm**2 / molecule

    def make_phase_function(self):
        rayleigh_phase_function = np.zeros((self.n_moments, self.layers.n_layers, len(self.wavelengths)))
        rayleigh_phase_function[0, :, :] = 1
        rayleigh_phase_function[2, :, :] = 0.1
        return rayleigh_phase_function

    def make_hyperspectral_layered_phase_function(self):
        return self.hyperspectral_optical_depths * self.phase_function
