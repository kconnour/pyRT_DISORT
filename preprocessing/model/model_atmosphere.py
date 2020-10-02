# 3rd-party imports
import numpy as np


class ModelAtmosphere:
    """Make a class to hold all the atmospheric constituents and calculate the dtauc, ssalb, and pmom arrays"""
    def __init__(self):
        self.constituent_total_optical_depths = []
        self.constituent_scattering_optical_depths = []
        self.constituent_legendre_coefficients = []
        self.hyperspectral_total_optical_depths = np.nan
        self.hyperspectral_total_single_scattering_albedos = np.nan
        self.hyperspectral_legendre_moments = np.nan

    def add_constituent(self, properties):
        """ Add an atmospheric constituent to the model

        Parameters
        ----------
        properties: tuple
            (total optical depth, scattering optical depth, legendre coefficients) for this constitudent

        Returns
        -------
        None
        """
        assert isinstance(properties, tuple), 'properties needs to be a tuple.'
        assert len(properties) == 3, 'properties needs to be of length 3.'
        self.constituent_total_optical_depths.append(properties[0])
        self.constituent_scattering_optical_depths.append(properties[1])
        self.constituent_legendre_coefficients.append(properties[2])

    def compute_model(self):
        """ Compute the properties of this model. Run this method after everything is added to the model.

        Returns
        -------
        None
        """
        self.__calculate_hyperspectral_total_optical_depths()
        self.__calculate_hyperspectral_total_single_scattering_albedos()
        self.__calculate_hyperspectral_legendre_coefficients()

    def __calculate_hyperspectral_total_optical_depths(self):
        self.hyperspectral_total_optical_depths = sum(self.constituent_total_optical_depths)

    def __calculate_hyperspectral_total_single_scattering_albedos(self):
        self.hyperspectral_total_single_scattering_albedos = sum(self.constituent_scattering_optical_depths) / \
                                                             self.hyperspectral_total_optical_depths

    def __calculate_hyperspectral_legendre_coefficients(self):
        self.hyperspectral_legendre_moments = sum(self.constituent_legendre_coefficients) / \
                                              (self.hyperspectral_total_optical_depths *
                                               self.hyperspectral_total_single_scattering_albedos)
