# 3rd-party imports
import numpy as np


class ModelAtmosphere:
    """A ModelAtmosphere object holds the atmospheric constituents and can calculate the DTAUC, SSALB, and PMOM
    arrays for DISORT"""
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
            (total optical depth, scattering optical depth, legendre coefficients) for this constituent

        """
        self.__check_constituent_addition(properties)
        self.constituent_total_optical_depths.append(properties[0])
        self.constituent_scattering_optical_depths.append(properties[1])
        self.constituent_legendre_coefficients.append(properties[2])

    @staticmethod
    def __check_constituent_addition(properties):
        if not isinstance(properties, tuple):
            raise TypeError('properties must be a tuple')
        if len(properties) != 3:
            raise ValueError('properties must be of length 3')
        if not all(isinstance(x, np.ndarray) for x in properties):
            raise TypeError('All elements in properties must be a np.ndarray')

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
        max_moments = self.__get_max_moments()
        self.__match_moments(self.constituent_legendre_coefficients, max_moments)
        self.hyperspectral_legendre_moments = sum(self.constituent_legendre_coefficients) / \
                                              (self.hyperspectral_total_optical_depths *
                                               self.hyperspectral_total_single_scattering_albedos)

    def __get_max_moments(self):
        max_moments = 0
        for constituent in self.constituent_legendre_coefficients:
            if constituent.shape[0] > max_moments:
                max_moments = constituent.shape[0]
        return max_moments

    def __match_moments(self, phase_functions, max_moments):
        for counter, pf in enumerate(phase_functions):
            if pf.shape[0] < max_moments:
                self.constituent_legendre_coefficients[counter] = self.__add_moments(pf, max_moments)

    @staticmethod
    def __add_moments(phase_function, max_moments):
        starting_inds = np.linspace(phase_function.shape[0], phase_function.shape[0],
                                    num=max_moments - phase_function.shape[0], dtype=int)
        return np.insert(phase_function, starting_inds, 0, axis=0)
