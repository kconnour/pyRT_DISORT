# 3rd-party imports
import numpy as np


class Aerosol:
    """ Create a class to hold all of the information about an aerosol"""

    def __init__(self, aerosol_file, wavelengths, reference_wavelength):
        """ Initialize the class to hold all the aerosol's properties

        Parameters
        ----------
        aerosol_file: np.ndarray
            An array containing the aerosol's properties
        wavelengths: np.ndarray
            The wavelengths at which this aerosol was observed
        reference_wavelength: float
            The wavelength at which to scale the wavelengths
        """

        self.aerosol_file = aerosol_file
        self.wavelengths = wavelengths
        self.reference_wavelength = reference_wavelength

        assert isinstance(self.aerosol_file, np.ndarray), 'aerosol_file must be a numpy array.'
        assert isinstance(self.wavelengths, np.ndarray), 'wavelengths needs to be a numpy array.'
        assert isinstance(self.reference_wavelength, float), 'reference_wavelength needs to be a float.'

        # Make sure the aerosol knows its properties
        self.wavelengths_quantities, self.c_extinction, self.c_scattering, self.kappa, self.g, \
            self.p_max, self.theta_max = self.__read_aerosol_file()
        self.__inform_if_outside_wavelength_range()
        self.extinction_ratios = self.__calculate_wavelength_extinction_ratios()
        self.hyperspectral_single_scattering_albedos = self.__calculate_hyperspectral_single_scattering_albedos()
        self.hyperspectral_asymmetry_parameters = self.__calculate_hyperspectral_asymmetry_parameters()

    def __read_aerosol_file(self):
        wavelengths = self.aerosol_file[:, 0]
        c_extinction = self.aerosol_file[:, 1]
        c_scattering = self.aerosol_file[:, 2]
        kappa = self.aerosol_file[:, 3]
        g = self.aerosol_file[:, 4]

        if self.aerosol_file.shape[1] == 5:
            p_max = np.array([])
            theta_max = np.array([])
        else:
            p_max = self.aerosol_file[:, 5]
            theta_max = self.aerosol_file[:, 6]

        return wavelengths, c_extinction, c_scattering, kappa, g, p_max, theta_max

    def __inform_if_outside_wavelength_range(self):
        if np.size((too_short := self.wavelengths[self.wavelengths < self.wavelengths_quantities[0]]) != 0):
            print('The following input wavelengths: {} microns are shorter than {:.3f} microns---the shortest '
                  'wavelength in the file. Using properties from that wavelength.'
                  .format(too_short, self.wavelengths_quantities[0]))
        if np.size((too_long := self.wavelengths[self.wavelengths > self.wavelengths_quantities[-1]]) != 0):
            print('The following input wavelengths: {} microns are longer than {:.1f} microns---the shortest '
                  'wavelength in the file. Using properties from that wavelength.'
                  .format(too_long, self.wavelengths_quantities[-1]))

    def __calculate_wavelength_extinction_ratios(self):
        reference_c_ext = np.interp(self.reference_wavelength, self.wavelengths_quantities, self.c_extinction)
        wavelengths_c_ext = np.interp(self.wavelengths, self.wavelengths_quantities, self.c_extinction)
        return wavelengths_c_ext / reference_c_ext

    def __calculate_hyperspectral_single_scattering_albedos(self):
        interpolated_extinction = np.interp(self.wavelengths, self.wavelengths_quantities, self.c_extinction)
        interpolated_scattering = np.interp(self.wavelengths, self.wavelengths_quantities, self.c_scattering)
        return interpolated_scattering / interpolated_extinction

    def __calculate_hyperspectral_asymmetry_parameters(self):
        return np.interp(self.wavelengths, self.wavelengths_quantities, self.g)
