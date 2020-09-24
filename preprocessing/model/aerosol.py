# 3rd-party imports
import numpy as np


class Aerosol:
    def __init__(self, aerosol_file, wavelengths, reference_wavelength):
        """ Initialize the class to hold all the aerosol's properties

        Parameters
        ----------
        aerosol_file: str
            The complete path of the file containing the aerosol's properties
        wavelengths: np.ndarray
            The wavelengths at which this aerosol was observed
        reference_wavelength: float
            The wavelength at which to scale the wavelengths
        """

        self.aerosol_file = aerosol_file
        self.wavelengths = wavelengths
        self.reference_wavelength = reference_wavelength

        assert isinstance(self.aerosol_file, str), 'aerosol_file nees to be a string.'
        assert isinstance(self.wavelengths, np.ndarray), 'wavelengths needs to be a numpy array.'
        assert isinstance(self.reference_wavelength, float), 'reference_wavelength needs to be a float.'

        # Make sure the aerosol knows its properties
        self.wavelengths_quantities, self.c_extinction, self.c_scattering, self.kappa, self.g, \
            self.p_max, self.theta_max = self.read_aerosol_file()
        self.inform_if_outside_wavelength_range()
        self.extinction_ratios = self.calculate_wavelength_extinction_ratios()
        self.single_scattering_albedos = self.calculate_single_scattering_albedos()
        self.asymmetry_parameters = self.calculate_asymmetry_parameters()

    def read_aerosol_file(self):
        aerosol_properties = np.load(self.aerosol_file, allow_pickle=True)
        wavelengths = aerosol_properties[:, 0]
        c_extinction = aerosol_properties[:, 1]
        c_scattering = aerosol_properties[:, 2]
        kappa = aerosol_properties[:, 3]
        g = aerosol_properties[:, 4]

        if aerosol_properties.shape[1] == 5:
            p_max = np.array([])
            theta_max = np.array([])
        else:
            p_max = aerosol_properties[:, 5]
            theta_max = aerosol_properties[:, 6]

        return wavelengths, c_extinction, c_scattering, kappa, g, p_max, theta_max

    def inform_if_outside_wavelength_range(self):
        if np.size((too_short := self.wavelengths[self.wavelengths < self.wavelengths_quantities[0]]) != 0):
            print('The following input wavelengths: {} microns are shorter than {:.1f} microns---the shortest '
                  'wavelength in the file. Using properties from that wavelength.'
                  .format(too_short, self.wavelengths_quantities[0]))
        if np.size((too_long := self.wavelengths[self.wavelengths > self.wavelengths_quantities[-1]]) != 0):
            print('The following input wavelengths: {} microns are longer than {:.1f} microns---the shortest '
                  'wavelength in the file. Using properties from that wavelength.'
                  .format(too_long, self.wavelengths_quantities[-1]))

    def calculate_wavelength_extinction_ratios(self):
        # Calculate the wavelength scaling between the input wavelengths and reference wavelength
        reference_c_ext = np.interp(self.reference_wavelength, self.wavelengths_quantities, self.c_extinction)
        wavelengths_c_ext = np.interp(self.wavelengths, self.wavelengths_quantities, self.c_extinction)
        return wavelengths_c_ext / reference_c_ext

    def calculate_single_scattering_albedos(self):
        # Calculate the single scattering albedo = C_scattering / C_extinction at the input wavelengths
        interpolated_extinction = np.interp(self.wavelengths, self.wavelengths_quantities, self.c_extinction)
        interpolated_scattering = np.interp(self.wavelengths, self.wavelengths_quantities, self.c_scattering)
        return interpolated_scattering / interpolated_extinction

    def calculate_asymmetry_parameters(self):
        return np.interp(self.wavelengths, self.wavelengths_quantities, self.g)
