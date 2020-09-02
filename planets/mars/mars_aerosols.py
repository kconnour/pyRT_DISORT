# 3rd-party imports
import numpy as np

# Local imports
from aerosol import Aerosol


class MarsDust(Aerosol):
    def __init__(self, dust_file, wavelength_reference=9.3*10**3):
        self.dust_file = dust_file
        self.wave_ref = wavelength_reference
        self.wavs, self.c_ext, self.c_sca, self.kappa, self.g = self.read_dust_file()

    def read_dust_file(self):
        """ Read the dust aerosol file

        Returns
        -------
        wavelengths: np.ndarray
            The wavelengths from the aerosol file
        c_extinction: np.ndarray
            The C_extinction coefficients from the aerosol file
        c_scattering: np.ndarray
            The C_scattering coefficients from the aerosol file
        kappa: np.ndarray
            The kappa coefficients from the aerosol file
        g: np.ndarray
            The HG g coefficients from the aerosol file
        """
        dust_properties = np.load(self.dust_file, allow_pickle=True)
        wavelengths = dust_properties[:, 0]
        c_extinction = dust_properties[:, 1]
        c_scattering = dust_properties[:, 2]
        kappa = dust_properties[:, 3]
        g = dust_properties[:, 4]
        return wavelengths, c_extinction, c_scattering, kappa, g

    def get_dust_asymmetry_parameter(self, wavelength):
        """ Interpolate the HG asymmetry parameter at a given wavelength

        Parameters
        ----------
        wavelength: float
            The wavelength to get the parameter at

        Returns
        -------
        interpolated_g: float
            The HG asymmetry parameter at the input wavelength
        """
        self.check_wavelength(wavelength)
        interpolated_g = np.interp(wavelength, self.wavs, self.g)
        return interpolated_g

    def check_wavelength(self, wavelength):
        """Inform the user if the wavelength they are using is within the range in dust_file

        Parameters
        ----------
        wavelength: float
            The wavelength

        Returns
        -------
        None
        """
        if wavelength < self.wavs[0]:
            print('{} nm is shorter than {:.0f} nm---the shortest wavelength in the file. '
                  'Using g from that wavelength'.format(wavelength, self.wavs[0]))
        if wavelength > self.wavs[-1]:
            print('{} nm is longer than {:.0f} nm---the longest wavelength in the file. '
                  'Using g from that wavelength'.format(wavelength, self.wavs[-1]))

    def calculate_wavelength_scaling(self, wavelength):
        """ Make the wavelength scaling between a wavelength and the reference wavelength

        Parameters
        ----------
        wavelength: float
            The wavelength

        Returns
        -------
        scaling: float
            The ratio between C_extinction at the wavelength and the reference wavelength
        """
        coefficients = np.interp(np.array([wavelength, self.wave_ref]), self.wavs, self.c_ext)
        scaling = coefficients[0] / coefficients[1]
        return scaling
