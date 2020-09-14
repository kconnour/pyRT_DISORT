# 3rd-party imports
import numpy as np


class Aerosol:
    def __init__(self, aerosol_file, phase_function, wavelengths, wavelength_reference):
        """ Initialize the class

        Parameters
        ----------
        aerosol_file: str
            The Unix-like path to the file containing the aerosol's properties
        phase_function: PhaseFunction (duck typed...)
            The phase function object
        wavelengths: np.ndarray
            The wavelengths at which this aerosol was observed
        wavelength_reference: float
            The wavelength at which to scale the wavelengths
        """

        self.aerosol_file = aerosol_file
        self.phase_function = phase_function
        self.wavelengths = wavelengths
        self.wave_ref = wavelength_reference

        # Make sure the aerosol knows its properties
        self.wavs, self.c_ext, self.c_sca, self.kappa, self.g, self.p_max, self.theta_max = self.read_aerosol_file()
        self.check_wavelength()
        self.extinction_ratio = self.calculate_wavelength_extinction_ratio()
        self.single_scattering_albedo = self.calculate_single_scattering_albedo()

    def read_aerosol_file(self):
        """ Read the aerosol file

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
        p_max: np.ndarray
            The p_max from the aerosol file (if included)
        theta_max: np.ndarray
            The thetmax from the aerosol file (if included)
        """
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

    def check_wavelength(self):
        """ Inform the user if the wavelengths they are using are not within the range in aerosol_file

        Returns
        -------
        None
        """
        if np.size((too_short := self.wavelengths[self.wavelengths < self.wavs[0]]) != 0):
            print('{} nm is shorter than {:.1f} microns---the shortest wavelength in the file. '
                  'Using properties from that wavelength'.format(too_short, self.wavs[0]))
        if np.size((too_long := self.wavelengths[self.wavelengths > self.wavs[-1]]) != 0):
            print('{} nm is longer than {:.1f} microns---the longest wavelength in the file. '
                  'Using properties from that wavelength'.format(too_long, self.wavs[-1]))

    def calculate_wavelength_extinction_ratio(self):
        """ Calculate the wavelength scaling between the input wavelengths and reference wavelength

        Returns
        -------
        extinction_ratio: np.ndarray (len(wavelengths))
            The ratios between C_extinction at wavelengths and the reference wavelength
        """
        reference_c_ext = np.interp(np.array([self.wave_ref]), self.wavs, self.c_ext)
        wavelengths_c_ext = np.interp(self.wavelengths, self.wavs, self.c_ext)
        extinction_ratio = wavelengths_c_ext / reference_c_ext
        return extinction_ratio

    def calculate_single_scattering_albedo(self):
        """ Calculate the single scattering albedo = C_scattering / C_extinction at the input wavelengths

        Returns
        -------
        single_scattering_albedo: np.ndarray (len(wavelengths))
            The single_scattering_albedo at the input wavelengths
        """
        interpolated_extinction = np.interp(self.wavelengths, self.wavs, self.c_ext)
        interpolated_scattering = np.interp(self.wavelengths, self.wavs, self.c_sca)
        single_scattering_albedo = interpolated_scattering / interpolated_extinction
        return single_scattering_albedo

    def calculate_asymmetry_parameter(self):
        """ Calculate the HG asymmetry parameter at the input wavelengths

        Returns
        -------
        interpolated_g: np.ndarray (len(wavelengths))
            The interpolated asymmetry parameter
        """
        interpolated_g = np.interp(self.wavelengths, self.wavs, self.g)
        return interpolated_g
