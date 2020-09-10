import numpy as np
from generic.phase_function import EmpiricalPhaseFunction, HenyeyGreenstein


class Aerosol:
    def __init__(self, n_moments, phase_function_type, wavelength_reference, wavelengths, aerosol_file='', g=np.nan,
                 legendre_file=''):
        """ Initialize the class

        Parameters
        ----------
        n_moments: int
            The number of phase function moments
        phase_function_type: str
            A string of which phase function to use. Can be 'hg' for HG. It's a parameter to extend the class in the
            future
        wavelength_reference: float
            The reference wavelength to scale the aerosol to
        wavelengths: np.ndarray
            The wavelengths to at which this aerosol was observed
        aerosol_file: str, optional
            The Unix-like path to the aerosol info file. Default is ''
        g: float, optional
            The HG asymmetry parameter. Default is np.nan
        legendre_file: str, optional
            The Unix-like path to the Legendre coefficients file. Default is ''
        """
        self.n_moments = n_moments
        self.phase_function_type = phase_function_type
        self.wave_ref = wavelength_reference
        self.wavelengths = wavelengths
        self.aerosol_file = aerosol_file
        self.g = g
        self.legendre_file = legendre_file

        if self.aerosol_file:
            self.wavs, self.c_ext, self.c_sca, self.kappa, self.g, self.p_max, self.theta_max = self.read_aerosol_file()
            self.scaling = self.calculate_wavelength_scaling()
            self.scattering_coeff = self.calculate_aerosol_scattering_coefficients()

        if self.legendre_file:
            self.phase_function = self.make_empirical_phase()
        else:
            self.phase_function = self.make_hg_phase()

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
                  'Using g from that wavelength'.format(too_short, self.wavs[0]))
        if np.size((too_long := self.wavelengths[self.wavelengths > self.wavs[-1]]) != 0):
            print('{} nm is longer than {:.1f} microns---the longest wavelength in the file. '
                  'Using g from that wavelength'.format(too_long, self.wavs[-1]))

    def calculate_wavelength_scaling(self):
        """ Make the wavelength scaling between the reference wavelength and input wavelengths

        Returns
        -------
        scaling: np.ndarray
            The ratios between C_extinction at the wavelengths and the reference wavelength
        """
        reference_c_ext = np.interp(np.array([self.wave_ref]), self.wavs, self.c_ext)
        wavelengths_c_ext = np.interp(self.wavelengths, self.wavs, self.c_ext)
        scaling = wavelengths_c_ext / reference_c_ext
        return scaling

    def get_asymmetry_parameters(self):
        """ Interpolate the HG asymmetry parameter at a given wavelength

        Returns
        -------
        interpolated_g: np.ndarray
            The HG asymmetry parameter at the input wavelengths
        """
        self.check_wavelength()
        interpolated_g = np.interp(self.wavelengths, self.wavs, self.g)
        return interpolated_g

    def make_hg_phase(self):
        """ Make the possibly wavelength-dependent HG phase function

        Returns
        -------
        phase_function: np.ndarray
            The 2D array of the phase function moments
        """
        # If you want a HG phase function and know the g values at all the wavelengths
        if self.phase_function_type == 'hg' and self.aerosol_file:
            asymmetry_parameters = self.get_asymmetry_parameters()
            phase_function = np.zeros((self.n_moments, len(self.wavelengths)))
            for wavelength in range(len(self.wavelengths)):
                hg = HenyeyGreenstein(asymmetry_parameters[wavelength], self.n_moments)
                phase_function[:, wavelength] = hg.moments

        # If you want a HG phase function and don't know g, use user input
        elif self.phase_function_type == 'hg':
            phase_function = np.zeros((self.n_moments, len(self.wavelengths)))
            for wavelength in range(len(self.wavelengths)):
                hg = HenyeyGreenstein(self.g, self.n_moments)
                phase_function[:, wavelength] = hg.moments

        return phase_function

    def make_empirical_phase(self):
        """ Make a wavelength-independent phase function (for now...)

        Returns
        -------
        phase_function: np.ndarray
            The 2D array of the phase function moments
        """
        if self.legendre_file:
            legendre_coefficients = EmpiricalPhaseFunction(self.legendre_file, self.n_moments).moments
            phase_function = np.zeros((self.n_moments, len(self.wavelengths)))
            for wavelength in range(len(self.wavelengths)):
                phase_function[:, wavelength] = legendre_coefficients
        return phase_function

    def calculate_aerosol_scattering_coefficients(self):
        """ Calculate the scattering coefficient, C_scattering / C_extinction at the wavelengths

        Returns
        -------
        scattering_coefficients: np.ndarray
            The scattering coefficients at the input wavelengths
        """
        interpolated_extinction = np.interp(self.wavelengths, self.wavs, self.c_ext)
        interpolated_scattering = np.interp(self.wavelengths, self.wavs, self.c_sca)
        scattering_coefficients = interpolated_scattering / interpolated_extinction
        return scattering_coefficients
