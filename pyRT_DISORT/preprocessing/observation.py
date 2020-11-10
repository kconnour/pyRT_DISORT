import numpy as np
import scipy.interpolate.interpolate as interpolate


class Observation:
    def __init__(self, short_wavelength, long_wavelength, solar_zenith_angle, emission_angle, phase_angle):
        """ Initialize the class

        Parameters
        ----------
        short_wavelength: np.ndarray
            The short wavelengths [microns] for each spectral bin.
        long_wavelength: np.ndarray
            The long wavelengths [microns] for each spectral bin.
        solar_zenith_angle: np.ndarray
            Flattened array of the pixel solar zenith angles [degrees].
        emission_angle: np.ndarray
            Flattened array of the pixel emission angle [degrees].
        phase_angle: np.ndarray
            Flattened array of the pixel phase angle [degrees].

        Notes
        -----
        short_wavelength and long_wavelength should be the same length. Additionally, solar_zenith_angle,
        emission_angle, and phase_angle should be the same length.
        """
        self.short_wavelength = short_wavelength
        self.long_wavelength = long_wavelength
        self.sza = solar_zenith_angle
        self.emission = emission_angle
        self.phase = phase_angle
        self.phi0 = 0
        self.__check_inputs()

        # Ensure the object knows everything it ought to know
        self.low_wavenumber = self.__wavelength_to_wavenumber(self.long_wavelength)
        self.high_wavenumber = self.__wavelength_to_wavenumber(self.short_wavelength)
        self.mu = self.__compute_angle_cosine(self.emission)
        self.mu0 = self.__compute_angle_cosine(self.sza)
        self.phi = self.__compute_phi()

    def __check_inputs(self):
        assert isinstance(self.short_wavelength, np.ndarray), 'short_wavelengths must be a numpy array'
        assert isinstance(self.long_wavelength, np.ndarray), 'long_wavelengths must be a numpy array'
        assert self.short_wavelength.shape == self.long_wavelength.shape, 'short_wavelengths and long_wavelengths ' \
                                                                          'must have the same shape'
        assert isinstance(self.sza, np.ndarray), 'solar_zenith_angle must be a numpy array'
        assert isinstance(self.emission, np.ndarray), 'emission_angle must be a numpy array'
        assert isinstance(self.phase, np.ndarray), 'phase_angle must be a numpy array'
        assert self.sza.shape == self.emission.shape == self.phase.shape, 'solar_zenith_angle, emission_angle, and' \
                                                                          'phase_angle must have the same shape'

    @staticmethod
    def __wavelength_to_wavenumber(wavelength):
        cm_wavelength = wavelength * 10**-4
        return 1 / cm_wavelength

    @staticmethod
    def __compute_angle_cosine(angle):
        return np.cos(np.radians(angle))

    def __compute_phi(self):
        # I'm avoiding doing "Trap the case of the emission angle or solar zenith angle = 0" and
        # "Trap the round-off case for arc cosine" for efficiency... will deal with once I run into a problem
        sin_emission_angle = np.cos(np.radians(90 - self.emission))
        sin_solar_zenith_angle = np.cos(np.radians(90 - self.sza))
        d_phi = np.arccos((self.__compute_angle_cosine(self.phase) - self.mu * self.mu0) /
                          (sin_emission_angle * sin_solar_zenith_angle))
        return self.phi0 + 180 - np.degrees(d_phi)
