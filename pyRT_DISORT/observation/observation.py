# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.utilities.array_checks import ArrayChecker


class Observation:
    """An Observation object holds the relevant observational information for doing RT."""
    def __init__(self, short_wavelength, long_wavelength, solar_zenith_angle, emission_angle, phase_angle):
        """
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

        Attributes
        ----------
        short_wavelength: np.ndarray
            The input short wavelengths
        long_wavelength: np.ndarray
            The input long wavelengths
        low_wavenumber: np.ndarray
            The wavenumbers of long_wavelength [1 /cm]
        high_wavenumber: np.ndarray
            The wavenumbers of short_wavelength [1 /cm]
        solar_zenith_angle: np.ndarray
            The input solar zenith angles
        emission_angle: np.ndarray
            The input emission angles
        phase_angle: np.ndarray
            The input phase angles
        mu0: np.ndarray
            The cosine of the input solar zenith angles
        mu: np.ndarray
            The cosine of the input emission angles
        phi0: float
            0 (I'm assuming phi0 is always 0)
        phi: np.ndarray
            The computed phi angles

        Notes
        -----
        short_wavelength and long_wavelength should be the same length. Additionally, solar_zenith_angle,
        emission_angle, and phase_angle should be the same length.
        """
        self.short_wavelength = short_wavelength
        self.long_wavelength = long_wavelength
        self.solar_zenith_angle = solar_zenith_angle
        self.emission_angle = emission_angle
        self.phase_angle = phase_angle

        self.__check_inputs_are_physical()

        self.low_wavenumber = self.__convert_wavelength_to_wavenumber(self.long_wavelength)
        self.high_wavenumber = self.__convert_wavelength_to_wavenumber(self.short_wavelength)
        self.mu0 = self.__compute_angle_cosine(self.solar_zenith_angle)
        self.mu = self.__compute_angle_cosine(self.emission_angle)
        self.phi0 = 0
        self.phi = self.__compute_phi()

    def __check_inputs_are_physical(self):
        self.__check_wavelengths_are_physical()
        self.__check_angles_are_physical()

    def __check_wavelengths_are_physical(self):
        self.__check_short_wavelengths_are_physical()
        self.__check_long_wavelengths_are_physical()
        self.__check_wavelengths_are_same_shape()
        self.__check_long_wavelength_is_larger()

    def __check_short_wavelengths_are_physical(self):
        short_wav_checker = ArrayChecker(self.short_wavelength, 'short_wavelength')
        short_wav_checker.check_object_is_array()
        short_wav_checker.check_ndarray_is_numeric()
        short_wav_checker.check_ndarray_is_positive_finite()
        short_wav_checker.check_ndarray_is_1d()

    def __check_long_wavelengths_are_physical(self):
        long_wav_checker = ArrayChecker(self.long_wavelength, 'long_wavelength')
        long_wav_checker.check_object_is_array()
        long_wav_checker.check_ndarray_is_numeric()
        long_wav_checker.check_ndarray_is_positive_finite()
        long_wav_checker.check_ndarray_is_1d()

    def __check_wavelengths_are_same_shape(self):
        if self.short_wavelength.shape != self.long_wavelength.shape:
            raise ValueError('short_wavelength and long_wavelength must have the same shape')

    def __check_long_wavelength_is_larger(self):
        if not np.all(self.long_wavelength > self.short_wavelength):
            raise ValueError('long_wavelength must always be larger than the corresponding short_wavelength')

    def __check_angles_are_physical(self):
        self.__check_solar_zenith_angles_are_physical()
        self.__check_emission_angles_are_physical()
        self.__check_phase_angles_are_physical()
        self.__check_angles_are_same_shape()

    def __check_solar_zenith_angles_are_physical(self):
        solar_zenith_angle_checker = ArrayChecker(self.solar_zenith_angle, 'solar_zenith_angle')
        solar_zenith_angle_checker.check_object_is_array()
        solar_zenith_angle_checker.check_ndarray_is_numeric()
        solar_zenith_angle_checker.check_ndarray_is_in_range(0, 180)
        solar_zenith_angle_checker.check_ndarray_is_1d()

    def __check_emission_angles_are_physical(self):
        emission_angle_checker = ArrayChecker(self.emission_angle, 'emission_angle')
        emission_angle_checker.check_object_is_array()
        emission_angle_checker.check_ndarray_is_numeric()
        emission_angle_checker.check_ndarray_is_in_range(0, 90)
        emission_angle_checker.check_ndarray_is_1d()

    def __check_phase_angles_are_physical(self):
        phase_angle_checker = ArrayChecker(self.phase_angle, 'phase_angle')
        phase_angle_checker.check_object_is_array()
        phase_angle_checker.check_ndarray_is_numeric()
        phase_angle_checker.check_ndarray_is_in_range(0, 180)
        phase_angle_checker.check_ndarray_is_1d()

    def __check_angles_are_same_shape(self):
        if not self.solar_zenith_angle.shape == self.emission_angle.shape == self.phase_angle.shape:
            raise ValueError('solar_zenith_angle, phase_angle, and emission_angle must all have the same shape')

    @staticmethod
    def __convert_wavelength_to_wavenumber(wavelength):
        return 1 / (wavelength * 10**-4)

    @staticmethod
    def __compute_angle_cosine(angle):
        return np.cos(np.radians(angle))

    def __compute_phi(self):
        sin_emission_angle = np.sin(np.radians(self.emission_angle))
        sin_solar_zenith_angle = np.sin(np.radians(self.solar_zenith_angle))
        if sin_emission_angle == 0 or sin_solar_zenith_angle == 0:
            d_phi = np.pi
        else:
            d_phi = np.arccos((self.__compute_angle_cosine(self.phase_angle) - self.mu * self.mu0) /
                          (sin_emission_angle * sin_solar_zenith_angle))
        return self.phi0 + 180 - np.degrees(d_phi)
