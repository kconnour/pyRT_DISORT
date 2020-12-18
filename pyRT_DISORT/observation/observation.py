# 3rd-party imports
import numpy as np
import time

# Local imports
from pyRT_DISORT.utilities.array_checks import ArrayChecker


class Observation:
    """An Observation object holds the relevant observational information for doing RT."""
    def __init__(self, solar_zenith_angle, emission_angle, phase_angle):
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
        self.solar_zenith_angle = solar_zenith_angle
        self.emission_angle = emission_angle
        self.phase_angle = phase_angle

        self.mu0 = self.__compute_angle_cosine(self.solar_zenith_angle)
        self.mu = self.__compute_angle_cosine(self.emission_angle)
        self.phi0 = 0
        self.phi = self.__compute_phi()

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


class Angles:
    def __init__(self, solar_zenith_angles, emission_angles, phase_angles):
        """ An Angles object holds the input angles and computes mu, mu0, phi, and phi0 from these angles.
        Parameters
        ----------
        solar_zenith_angles: np.ndarray
            Flattened array of the pixel solar zenith angles [degrees].
        emission_angles: np.ndarray
            Flattened array of the pixel emission angles [degrees].
        phase_angles: np.ndarray
            Flattened array of the pixel phase angles [degrees].

        Attributes
        ----------
        solar_zenith_angles: np.ndarray
            The input solar zenith angles.
        emission_angles: np.ndarray
            The input emission angles.
        phase_angles: np.ndarray
            The input phase angles.
        mu: np.ndarray
            The cosine of the input emission angles.
        mu0: np.ndarray
            The cosine of the input solar zenith angles.
        phi: np.ndarray
            The computed phi angles [degrees].
        phi0: np.ndarray
            0 (I'm assuming phi0 is always 0).
        """
        self.solar_zenith_angles = solar_zenith_angles
        self.emission_angles = emission_angles
        self.phase_angles = phase_angles

        self.__check_angles_are_physical()

        self.mu = self.__compute_angle_cosine(self.emission_angles)
        self.mu0 = self.__compute_angle_cosine(self.solar_zenith_angles)
        self.phi0 = self.__make_phi0()
        self.phi = self.__compute_phi()

    def __check_angles_are_physical(self):
        self.__check_solar_zenith_angles_are_physical()
        self.__check_emission_angles_are_physical()
        self.__check_phase_angles_are_physical()
        self.__check_angles_are_same_shape()

    def __check_solar_zenith_angles_are_physical(self):
        solar_zenith_angle_checker = ArrayChecker(self.solar_zenith_angles, 'solar_zenith_angles')
        solar_zenith_angle_checker.check_object_is_array()
        solar_zenith_angle_checker.check_ndarray_is_numeric()
        solar_zenith_angle_checker.check_ndarray_is_in_range(0, 180)
        solar_zenith_angle_checker.check_ndarray_is_1d()

    def __check_emission_angles_are_physical(self):
        emission_angle_checker = ArrayChecker(self.emission_angles, 'emission_angles')
        emission_angle_checker.check_object_is_array()
        emission_angle_checker.check_ndarray_is_numeric()
        emission_angle_checker.check_ndarray_is_in_range(0, 90)
        emission_angle_checker.check_ndarray_is_1d()

    def __check_phase_angles_are_physical(self):
        phase_angle_checker = ArrayChecker(self.phase_angles, 'phase_angles')
        phase_angle_checker.check_object_is_array()
        phase_angle_checker.check_ndarray_is_numeric()
        phase_angle_checker.check_ndarray_is_in_range(0, 180)
        phase_angle_checker.check_ndarray_is_1d()

    def __check_angles_are_same_shape(self):
        if not self.solar_zenith_angles.shape == self.emission_angles.shape == self.phase_angles.shape:
            raise ValueError('solar_zenith_angles, phase_angles, and emission_angles must all have the same shape')

    @staticmethod
    def __compute_angle_cosine(angle):
        return np.cos(np.radians(angle))

    def __make_phi0(self):
        return np.zeros(len(self.phase_angles))

    def __compute_phi(self):
        with np.errstate(invalid='ignore'):
            try:
                sin_emission_angle = np.sin(np.radians(self.emission_angles))
                sin_solar_zenith_angle = np.sin(np.radians(self.solar_zenith_angles))
                d_phi = np.where(np.logical_or(sin_emission_angle == 0, sin_solar_zenith_angle == 0), np.pi,
                                 np.arccos(np.clip(
                                     (self.__compute_angle_cosine(self.phase_angles) - self.mu * self.mu0) / \
                                     (sin_emission_angle * sin_solar_zenith_angle), -1, 1)))
                return self.phi0 + 180 - np.degrees(d_phi)
            except FloatingPointError:
                raise ValueError(f'There is an error with at least one of the angles when computing phi')


class Wavelengths:
    def __init__(self, short_wavelengths, long_wavelengths):
        """ A Wavelengths object holds the input short and long wavelengths and their corresponding wavenumbers.

        Parameters
        ----------
        short_wavelengths: np.ndarray
            The short wavelengths [microns] for each spectral bin.
        long_wavelengths: np.ndarray
            The long wavelengths [microns] for each spectral bin.

        Attributes
        ----------
        short_wavelength: np.ndarray
            The input short wavelengths.
        long_wavelength: np.ndarray
            The input long wavelengths.
        low_wavenumber: np.ndarray
            The wavenumbers of long_wavelength [1 /cm].
        high_wavenumber: np.ndarray
            The wavenumbers of short_wavelength [1 /cm].
        """
        self.short_wavelengths = short_wavelengths
        self.long_wavelengths = long_wavelengths

        self.__check_wavelengths_are_physical()

        self.low_wavenumbers = self.__convert_wavelengths_to_wavenumbers(self.long_wavelengths, 'long_wavelengths')
        self.high_wavenumbers = self.__convert_wavelengths_to_wavenumbers(self.short_wavelengths, 'short_wavelengths')

    def __check_wavelengths_are_physical(self):
        self.__check_short_wavelengths_are_physical()
        self.__check_long_wavelengths_are_physical()
        self.__check_wavelengths_are_same_shape()
        self.__check_long_wavelength_is_larger()

    def __check_short_wavelengths_are_physical(self):
        short_wav_checker = ArrayChecker(self.short_wavelengths, 'short_wavelengths')
        short_wav_checker.check_object_is_array()
        short_wav_checker.check_ndarray_is_numeric()
        short_wav_checker.check_ndarray_is_positive_finite()
        short_wav_checker.check_ndarray_is_1d()

    def __check_long_wavelengths_are_physical(self):
        long_wav_checker = ArrayChecker(self.long_wavelengths, 'long_wavelengths')
        long_wav_checker.check_object_is_array()
        long_wav_checker.check_ndarray_is_numeric()
        long_wav_checker.check_ndarray_is_positive_finite()
        long_wav_checker.check_ndarray_is_1d()

    def __check_wavelengths_are_same_shape(self):
        if self.short_wavelengths.shape != self.long_wavelengths.shape:
            raise ValueError('short_wavelengths and long_wavelengths must have the same shape')

    def __check_long_wavelength_is_larger(self):
        if not np.all(self.long_wavelengths > self.short_wavelengths):
            raise ValueError('long_wavelengths must always be larger than the corresponding short_wavelengths')

    @staticmethod
    def __convert_wavelengths_to_wavenumbers(wavelength, wavelength_name):
        with np.errstate(divide='raise'):
            try:
                return 1 / (wavelength * 10 ** -4)
            except FloatingPointError:
                raise ValueError(f'At least one value in {wavelength_name} is too small to perform calculations!')
