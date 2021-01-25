# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.utilities.array_checks import ArrayChecker


# TODO: see if I can do np.ndarray[float] in the type hinting
class Angles:
    def __init__(self, incidence_angles: np.ndarray,
                 emission_angles: np.ndarray, phase_angles: np.ndarray) -> None:
        """ An Angles data structure holds the input angles and computes
        mu, mu0, phi, and phi0 from these angles.

        Parameters
        ----------
        incidence_angles: np.ndarray
            Flattened array of the pixel incidence angles [degrees].
        emission_angles: np.ndarray
            Flattened array of the pixel emission angles [degrees].
        phase_angles: np.ndarray
            Flattened array of the pixel phase angles [degrees].
        """
        self.__incidence = incidence_angles
        self.__emission = emission_angles
        self.__phase = phase_angles

        self.__check_angles_are_physical()

    def __check_angles_are_physical(self) -> None:
        self.__check_incidence_angles_are_physical()
        self.__check_emission_angles_are_physical()
        self.__check_phase_angles_are_physical()
        self.__raise_value_error_if_angles_are_not_same_shape()

    def __check_incidence_angles_are_physical(self) -> None:
        incidence_angle_checker = ArrayChecker(self.__incidence,
                                               'incidence_angles')
        incidence_angle_checker.check_object_is_array()
        incidence_angle_checker.check_ndarray_is_numeric()
        incidence_angle_checker.check_ndarray_is_in_range(0, 180)
        incidence_angle_checker.check_ndarray_is_1d()

    def __check_emission_angles_are_physical(self) -> None:
        emission_angle_checker = ArrayChecker(self.__emission,
                                              'emission_angles')
        emission_angle_checker.check_object_is_array()
        emission_angle_checker.check_ndarray_is_numeric()
        emission_angle_checker.check_ndarray_is_in_range(0, 90)
        emission_angle_checker.check_ndarray_is_1d()

    def __check_phase_angles_are_physical(self) -> None:
        phase_angle_checker = ArrayChecker(self.__phase, 'phase_angles')
        phase_angle_checker.check_object_is_array()
        phase_angle_checker.check_ndarray_is_numeric()
        phase_angle_checker.check_ndarray_is_in_range(0, 180)
        phase_angle_checker.check_ndarray_is_1d()

    def __raise_value_error_if_angles_are_not_same_shape(self) -> None:
        if not self.__incidence.shape == self.__emission.shape == \
               self.__phase.shape:
            raise ValueError('incidence_angles, phase_angles, and '
                             'emission_angles must all have the same shape.')

    @property
    def emission(self) -> np.ndarray:
        """ Get the input emission angles.

        Returns
        -------
        emission_angle: np.ndarray
            The emission angles.
        """
        return self.__emission

    @property
    def incidence(self) -> np.ndarray:
        """ Get the input incidence (solar zenith) angles.

        Returns
        -------
        incidence_angle: np.ndarray
            The incidence angles.
        """
        return self.__incidence

    @property
    def mu(self) -> np.ndarray:
        """ Compute mu: the cosine of the input emission angles.

        Returns
        -------
        mu: np.ndarray
            The mu angles.
        """
        return self.__compute_angle_cosine(self.__emission)

    @property
    def mu0(self) -> np.ndarray:
        """ Compute mu0: the cosine of the input incidence angles.

        Returns
        -------
        mu0: np.ndarray
            The mu0 angles.
        """
        return self.__compute_angle_cosine(self.__incidence)

    @property
    def phase(self) -> np.ndarray:
        """ Get the input phase angles.

        Returns
        -------
        phase_angle: np.ndarray
            The phase angles.
        """
        return self.__phase

    @property
    def phi(self) -> np.ndarray:
        """

        Returns
        -------

        """
        return self.__make_phi()

    @property
    def phi0(self) -> np.ndarray:
        """ Compute phi0. I assume this is always 0.

        Returns
        -------
        phi0: np.ndarray
            All 0s
        """
        return self.__make_phi0()

    @staticmethod
    def __compute_angle_cosine(angle: np.ndarray) -> np.ndarray:
        return np.cos(np.radians(angle))

    def __compute_phi(self) -> np.ndarray:
        sin_emission_angle = np.sin(np.radians(self.__emission))
        sin_solar_zenith_angle = np.sin(np.radians(self.__incidence))
        cos_phase_angle = self.__compute_angle_cosine(self.__phase)
        d_phi = (cos_phase_angle - self.mu * self.mu0) / (
                sin_emission_angle * sin_solar_zenith_angle)
        return self.phi0 + 180 - np.degrees(d_phi)

    def __make_phi0(self) -> np.ndarray:
        return np.zeros(len(self.__phase))

    def __make_phi(self) -> np.ndarray:
        with np.errstate(invalid='raise'):
            try:
                return self.__compute_phi()
            except FloatingPointError:
                err_msg = 'Cannot compute the arccosine for computing phi. ' \
                          'This likely means you input an unrealistic ' \
                          'combination of angles.'
                raise FloatingPointError(err_msg)
            except ZeroDivisionError:
                err_msg = 'Cannot compute the arccosine for computing phi. ' \
                          'This likely means the input emission and ' \
                          'incidence angles are too small.'
                raise ZeroDivisionError(err_msg)


class Wavelengths:
    def __init__(self, short_wavelengths: np.ndarray,
                 long_wavelengths: np.ndarray) -> None:
        """ A Wavelengths data structure holds the input wavelengths and their
        corresponding wavenumbers.

        Parameters
        ----------
        short_wavelengths: np.ndarray
            The short wavelengths [microns] for each spectral bin.
        long_wavelengths: np.ndarray
            The long wavelengths [microns] for each spectral bin.
        """
        self.__short_wavelengths = short_wavelengths
        self.__long_wavelengths = long_wavelengths

        self.__check_wavelengths_are_physical()

    def __check_wavelengths_are_physical(self) -> None:
        self.__check_short_wavelengths_are_physical()
        self.__check_long_wavelengths_are_physical()
        self.__check_wavelengths_are_same_shape()
        self.__check_long_wavelength_is_larger()

    def __check_short_wavelengths_are_physical(self) -> None:
        short_wav_checker = ArrayChecker(self.__short_wavelengths,
                                         'short_wavelengths')
        short_wav_checker.check_object_is_array()
        short_wav_checker.check_ndarray_is_numeric()
        short_wav_checker.check_ndarray_is_positive_finite()
        short_wav_checker.check_ndarray_is_1d()

    def __check_long_wavelengths_are_physical(self) -> None:
        long_wav_checker = ArrayChecker(self.__long_wavelengths,
                                        'long_wavelengths')
        long_wav_checker.check_object_is_array()
        long_wav_checker.check_ndarray_is_numeric()
        long_wav_checker.check_ndarray_is_positive_finite()
        long_wav_checker.check_ndarray_is_1d()

    def __check_wavelengths_are_same_shape(self) -> None:
        if self.__short_wavelengths.shape != self.__long_wavelengths.shape:
            raise ValueError('short_wavelengths and long_wavelengths must '
                             'have the same shape.')

    def __check_long_wavelength_is_larger(self) -> None:
        if not np.all(self.long_wavelengths > self.short_wavelengths):
            raise ValueError('long_wavelengths must always be larger than '
                             'the corresponding short_wavelengths.')

    @property
    def high_wavenumber(self) -> np.ndarray:
        """ Calculate the wavenumbers corresponding to short_wavelength [1 /cm].

        Returns
        -------
        high_wavenumber: np.ndarray
            The low wavenumbers.
        """
        return self.__make_wavenumber_from_wavelength(
            self.__short_wavelengths, 'short_wavelengths')

    @property
    def long_wavelengths(self) -> np.ndarray:
        """ Get the long wavelengths [microns].

        Returns
        -------
        long_wavelengths: np.ndarray
            The long wavelengths.
        """
        return self.__long_wavelengths

    @property
    def low_wavenumber(self) -> np.ndarray:
        """ Calculate the wavenumbers corresponding to long_wavelength [1 /cm].

        Returns
        -------
        low_wavenumber: np.ndarray
            The low wavenumbers.
        """
        return self.__make_wavenumber_from_wavelength(
            self.__long_wavelengths, 'long_wavelengths')

    @property
    def short_wavelengths(self) -> np.ndarray:
        """ Get the short wavelengths [microns].

        Returns
        -------
        short_wavelengths: np.ndarray
            The short wavelengths.
        """
        return self.__short_wavelengths

    def __make_wavenumber_from_wavelength(self, wavelength: np.ndarray,
                                          wavelength_name: str):
        with np.errstate(divide='raise'):
            try:
                self.__convert_wavelengths_to_wavenumber(wavelength)
            except FloatingPointError:
                raise ValueError(f'At least one value in {wavelength_name} '
                                 f'is too small to perform calculations!')

    @staticmethod
    def __convert_wavelengths_to_wavenumber(wavelength: np.ndarray) \
            -> np.ndarray:
        return 1 / (wavelength * 10 ** -4)

    # TODO: warn if not in 100 nm -- 50 microns
