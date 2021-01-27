"""observation.py contains data structures to hold quantities commonly found in
an observation.
"""
import numpy as np
from pylint import epylint as lint
from pyRT_DISORT.utilities.array_checks import ArrayChecker


# TODO: Add latex to all docstrings. It'd be nice to say mu0 = cos(sza) for
#  whatever symbol sza is.
# TODO: I don't think all possible combos of angles are mathematically possible.
#  If so, raise a warning.
class Angles:
    """Angles is a data structure to contain all angles required by DISORT.

    Angles accepts ``observation'' angles and computes mu, mu0, phi, and phi0
    from these angles.

    """

    def __init__(self, incidence_angles: np.ndarray,
                 emission_angles: np.ndarray, phase_angles: np.ndarray) -> None:
        """Initialize the structure.

        Parameters
        ----------
        incidence_angles: np.ndarray
            Flattened array of the pixel incidence angles [degrees].
        emission_angles: np.ndarray
            Flattened array of the pixel emission angles [degrees].
        phase_angles: np.ndarray
            Flattened array of the pixel phase angles [degrees].

        Raises
        ------
        TypeError
            Raised if any of the inputs are not np.ndarrays.
        ValueError
            Raised if any of the inputs are not 1D arrays, or outside their
            possible range of values.

        """
        self.__incidence = incidence_angles
        self.__emission = emission_angles
        self.__phase = phase_angles

        self.__raise_error_if_input_angles_are_bad()

        self.__mu0 = self.__compute_mu0()
        self.__mu = self.__compute_mu()
        self.__phi0 = self.__make_phi0()
        self.__phi = self.__compute_phi()

    def __raise_error_if_input_angles_are_bad(self) -> None:
        self.__raise_error_if_incidence_angles_are_bad()
        self.__raise_error_if_emission_angles_are_bad()
        self.__raise_error_if_phase_angles_are_bad()
        self.__raise_value_error_if_angles_are_not_same_shape()

    def __raise_error_if_incidence_angles_are_bad(self) -> None:
        self.__raise_error_if_angles_are_bad(
            self.__incidence, 'incidence_angles', 0, 180)

    def __raise_error_if_emission_angles_are_bad(self) -> None:
        self.__raise_error_if_angles_are_bad(
            self.__emission, 'emission_angles', 0, 90)

    def __raise_error_if_phase_angles_are_bad(self) -> None:
        self.__raise_error_if_angles_are_bad(
            self.__phase, 'phase_angles', 0, 180)

    def __raise_error_if_angles_are_bad(self, angle: np.ndarray, name: str,
                                        low: int, high: int) -> None:
        try:
            checks = self.__make_angle_checks(angle, low, high)
        except TypeError:
            raise TypeError(f'{name} is not a np.ndarray.')
        if not all(checks):
            raise ValueError(
                f'{name} must be a 1D array in range [{low}, {high}]')

    @staticmethod
    def __make_angle_checks(angle: np.ndarray, low: int, high: int) \
            -> list[bool]:
        angle_checker = ArrayChecker(angle)
        checks = [angle_checker.determine_if_array_is_numeric(),
                  angle_checker.determine_if_array_is_in_range(low, high),
                  angle_checker.determine_if_array_is_1d()]
        return checks

    def __raise_value_error_if_angles_are_not_same_shape(self) -> None:
        same_shape = self.__incidence.shape == self.__emission.shape == \
                     self.__phase.shape
        if not same_shape:
            raise ValueError('incidence_angles, emission_angles, and '
                             'phase_angles must all have the same shape.')

    def __compute_mu0(self) -> np.ndarray:
        return self.__compute_angle_cosine(self.__incidence)

    def __compute_mu(self) -> np.ndarray:
        return self.__compute_angle_cosine(self.__emission)

    @staticmethod
    def __compute_angle_cosine(angle: np.ndarray) -> np.ndarray:
        return np.cos(np.radians(angle))

    def __make_phi0(self) -> np.ndarray:
        return np.zeros(len(self.__phase))

    # TODO: This feels really messy mathematically... can I do better?
    def __compute_phi(self) -> np.ndarray:
        with np.errstate(invalid='raise'):
            sin_emission_angle = np.sin(np.radians(self.__emission))
            sin_solar_zenith_angle = np.sin(np.radians(self.__incidence))
            cos_phase_angle = self.__compute_angle_cosine(self.__phase)
            try:
                tmp_arg = (cos_phase_angle - self.mu * self.mu0) / \
                          (sin_emission_angle * sin_solar_zenith_angle)
                d_phi = np.arccos(np.clip(tmp_arg, -1, 1))
            except FloatingPointError:
                d_phi = np.pi
            return self.phi0 + 180 - np.degrees(d_phi)

    @property
    def emission(self) -> np.ndarray:
        """Get the input emission angles [degrees].

        Returns
        -------
        np.ndarray
            The input emission angles.

        """
        return self.__emission

    @property
    def incidence(self) -> np.ndarray:
        """Get the input incidence (solar zenith) angles [degrees].

        Returns
        -------
        np.ndarray
            The input incidence angles.

        """
        return self.__incidence

    @property
    def mu(self) -> np.ndarray:
        """Compute mu where mu is the cosine of the input emission angles.

        Returns
        -------
        np.ndarray
            The cosine of the input emission angles.

        """
        return self.__mu

    @property
    def mu0(self) -> np.ndarray:
        """Compute mu0 where mu0 is the cosine of the input incidence angles.

        Returns
        -------
        np.ndarray
            The cosine of the input incidence angles.

        """
        return self.__mu0

    @property
    def phase(self) -> np.ndarray:
        """Get the input phase angles [degrees].

        Returns
        -------
        np.ndarray
            The input phase angles.

        """
        return self.__phase

    @property
    def phi(self) -> np.ndarray:
        """Compute phi where phi is the azimuth angle [degrees].

        Returns
        -------
        np.ndarray
            The azimuth angles.

        """
        return self.__phi

    @property
    def phi0(self) -> np.ndarray:
        """Compute phi0. I assume this is always 0.

        Returns
        -------
        np.ndarray
            All 0s.

        """
        return self.__phi0


'''class Wavelengths:
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

    # TODO: warn if not in 100 nm -- 50 microns'''


if __name__ == '__main__':
    lint.py_run('observation.py')
