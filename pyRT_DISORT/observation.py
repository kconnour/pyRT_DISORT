"""The :code:`observation` module contains data structures to hold and compute
quantities commonly found in an observation.
"""
import numpy as np


# TODO: I'm not sure that all combination of angles are physically realistic. If
#  so, raise a warning
# TODO: Presumably a user could want phi0 that's not all 0s, so add that ability
class Angles:
    r"""A data structure that contains angles required by DISORT.

    Angles accepts the incidence, emission, and phase angles from an observation
    and computes :math:`\mu, \mu_0, \phi`, and :math:`\phi_0` from these angles.

    """

    def __init__(self, incidence: np.ndarray, emission: np.ndarray,
                 phase: np.ndarray) -> None:
        """
        Parameters
        ----------
        incidence
            Pixel incidence (solar zenith) angle [degrees]. All values must be
            between 0 and 180 degrees.
        emission
            Pixel emission (emergence) angle [degrees]. All values must be
            between 0 and 90 degrees.
        phase
            Pixel phase angle [degrees]. All values must be between 0 and 180
            degrees.

        Raises
        ------
        TypeError
            Raised if any of the angles are not a numpy.ndarray.
        ValueError
            Raised if any of the input arrays are not the same shape or if they
            contain values outside of their mathematically valid range.

        Notes
        -----
        The incidence, emission, and phase angles must have the same shape. This
        structure can accommodate pixels of any shape.

        """
        self.__incidence = incidence
        self.__emission = emission
        self.__phase = phase

        self.__raise_error_if_angles_are_bad()

        self.__mu0 = self.__compute_mu0()
        self.__mu = self.__compute_mu()
        self.__phi0 = self.__make_phi0()
        self.__phi = self.__compute_phi()

    def __raise_error_if_angles_are_bad(self) -> None:
        self.__raise_type_error_if_angles_are_not_all_ndarray()
        self.__raise_value_error_if_angles_are_not_all_same_shape()
        self.__raise_value_error_if_angles_are_unphysical()

    def __raise_type_error_if_angles_are_not_all_ndarray(self) -> None:
        self.__raise_type_error_if_angle_is_not_ndarray(
            self.__incidence, 'incidence')
        self.__raise_type_error_if_angle_is_not_ndarray(
            self.__emission, 'emission')
        self.__raise_type_error_if_angle_is_not_ndarray(
            self.__phase, 'phase')

    @staticmethod
    def __raise_type_error_if_angle_is_not_ndarray(
            angle: np.ndarray, name: str) -> None:
        if not isinstance(angle, np.ndarray):
            message = f'{name} must be a numpy.ndarray.'
            raise TypeError(message)

    def __raise_value_error_if_angles_are_not_all_same_shape(self) -> None:
        if not self.__incidence.shape == self.__emission.shape == \
               self.__phase.shape:
            message = 'incidence, emission, and phase must all have the same ' \
                      'shape.'
            raise ValueError(message)

    def __raise_value_error_if_angles_are_unphysical(self) -> None:
        self.__raise_value_error_if_angles_are_not_in_range(
            self.__incidence, 0, 180, 'incidence')
        self.__raise_value_error_if_angles_are_not_in_range(
            self.__emission, 0, 90, 'emission')
        self.__raise_value_error_if_angles_are_not_in_range(
            self.__phase, 0, 180, 'phase')

    @staticmethod
    def __raise_value_error_if_angles_are_not_in_range(
            angles: np.ndarray, low: float, high: float, name: str) -> None:
        if not (np.all(low <= angles) and np.all(angles <= high)):
            message = f'All values in {name} must be between {low} and ' \
                      f'{high} degrees.'
            raise ValueError(message)

    def __compute_mu0(self) -> np.ndarray:
        return self.__compute_angle_cosine(self.__incidence)

    def __compute_mu(self) -> np.ndarray:
        return self.__compute_angle_cosine(self.__emission)

    def __make_phi0(self) -> np.ndarray:
        return np.zeros(self.__phase.shape)

    # TODO: is there a cleaner way to make this variable?
    def __compute_phi(self) -> np.ndarray:
        sin_emission_angle = np.sin(np.radians(self.__emission))
        sin_solar_zenith_angle = np.sin(np.radians(self.__incidence))
        cos_phase_angle = self.__compute_angle_cosine(self.__phase)
        with np.errstate(divide='ignore', invalid='ignore'):
            tmp_arg = np.true_divide(
                cos_phase_angle - self.mu * self.mu0,
                sin_emission_angle * sin_solar_zenith_angle)
            tmp_arg[~np.isfinite(tmp_arg)] = -1
            d_phi = np.arccos(np.clip(tmp_arg, -1, 1))
        return self.phi0 + 180 - np.degrees(d_phi)

    @staticmethod
    def __compute_angle_cosine(angle: np.ndarray) -> np.ndarray:
        return np.cos(np.radians(angle))

    @property
    def incidence(self) -> np.ndarray:
        """Get the input incidence (solar zenith) angle [degrees].

        """
        return self.__incidence

    @property
    def emission(self) -> np.ndarray:
        """Get the input emission (emergence) angle [degrees].

        """
        return self.__emission

    @property
    def phase(self) -> np.ndarray:
        """Get the input phase angle [degrees].

        """
        return self.__phase

    @property
    def mu0(self) -> np.ndarray:
        r"""Get :math:`\mu_0` where :math:`\mu_0` is the cosine of
        :code:`incidence_angle`.

        Notes
        -----
        Each element in this variable is named :code:`UMU0` in DISORT.

        """
        return self.__mu0

    @property
    def mu(self) -> np.ndarray:
        r"""Get :math:`\mu` where :math:`\mu` is the cosine of
        :code:`emission_angle`.

        Notes
        -----
        Each element in this variable is named :code:`UMU` in DISORT.

        """
        return self.__mu

    @property
    def phi0(self) -> np.ndarray:
        r"""Get :math:`\phi_0`. I assume this is always an array of 0s.

        Notes
        -----
        Each element in this variable is named :code:`PHI0` in DISORT.

        """
        return self.__phi0

    @property
    def phi(self) -> np.ndarray:
        r"""Get :math:`\phi` where :math:`\phi` is the azimuth angle [degrees].

        Notes
        -----
        Each element in this variable is named :code:`PHI` in DISORT.

        """
        return self.__phi


class Spectral:
    """A data structure that contains spectral info required by DISORT.

    Spectral accepts the short and long wavelength from an observation and
    computes their corresponding wavenumber.

    """

    def __init__(self, short_wavelength: np.ndarray,
                 long_wavelength: np.ndarray) -> None:
        """
        Parameters
        ----------
        short_wavelength
            The short wavelength [microns] for each spectral bin.
        long_wavelength
            The long wavelength [microns] for each spectral bin.

        Raises
        ------
        TypeError
            Raised if either of the input wavelengths are not a numpy.ndarray.
        ValueError
            Raised if either of the input arrays contain values outside of 0.1
            to 50 microns (I assume this is the valid range to do retrievals),
            if they are not the same shape, or any values in
            :code:`short_wavelength` are not larger than the corresponding
            values in :code:`long_wavelength`.

        Notes
        -----
        The short and long wavelengths must have the same shape. This structure
        can accommodate pixels of any shape.

        """
        self.__short_wavelength = Wavelength(short_wavelength)
        self.__long_wavelength = Wavelength(long_wavelength)

        self.__raise_error_if_wavelengths_are_bad()

        self.__high_wavenumber = self.__calculate_high_wavenumber()
        self.__low_wavenumber = self.__calculate_low_wavenumber()

    def __raise_error_if_wavelengths_are_bad(self) -> None:
        self.__raise_value_error_if_wavelengths_are_not_all_same_shape()
        self.__raise_value_error_if_long_wavelength_is_not_larger()

    def __raise_value_error_if_wavelengths_are_not_all_same_shape(self) -> None:
        if self.__short_wavelength.wavelength.shape != \
                self.__long_wavelength.wavelength.shape:
            message = 'short_wavelength and long_wavelength must both have ' \
                      'the same shape.'
            raise ValueError(message)

    def __raise_value_error_if_long_wavelength_is_not_larger(self) -> None:
        if np.any(self.__short_wavelength.wavelength >=
                  self.__long_wavelength.wavelength):
            message = 'Some values in long_wavelength are not larger ' \
                      'than the corresponding values in short_wavelength.'
            raise ValueError(message)

    def __calculate_high_wavenumber(self) -> np.ndarray:
        return self.__short_wavelength.wavelength_to_wavenumber()

    def __calculate_low_wavenumber(self) -> np.ndarray:
        return self.__long_wavelength.wavelength_to_wavenumber()

    @property
    def short_wavelength(self) -> np.ndarray:
        """Get the input short wavelength [microns].

        """
        return self.__short_wavelength.wavelength

    @property
    def long_wavelength(self) -> np.ndarray:
        """Get the input long wavelength [microns].

        """
        return self.__long_wavelength.wavelength

    @property
    def high_wavenumber(self) -> np.ndarray:
        r"""Get the high wavenumber [cm :superscript:`-1`]---the wavenumber
        corresponding to :code:`short_wavelength`.

        Notes
        -----
        In DISORT, this variable is named :code:`WVNMHI`. It is only needed by
        DISORT if :py:attr:`~radiation.ThermalEmission.thermal_emission` is set
        to :code:`True`.

        """
        return self.__high_wavenumber

    @property
    def low_wavenumber(self) -> np.ndarray:
        r"""Get the low wavenumber [cm :superscript:`-1`]---the wavenumber
        corresponding to :code:`long_wavelength`.

        Notes
        -----
        In DISORT, this variable is named :code:`WVNMLO`. It is only needed by
        DISORT if :py:attr:`~radiation.ThermalEmission.thermal_emission` is set
        to :code:`True`.

        """
        return self.__low_wavenumber


class Wavelength:
    """A class to hold on to an array of wavelengths.

    Wavelength accepts a numpy.ndarray of wavelengths and ensures all values
    in the array are acceptable for retrievals.

    """

    def __init__(self, wavelength: np.ndarray) -> None:
        """
        Parameters
        ----------
        wavelength
            Any ND array of wavelengths [microns].

        Raises
        ------
        TypeError
            Raised if :code:`wavelength` is not an instance of numpy.ndarray.
        ValueError
            Raised if any values in :code:`wavelength` are not between 0.1 and
            50 microns (I assume this is the valid range to do retrievals).

        """
        self.__wavelength = wavelength

        self.__raise_error_if_wavelength_is_bad()

    def __raise_error_if_wavelength_is_bad(self) -> None:
        self.__raise_type_error_if_wavelength_is_not_ndarray()
        self.__raise_value_error_if_wavelength_is_not_in_valid_range()

    def __raise_type_error_if_wavelength_is_not_ndarray(self) -> None:
        if not isinstance(self.__wavelength, np.ndarray):
            message = 'wavelength must be a numpy.ndarray.'
            raise TypeError(message)

    # TODO: I think this logic can be shortened
    def __raise_value_error_if_wavelength_is_not_in_valid_range(self) -> None:
        if not (np.all(0.1 <= self.__wavelength) and
                np.all(self.__wavelength <= 50)):
            message = 'All values in wavelength must be between 0.1 and 50 ' \
                      'microns.'
            raise ValueError(message)

    @property
    def wavelength(self) -> np.ndarray:
        """Get the input wavelength.

        """
        return self.__wavelength

    def wavelength_to_wavenumber(self) -> np.ndarray:
        """Convert the input wavelength to wavenumber.

        """
        return 10 ** 4 / self.__wavelength
