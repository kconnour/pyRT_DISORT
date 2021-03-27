"""The :code:`observation` module contains data structures to hold and compute
quantities commonly found in an observation.
"""
import numpy as np


# TODO: I'm not sure that all combination of angles are physically realistic. If
#  so, raise a warning
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
        This class can accommodate arrays of any shape as long as they all have
        the same shape.

        """
        self.__incidence = _Angle(incidence, 'incidence', 0, 180)
        self.__emission = _Angle(emission, 'emission', 0, 90)
        self.__phase = _Angle(phase, 'phase', 0, 180)

        self.__raise_error_if_angles_are_bad()

        self.__mu0 = self.__compute_mu0()
        self.__mu = self.__compute_mu()
        self.__phi0 = self.__make_phi0()
        self.__phi = self.__compute_phi()

    def __raise_error_if_angles_are_bad(self) -> None:
        self.__raise_value_error_if_angles_are_not_all_same_shape()

    def __raise_value_error_if_angles_are_not_all_same_shape(self) -> None:
        if not self.__incidence.shape == self.__emission.shape == \
               self.__phase.shape:
            message = 'incidence, emission, and phase must all have the same ' \
                      'shape.'
            raise ValueError(message)

    def __compute_mu0(self) -> np.ndarray:
        return self.__compute_angle_cosine(self.__incidence.val)

    def __compute_mu(self) -> np.ndarray:
        return self.__compute_angle_cosine(self.__emission.val)

    # TODO: Presumably a user could want phi0 that's not all 0s, so add that
    #  ability and remake property docstring
    def __make_phi0(self) -> np.ndarray:
        return np.zeros(self.__phase.shape)

    # TODO: is there a cleaner way to make this variable?
    def __compute_phi(self) -> np.ndarray:
        sin_emission_angle = np.sin(np.radians(self.__emission.val))
        sin_solar_zenith_angle = np.sin(np.radians(self.__incidence.val))
        cos_phase_angle = self.__compute_angle_cosine(self.__phase.val)
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
        """Get the input incidence angle [degrees].

        """
        return self.__incidence.val

    @property
    def emission(self) -> np.ndarray:
        """Get the input emission angle [degrees].

        """
        return self.__emission.val

    @property
    def phase(self) -> np.ndarray:
        """Get the input phase angle [degrees].

        """
        return self.__phase.val

    @property
    def mu0(self) -> np.ndarray:
        r"""Get :math:`\mu_0`---the cosine of ``incidence``.

        Notes
        -----
        Each element in this variable is named ``UMU0`` in DISORT.

        """
        return self.__mu0

    @property
    def mu(self) -> np.ndarray:
        r"""Get :math:`\mu`---the cosine of ``emission``.

        Notes
        -----
        Each element in this variable is named ``UMU`` in DISORT.

        """
        return self.__mu

    @property
    def phi0(self) -> np.ndarray:
        r"""Get :math:`\phi_0`. I assume this is always an array of 0s.

        Notes
        -----
        Each element in this variable is named ``PHI0`` in DISORT.

        """
        return self.__phi0

    @property
    def phi(self) -> np.ndarray:
        r"""Get :math:`\phi`---the azimuth angle [degrees].

        Notes
        -----
        Each element in this variable is named ``PHI`` in DISORT.

        """
        return self.__phi


class _Angle:
    """A data structure to hold on to an array of angles.

    _Angle accepts a numpy.ndarray of angles and ensures all values in the array
    are within a range of values (inclusive).

    """

    def __init__(self, angle: np.ndarray, name: str, low: float, high: float) \
            -> None:
        """
        Parameters
        ----------
        angle
            Array of angles [microns].
        name
            Name of the angle.
        low
            The lowest value any value in ``angle`` can be.
        high
            The highest value any value in ``angle`` can be.

        Raises
        ------
        TypeError
            Raised if ``angle`` is not a numpy.ndarray.
        ValueError
            Raised if any value in ``angle`` is outside its allowable range.

        """
        self.__angle = angle
        self.__name = name
        self.__low = low
        self.__high = high

        self.__raise_error_if_angle_is_bad()

    def __raise_error_if_angle_is_bad(self) -> None:
        self.__raise_type_error_if_angle_is_not_ndarray()
        self.__raise_value_error_if_angles_are_not_in_range()

    def __raise_type_error_if_angle_is_not_ndarray(self) -> None:
        if not isinstance(self.__angle, np.ndarray):
            message = f'{self.__name} must be a numpy.ndarray.'
            raise TypeError(message)

    def __raise_value_error_if_angles_are_not_in_range(self) -> None:
        if not (np.all(self.__low <= self.__angle) and
                np.all(self.__angle <= self.__high)):
            message = f'All values in {self.__name} must be between ' \
                      f'{self.__low} and {self.__high} degrees.'
            raise ValueError(message)

    def __getattr__(self, method):
        return getattr(self.val, method)

    @property
    def val(self) -> np.ndarray:
        return self.__angle


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
            The short wavelength [microns] of each spectral bin.
        long_wavelength
            The long wavelength [microns] of each spectral bin.

        Raises
        ------
        TypeError
            Raised if either of the wavelengths are not a numpy.ndarray.
        ValueError
            Raised if either of the input arrays are not the same shape, if any
            values in ``short_wavelength`` are not larger than the corresponding
            values in ``long_wavelength``, or if either of the input arrays
            contain values outside of 0.1 to 50 microns (I assume this is the
            valid range to do retrievals).

        Notes
        -----
        This class can accommodate arrays of any shape as long as they both have
        the same shape.

        """
        self.__short_wavelength = _Wavelength(short_wavelength,
                                              'short_wavelength')
        self.__long_wavelength = _Wavelength(long_wavelength, 'long_wavelength')

        self.__raise_error_if_wavelengths_are_bad()

        self.__high_wavenumber = self.__calculate_high_wavenumber()
        self.__low_wavenumber = self.__calculate_low_wavenumber()

    def __raise_error_if_wavelengths_are_bad(self) -> None:
        self.__raise_value_error_if_not_same_shape()
        self.__raise_value_error_if_long_wavelength_is_not_larger()

    def __raise_value_error_if_not_same_shape(self) -> None:
        if self.__short_wavelength.shape != self.__long_wavelength.shape:
            message = 'short_wavelength and long_wavelength must both have ' \
                      'the same shape.'
            raise ValueError(message)

    def __raise_value_error_if_long_wavelength_is_not_larger(self) -> None:
        if np.any(self.__short_wavelength.val >= self.__long_wavelength.val):
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
        return self.__short_wavelength.val

    @property
    def long_wavelength(self) -> np.ndarray:
        """Get the input long wavelength [microns].

        """
        return self.__long_wavelength.val

    @property
    def high_wavenumber(self) -> np.ndarray:
        r"""Get the high wavenumber [cm :sup:`-1`]---the wavenumber
        corresponding to ``short_wavelength``.

        Notes
        -----
        In DISORT, this variable is named ``WVNMHI``. It is only needed by
        DISORT if :py:attr:`~radiation.ThermalEmission.thermal_emission` is set
        to ``True``.

        """
        return self.__high_wavenumber

    @property
    def low_wavenumber(self) -> np.ndarray:
        r"""Get the low wavenumber [cm :sup:`-1`]---the wavenumber corresponding
        to ``long_wavelength``.

        Notes
        -----
        In DISORT, this variable is named ``WVNMLO``. It is only needed by
        DISORT if :py:attr:`~radiation.ThermalEmission.thermal_emission` is set
        to ``True``.

        """
        return self.__low_wavenumber


class _Wavelength:
    """A data structure to hold on to an array of wavelengths.

    _Wavelength accepts a numpy.ndarray of wavelengths and ensures all values
    in the array are acceptable for retrievals.

    """

    def __init__(self, wavelength: np.ndarray, name: str) -> None:
        """
        Parameters
        ----------
        wavelength
            Array of wavelengths [microns].
        name
            Name of the wavelength.

        Raises
        ------
        TypeError
            Raised if ``wavelength`` is not a numpy.ndarray.
        ValueError
            Raised if any values in ``wavelength`` are not between 0.1 and 50
            microns (I assume this is the valid range to do retrievals).

        """
        self.__wavelength = wavelength
        self.__name = name

        self.__raise_error_if_wavelength_is_bad()

    def __raise_error_if_wavelength_is_bad(self) -> None:
        self.__raise_type_error_if_wavelength_is_not_ndarray()
        self.__raise_value_error_if_wavelength_is_not_in_valid_range()

    def __raise_type_error_if_wavelength_is_not_ndarray(self) -> None:
        if not isinstance(self.__wavelength, np.ndarray):
            message = 'wavelength must be a numpy.ndarray.'
            raise TypeError(message)

    def __raise_value_error_if_wavelength_is_not_in_valid_range(self) -> None:
        if not np.all((0.1 <= self.__wavelength) & (self.__wavelength <= 50)):
            message = f'All values in {self.__name} must be between 0.1 and ' \
                      f'50 microns.'
            raise ValueError(message)

    def __getattr__(self, method):
        return getattr(self.val, method)

    @property
    def val(self) -> np.ndarray:
        """Get the value of the input wavelength.

        """
        return self.__wavelength

    def wavelength_to_wavenumber(self) -> np.ndarray:
        """Convert the input wavelength to wavenumber.

        """
        return 10 ** 4 / self.__wavelength
