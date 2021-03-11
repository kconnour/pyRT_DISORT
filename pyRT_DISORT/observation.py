"""The observation module contains data structures to hold and compute
quantities commonly found in an observation.
"""
import numpy as np


# TODO: I'm not sure that all combination of angles are physically realistic. If
#  so, raise a warning
# TODO: Presumably a user could want phi0 that's not all 0s, so add that ability
# TODO: If SZA, EA, and PA have standard symbols, it'd be great to update all
#  docstrings to be equations instead of mu is the cosine of the ea.
class Angles:
    r"""A data structure that contains angles required by DISORT.

    Angles accepts the incidence, emission, and phase angles from an observation
    and computes :math:`\mu, \mu_0, \phi`, and :math:`\phi_0` from these angles.

    """

    def __init__(self, incidence_angle: np.ndarray,
                 emission_angle: np.ndarray, phase_angle: np.ndarray) -> None:
        """
        Parameters
        ----------
        incidence_angle
            Pixel incidence angles [degrees].
        emission_angle
            Pixel emission angles [degrees].
        phase_angle
            Pixel phase angles [degrees].

        Raises
        ------
        TypeError
            Raised if any of the inputs are not an instance of numpy.ndarray, or
            if the arrays contain non-numeric values.
        ValueError
            Raised if any of the input arrays contain values outside of their
            mathematically valid range, or if the arrays cannot be broadcast
            together.

        """
        self.__incidence_angle = incidence_angle
        self.__emission_angle = emission_angle
        self.__phase_angle = phase_angle

        self.__raise_error_if_angles_are_unphysical()

        self.__mu0 = self.__compute_mu0()
        self.__mu = self.__compute_mu()
        self.__phi0 = self.__make_phi0()
        self.__phi = self.__compute_phi()

    def __raise_error_if_angles_are_unphysical(self) -> None:
        self.__raise_error_if_angles_are_not_in_range(
            self.__incidence_angle, 0, 180, 'incidence_angle')
        self.__raise_error_if_angles_are_not_in_range(
            self.__emission_angle, 0, 90, 'emission_angle')
        self.__raise_error_if_angles_are_not_in_range(
            self.__phase_angle, 0, 180, 'phase_angle')

    @staticmethod
    def __raise_error_if_angles_are_not_in_range(
            angles: np.ndarray, low: float, high: float, name: str) -> None:
        try:
            if np.any(angles < low) or np.any(angles > high):
                message = f'{name} must be between {low} and {high} degrees.'
                raise ValueError(message)
        except TypeError as te:
            message = f'{name} must be a numpy.ndarray of numeric values.'
            raise TypeError(message) from te

    def __compute_mu0(self) -> np.ndarray:
        return self.__compute_angle_cosine(self.__incidence_angle)

    def __compute_mu(self) -> np.ndarray:
        return self.__compute_angle_cosine(self.__emission_angle)

    def __make_phi0(self) -> np.ndarray:
        return np.zeros(self.__phase_angle.shape)

    # TODO: is there a cleaner way to make this variable?
    def __compute_phi(self) -> np.ndarray:
        try:
            sin_emission_angle = np.sin(np.radians(self.__emission_angle))
            sin_solar_zenith_angle = \
                np.sin(np.radians(self.__incidence_angle))
            cos_phase_angle = \
                self.__compute_angle_cosine(self.__phase_angle)
            with np.errstate(divide='ignore', invalid='ignore'):
                tmp_arg = np.true_divide(
                    cos_phase_angle - self.mu * self.mu0,
                    sin_emission_angle * sin_solar_zenith_angle)
                tmp_arg[~np.isfinite(tmp_arg)] = -1
                d_phi = np.arccos(np.clip(tmp_arg, -1, 1))
            return self.phi0 + 180 - np.degrees(d_phi)
        except ValueError as ve:
            raise ValueError('The input angles must be the same shape.') from ve

    @staticmethod
    def __compute_angle_cosine(angle: np.ndarray) -> np.ndarray:
        return np.cos(np.radians(angle))

    @property
    def incidence(self) -> np.ndarray:
        """Get the input incidence (solar zenith) angle [degrees].

        """
        return self.__incidence_angle

    @property
    def emission(self) -> np.ndarray:
        """Get the input emission angle [degrees].

        """
        return self.__emission_angle

    @property
    def phase(self) -> np.ndarray:
        """Get the input phase angle [degrees].

        """
        return self.__phase_angle

    @property
    def mu0(self) -> np.ndarray:
        r"""Get :math:`\mu_0` where :math:`\mu_0` is the cosine of the input
        incidence angle.

        Notes
        -----
        Each element in this variable is named :code:`UMU0` in DISORT.

        """
        return self.__mu0

    @property
    def mu(self) -> np.ndarray:
        r"""Get :math:`\mu` where :math:`\mu` is the cosine of the input
        emission angle.

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


# TODO: These function names seem bad. raise_value_error also raises other error
# TODO: Some wavelengths are impossibly small... filter those out
class Spectral:
    """A data structure that contains spectral info required by DISORT.

    Spectral accepts the short and long wavelengths from an observation and
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
            Raised if any of the inputs are not np.ndarrays, or if the arrays
            contain non-numeric values.
        ValueError
            Raised if either of the input spectral arrays contain unphysical
            values (non-positive or infinite wavelengths), if they're not the
            same shape, or if any values in short_wavelength are larger than the
            corresponding values in long_wavelength.

        """
        self.__short_wavelength = short_wavelength
        self.__long_wavelength = long_wavelength

        self.__raise_error_if_wavelengths_are_unphysical()

        self.__high_wavenumber = self.__calculate_high_wavenumber()
        self.__low_wavenumber = self.__calculate_low_wavenumber()

    def __raise_error_if_wavelengths_are_unphysical(self) -> None:
        self.__raise_value_error_if_either_wavelength_contains_nans()
        self.__raise_value_error_if_short_wavelength_contains_negative_values()
        self.__raise_value_error_if_long_wavelength_contains_inf()
        self.__raise_value_error_if_long_wavelength_is_not_larger()

    def __raise_value_error_if_either_wavelength_contains_nans(self) -> None:
        try:
            if np.any(np.isnan(self.__short_wavelength)):
                message = 'short_wavelength contains NaNs.'
                raise ValueError(message)
        except TypeError as te:
            message = 'short_wavelength must be a numpy.ndarray of numeric ' \
                      'values.'
            raise TypeError(message) from te
        try:
            if np.any(np.isnan(self.__long_wavelength)):
                message = 'long_wavelength contains NaNs.'
                raise ValueError(message)
        except TypeError as te:
            message = 'long_wavelength must be a numpy.ndarray of numeric ' \
                      'values.'
            raise TypeError(message) from te

    def __raise_value_error_if_short_wavelength_contains_negative_values(self) \
            -> None:
        if np.any(self.__short_wavelength <= 0):
            message = 'short_wavelength contains non-positive values.'
            raise ValueError(message)

    def __raise_value_error_if_long_wavelength_contains_inf(self) -> None:
        if np.any(np.isinf(self.__long_wavelength)):
            message = 'long_wavelength contains infinite values.'
            raise ValueError(message)

    def __raise_value_error_if_long_wavelength_is_not_larger(self) -> None:
        try:
            if np.any(self.__short_wavelength >= self.__long_wavelength):
                message = 'Some values in long_wavelength are not larger ' \
                          'than the corresponding values in short_wavelength.'
                raise ValueError(message)
        except ValueError as ve:
            message = 'The spectral arrays must have the same shape.'
            raise ValueError(message) from ve

    def __calculate_high_wavenumber(self) -> np.ndarray:
        return self.__convert_wavelength_to_wavenumber(
            self.__short_wavelength, 'short_wavelength')

    def __calculate_low_wavenumber(self) -> np.ndarray:
        return self.__convert_wavelength_to_wavenumber(
            self.__long_wavelength, 'long_wavelength')

    @staticmethod
    def __convert_wavelength_to_wavenumber(wavelength: np.ndarray,
                                           wavelength_name: str) -> np.ndarray:
        with np.errstate(divide='raise'):
            try:
                return 1 / (wavelength * 10 ** -4)
            except FloatingPointError as fpe:
                message = f'At least one value in {wavelength_name} is too' \
                          f'small to perform calculations!'
                raise ValueError(message) from fpe

    @property
    def short_wavelength(self) -> np.ndarray:
        """Get the input short wavelength [microns].

        """
        return self.__short_wavelength

    @property
    def long_wavelength(self) -> np.ndarray:
        """Get the input long wavelength [microns].

        """
        return self.__long_wavelength

    @property
    def high_wavenumber(self) -> np.ndarray:
        """Get the high wavenumbers [1/cm]---the wavenumbers corresponding to
        short_wavelength.

        Notes
        -----
        In DISORT, this variable is named :code:`WVNMHI`. It is only needed by
        DISORT if :code:`thermal_emission==True` (defined in
        :class:`radiation.ThermalEmission`), or if DISORT is run multiple times
        and BDREF is spectrally dependent.

        """
        return self.__high_wavenumber

    @property
    def low_wavenumber(self) -> np.ndarray:
        """Get the low wavenumbers [1/cm]---the wavenumbers associated with
        long_wavelength.

        Notes
        -----
        In DISORT, this variable is named :code:`WVNMLO`. It is only needed by
        DISORT if :code:`thermal_emission==True` (defined in
        :class:`radiation.ThermalEmission`), or if DISORT is run multiple times
        and BDREF is spectrally dependent.

        """
        return self.__low_wavenumber
