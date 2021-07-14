"""This module provides a number of classes (and functions to help create those
classes) useful for computing quantities required by DISORT that are commonly
found in an observation.

"""
import warnings
import numpy as np


class Angles:
    r"""A data structure that contains angles required by DISORT.

    It accepts both the incidence and azimuth angles of the incident beam
    as well as emission and azimuth angles from an observation. It holds
    these values and computes both :math:`\mu_0` and :math:`\mu` from these
    angles.

    This class can compute all angular quantities required by DISORT at once,
    even multiple observations. Both :code:`incidence` and :code:`beam_azimuth`
    must have the same shape (the "observation shape"). Both :code:`emission`
    and :code:`azimuth` must have that same shape with an additional axis at the
    end, though the length of that axis can be different for both of these
    inputs.

    Parameters
    ----------
    incidence
        Incidence (solar zenith) angles [degrees]. All values must be between 0
        and 180 degrees.
    emission
        Emission (emergence) angles [degrees]. All values must be between 0 and
        180 degrees.
    azimuth
        Azimuth angles [degrees]. All values must be between 0 and 360 degrees.
    beam_azimuth
        Azimuth angles of the incident beam [degrees]. All values must be
        between 0 and 360 degrees.

    Raises
    ------
    TypeError
        Raised if any of the inputs are not a numpy.ndarray.
    ValueError
        Raised if any of the input arrays contain values outside of their
        mathematically valid range, or if the input arrays do not have the
        same observation shape.

    Warnings
    --------
    UserWarning
        Issued if any values in :code:`incidence` are greater than 90 degrees.

    See Also
    --------
    phase_to_angles: Create instances of this class if the phase angles are
                     known, but the azimuth angles are unknown.
    sky_image: Create instances of this class from a single sky image.

    Notes
    -----
    DISORT wants a float for :code:`UMU0` and :code:`PHI0`; it wants a 1D array
    for :code:`UMU` and :code:`PHI`. Selecting the proper indices from the
    observation dimension(s) is necessary to get these data types.

    Examples
    --------
    Instantiate this class for a (3, 5) sky image with a single incident beam.
    This could represent a single image taken from a rover.

    >>> import numpy as np
    >>> from pyRT_DISORT.observation import Angles
    >>> incidence_ang = np.array([30])
    >>> beam_azimuth = np.array([40])
    >>> emission_ang = np.linspace(30, 60, num=3)[np.newaxis, :]
    >>> azimuth_ang = np.linspace(20, 50, num=5)[np.newaxis, :]
    >>> angles = Angles(incidence_ang, emission_ang, azimuth_ang, beam_azimuth)
    >>> print(angles)
    Angles:
       mu = [[0.8660254  0.70710678 0.5       ]]
       mu0 = [0.8660254]
       phi = [[20.  27.5 35.  42.5 50. ]]
       phi0 = [40]

    Instantiate this class for a sequence of 50 images at a fixed position over
    a period time where the incidence angle and beam azimuth angle varies from
    image to image. This could represent a movie taken from a rover.

    >>> import numpy as np
    >>> from pyRT_DISORT.observation import Angles
    >>> incidence_ang = np.linspace(30, 35, num=50)
    >>> beam_azimuth = np.linspace(40, 50, num=50)
    >>> emission_ang = np.broadcast_to(np.linspace(30, 60, num=3), (50, 3))
    >>> azimuth_ang = np.broadcast_to(np.linspace(20, 50, num=5), (50, 5))
    >>> angles = Angles(incidence_ang, emission_ang, azimuth_ang, beam_azimuth)
    >>> print(angles.mu0.shape, angles.mu.shape, angles.phi.shape)
    (50,) (50, 3) (50, 5)

    Instantiate this class for a (40, 50) image where each pixel has its own
    set of angles. This could represent an image taken from orbit.

    >>> import numpy as np
    >>> from pyRT_DISORT.observation import Angles
    >>> ang = np.outer(np.linspace(1, 2, num=40), np.linspace(10, 40, num=50))
    >>> expanded_ang = np.expand_dims(ang, -1)
    >>> angles = Angles(ang, expanded_ang, expanded_ang, ang)
    >>> print(angles.mu0.shape, angles.mu.shape, angles.phi.shape)
    (40, 50) (40, 50, 1) (40, 50, 1)

    """
    def __init__(self, incidence: np.ndarray, emission: np.ndarray,
                 azimuth: np.ndarray, beam_azimuth: np.ndarray) -> None:
        self._incidence = _Angle(incidence, 'incidence', 0, 180)
        self._emission = _Angle(emission, 'emission', 0, 180)
        self._azimuth = _Angle(azimuth, 'azimuth', 0, 360)
        self._azimuth0 = _Angle(beam_azimuth, 'beam_azimuth', 0, 360)

        self._raise_value_error_if_inputs_have_different_obs_shapes()
        self._warn_if_incidence_angle_is_greater_than_90()

        self._mu0 = self.__compute_mu0()
        self._mu = self.__compute_mu()

    def __str__(self) -> str:
        return f'Angles:\n' \
               f'   mu = {self.mu}\n' \
               f'   mu0 = {self.mu0}\n' \
               f'   phi = {self.phi}\n' \
               f'   phi0 = {self.phi0}'

    def _raise_value_error_if_inputs_have_different_obs_shapes(self) -> None:
        self._raise_value_error_if_observation_dimensions_do_not_match()

    def _raise_value_error_if_observation_dimensions_do_not_match(self) \
            -> None:
        if not (self._incidence.val.shape == self._emission.val.shape[:-1] ==
                self._azimuth0.val.shape == self._azimuth.val.shape[: -1]):
            print(self._incidence.val.shape, self._azimuth0.val.shape)
            message = 'The pixel dimensions do not match.'
            raise ValueError(message)

    def _warn_if_incidence_angle_is_greater_than_90(self) -> None:
        if np.any(self._incidence.val > 90):
            message = 'Some values in incidence are greater than 90 degrees.'
            warnings.warn(message)

    def __compute_mu0(self) -> np.ndarray:
        return self._incidence.cosine()

    def __compute_mu(self) -> np.ndarray:
        return self._emission.cosine()

    @property
    def incidence(self) -> np.ndarray:
        """Get the input incidence angles [degrees].

        """
        return self._incidence.val

    @property
    def emission(self) -> np.ndarray:
        """Get the input emission angles [degrees].

        """
        return self._emission.val

    @property
    def mu0(self) -> np.ndarray:
        r"""Get :math:`\mu_0`---the cosine of ``incidence``.

        Notes
        -----
        Each element along the observation dimension(s) is named :code:`UMU0` in
        DISORT.

        """
        return self._mu0

    @property
    def mu(self) -> np.ndarray:
        r"""Get :math:`\mu`---the cosine of ``emission``.

        Notes
        -----
        Each element along the observation dimension(s) is named :code:`UMU` in
        DISORT.

        """
        return self._mu

    @property
    def phi0(self) -> np.ndarray:
        r"""Get :math:`\phi_0`---the azimuth angles of the incident beam
        [degrees]. This is the same as the input to :code:`beam_azimuth`.

        Notes
        -----
        Each element along the observation dimension(s) is named :code:`PHI0` in
        DISORT.

        """
        return self._azimuth0.val

    @property
    def phi(self) -> np.ndarray:
        r"""Get :math:`\phi`---the azimuth angles [degrees]. This is the same as
        the input to :code:`azimuth`.

        Notes
        -----
        Each element along the observation dimension(s) is named :code:`PHI` in
        DISORT.

        """
        return self._azimuth.val


class _Angle:
    """A class to work with angles.

    It accepts a numpy.ndarray of angles and ensures all values in the array
    are within a range of values (inclusive). It provides basic methods for
    manipulating these angles.

    """
    def __init__(self, angle: np.ndarray, name: str, low: float, high: float) \
            -> None:
        """
        Parameters
        ----------
        angle
            Arbitrary array of angles [degrees].
        name
            Name of the angle.
        low
            The lowest value any value in :code:`angle` can be.
        high
            The highest value any value in :code:`angle` can be.

        Raises
        ------
        TypeError
            Raised if :code:`angle` is not a numpy.ndarray.
        ValueError
            Raised if any value in :code:`angle` is outside its allowable range.

        """
        self.__angle = angle
        self.__name = name
        self.__low = low
        self.__high = high

        self.__raise_error_if_angle_is_bad()

    def __getattr__(self, method):
        return getattr(self.__angle, method)

    def __raise_error_if_angle_is_bad(self) -> None:
        self.__raise_type_error_if_angle_is_not_ndarray()
        self.__raise_value_error_if_values_in_angle_are_not_in_range()

    def __raise_type_error_if_angle_is_not_ndarray(self) -> None:
        if not isinstance(self.__angle, np.ndarray):
            message = f'{self.__name} must be a numpy.ndarray.'
            raise TypeError(message)

    def __raise_value_error_if_values_in_angle_are_not_in_range(self) -> None:
        if np.any(self.val < self.__low) or np.any(self.val > self.__high):
            message = f'All values in {self.__name} must be between ' \
                      f'{self.__low} and {self.__high} degrees.'
            raise ValueError(message)

    @property
    def val(self) -> np.ndarray:
        return self.__angle

    def cosine(self) -> np.ndarray:
        return np.cos(np.radians(self.__angle))

    def sine(self) -> np.ndarray:
        return np.sin(np.radians(self.__angle))


# TODO: Is there a cleaner way to compute this?
# TODO: Test shapes match? If not the computation will break
def make_azimuth(incidence: np.ndarray, emission: np.ndarray,
                 phase: np.ndarray) -> np.ndarray:
    r"""Construct azimuth angles from a set of incidence, emission, and phase
    angles.

    Parameters
    ----------
    incidence
        Incidence (solar zenith) angles [degrees]. All values must be between 0
        and 180 degrees.
    emission
        Emission (emergence) angles [degrees]. All values must be between 0 and
        180 degrees.
    phase
        Phase angles [degrees]. All values must be between 0 and 180 degrees.

    Raises
    ------
    TypeError
        Raised if any of the angles are not a numpy.ndarray.
    ValueError
        Raised if any of the input arrays contain values outside of their
        mathematically valid range.

    Notes
    -----
    It would almost always be beneficial for all of the inputs to have the same
    shape, but this is not strictly enforced. In any case, the input arrays
    must have compatible shapes.

    Examples
    --------
    For a random assortment of input angles:

    >>> import numpy as np
    >>> incidence = np.array([20, 30, 40])
    >>> emission = np.array([30, 40, 50])
    >>> phase = np.array([25, 30, 35])
    >>> make_azimuth(incidence, emission, phase)
    array([122.74921226, 129.08074256, 131.57329276])

    """
    incidence = _Angle(incidence, 'incidence', 0, 180)
    emission = _Angle(emission, 'emission', 0, 180)
    phase = _Angle(phase, 'phase', 0, 180)

    with np.errstate(divide='ignore', invalid='ignore'):
        tmp_arg = np.true_divide(
            phase.cosine() - emission.cosine() * incidence.cosine(),
            emission.sine() * incidence.sine())
        tmp_arg[~np.isfinite(tmp_arg)] = -1
        d_phi = np.arccos(np.clip(tmp_arg, -1, 1))

    return 180 - np.degrees(d_phi)


def phase_to_angles(incidence: np.ndarray, emission: np.ndarray,
                    phase: np.ndarray) -> Angles:
    r"""Construct an instance of Angles in the case where the phase angles are
    known (and presumably azimuth angles are unknown).

    Parameters
    ----------
    incidence
        Incidence (solar zenith) angles [degrees]. All values must be between 0
        and 180 degrees.
    emission
        Emission (emergence) angles [degrees]. All values must be between 0 and
        180 degrees.
    phase
        Phase angles [degrees]. All values must be between 0 and 180 degrees.

    Raises
    ------
    TypeError
        Raised if any of the angles are not a numpy.ndarray.
    ValueError
        Raised if any of the input arrays contain values outside of their
        mathematically valid range.

    Warnings
    --------
    UserWarning
        Issued if any values in :code:`incidence` are greater than 90 degrees.

    Notes
    -----
    It would almost always be beneficial for all of the inputs to have the same
    shape, but this is not strictly enforced. In any case, the input arrays
    must have compatible shapes.

    Examples
    --------
    For a random assortment of input angles:

    >>> import numpy as np
    >>> incidence = np.array([20, 30, 40])
    >>> emission = np.array([30, 40, 50])
    >>> phase = np.array([25, 30, 35])
    >>> angles = phase_to_angles(incidence, emission, phase)
    >>> print(angles)
    Angles:
       mu = [[0.8660254 ]
     [0.76604444]
     [0.64278761]]
       mu0 = [0.93969262 0.8660254  0.76604444]
       phi = [[122.74921226]
     [129.08074256]
     [131.57329276]]
       phi0 = [0. 0. 0.]

    """
    phi = make_azimuth(incidence, emission, phase)
    phi0 = np.zeros(phase.shape)
    return Angles(incidence, emission[:, np.newaxis], phi[:, np.newaxis], phi0)


def sky_image(incidence: float, emission: np.ndarray, azimuth: np.ndarray,
              beam_azimuth: float) -> Angles:
    """Create an instance of Angles from a typical sky image---that is, a single
    incidence and beam azimuth angle are known, and the observational geometry
    defines a 1D array of emission and azimuth angles.

    Parameters
    ----------
    incidence
        Incidence (solar zenith) angle [degrees]. Value must be between 0 and
        180 degrees.
    emission
        Emission (emergence) angle [degrees]. All values must be between 0 and
        180 degrees.
    azimuth
        Azimuthal angle [degrees]. All values must be between 0 and 360 degrees.
    beam_azimuth
        Azimuth angle of the incident beam [degrees]. Value must be between 0
        and 360 degrees.

    Raises
    ------
    TypeError
        Raised if any of the angles are not a numpy.ndarray.
    ValueError
        Raised if any of the input arrays contain values outside of their
        mathematically valid range, or if the input arrays do not have the
        same observation shape.

    Warnings
    --------
    UserWarning
        Issued if any values in :code:`incidence` are greater than 90 degrees.

    Examples
    --------
    For a (3, 5) sky image with a single incident beam:

    >>> import numpy as np
    >>> incidence_ang = 30
    >>> beam_az = 40
    >>> emission_ang = np.linspace(30, 60, num=3)
    >>> azimuth_ang = np.linspace(20, 50, num=5)
    >>> angles = sky_image(incidence_ang, emission_ang, azimuth_ang, beam_az)
    >>> print(angles)
    Angles:
       mu = [[0.8660254  0.70710678 0.5       ]]
       mu0 = [0.8660254]
       phi = [[20.  27.5 35.  42.5 50. ]]
       phi0 = [40]

    """
    incidence = np.array([incidence])
    emission = np.expand_dims(emission, axis=0)
    azimuth = np.expand_dims(azimuth, axis=0)
    beam_azimuth = np.array([beam_azimuth])
    return Angles(incidence, emission, azimuth, beam_azimuth)


class Spectral:
    """A data structure that contains spectral info required by DISORT.

    It accepts the short and long wavelength from an observation and computes
    their corresponding wavenumber.

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
    the same shape. If you do not plan to use thermal emission, there is
    probably little benefit making an instance of this class.

    See Also
    --------
    constant_width: Create instances of this class if the wavelengths are
                    equally spaced.

    Examples
    --------
    For an observation taken at 1 to 30 microns, with each channel having a 50
    nm spectral width:

    >>> import numpy as np
    >>> center = np.linspace(1, 30, num=30)
    >>> half_width = 0.025
    >>> wavelengths = Spectral(center - half_width, center + half_width)
    >>> print(wavelengths.low_wavenumber.shape)
    (30,)

    For an image of shape (50, 60) with the same 20 wavelengths in each pixel:

    >>> center = np.linspace(1, 20, num=20)
    >>> half_width = 0.025
    >>> wav_grid = np.broadcast_to(center, (50, 60, 20))
    >>> wavelengths = Spectral(wav_grid - half_width, wav_grid + half_width)
    >>> print(wavelengths.low_wavenumber.shape)
    (50, 60, 20)

    """
    def __init__(self, short_wavelength: np.ndarray,
                 long_wavelength: np.ndarray) -> None:

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
        Each element along the observation dimension(s) is named :code:`WVNMHI`
        in DISORT. It is only needed by DISORT if
        :py:attr:`~radiation.ThermalEmission.thermal_emission` is set to
        :code:`True`.

        """
        return self.__high_wavenumber

    @property
    def low_wavenumber(self) -> np.ndarray:
        r"""Get the low wavenumber [cm :sup:`-1`]---the wavenumber corresponding
        to ``long_wavelength``.

        Notes
        -----
        Each element along the observation dimension(s) is named :code:`WVNMLO`
        in DISORT. It is only needed by DISORT if
        :py:attr:`~radiation.ThermalEmission.thermal_emission` is set to
        :code:`True`.

        """
        return self.__low_wavenumber


class _Wavelength:
    """A data structure to hold on to an array of wavelengths.

    It accepts a numpy.ndarray of wavelengths and ensures all values
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
            Raised if :code`wavelength` is not a numpy.ndarray.
        ValueError
            Raised if any values in :code`wavelength` are not between 0.1 and 50
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
        return self.__wavelength

    def wavelength_to_wavenumber(self) -> np.ndarray:
        return 10 ** 4 / self.__wavelength


# TODO: this is easy to break cause I don't test the inputs
def constant_width(center_wavelength: np.ndarray, width: float) -> Spectral:
    """Create an instance of Spectral assuming the wavelengths all have a
    constant spectral width.

    Parameters
    ----------
    center_wavelength
        The center wavelength [microns] of each spectral bin.
    width
        The spectral width of each spectral bin.

    Raises
    ------
    TypeError
        Raised if :code:`center_wavelength` is not a numpy.ndarray.
    ValueError
        I'll add this later.

    Examples
    --------
    For an observation taken at 1 to 30 microns, with each channel having a 50
    nm spectral width:

    >>> import numpy as np
    >>> center = np.linspace(1, 30, num=30)
    >>> width = 0.05
    >>> wavelengths = constant_width(center, width)
    >>> print(wavelengths.low_wavenumber.shape)
    (30,)

    """
    half = width / 2
    return Spectral(center_wavelength - half, center_wavelength + half)
