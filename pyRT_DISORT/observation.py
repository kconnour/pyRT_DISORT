"""This module provides a number of classes (and functions to help create those
classes) useful for computing quantities required by DISORT that are commonly
found in an observation.

"""
# TODO: __new__ instead of __init__?
# TODO: Only accept ndarrays throughout
# TODO: Angle should have names, like how _Wavelength does
# TODO: It would be better to have a BelowHorizon warning, or something like
#  that, instead of a generic UserWarning
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
        Raised if the inputs are not numpy.ndarrays or if they contain
        nonnumerical values.

    ValueError
        Raised if any values of the input arrays are outside their
        mathematically valid range, or if the inputs do not have the same
        observation shape.

    Warnings
    --------
    UserWarning
        Raised if any values in :code:`incidence` are greater than 90 degrees.

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

    >>> import numpy as np
    >>> from pyRT_DISORT.observation import Angles
    >>> incidence_ang = np.array([30])
    >>> beam_azimuth = np.array([40])
    >>> emission_ang = np.linspace(30, 60, num=3)[np.newaxis, :]
    >>> azimuth_ang = np.linspace(20, 50, num=5)[np.newaxis, :]
    >>> angles = Angles(incidence_ang, emission_ang, azimuth_ang, beam_azimuth)
    >>> angles
    Angles:
       mu = [[0.8660254  0.70710678 0.5       ]]
       mu0 = [0.8660254]
       phi = [[20.  27.5 35.  42.5 50. ]]
       phi0 = [40]

    Instantiate this class for a sequence of 50 images at a fixed position over
    a period time where the incidence angle and beam azimuth angle varies from
    image to image.

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
    set of angles.

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
        self._bundle = \
            _RoverAngleBundle(incidence, emission, azimuth, beam_azimuth)

        self._mu0 = self._compute_mu0()
        self._mu = self._compute_mu()

    def __str__(self):
        return f'Angles:\n' \
               f'   mu = {self.mu}\n' \
               f'   mu0 = {self.mu0}\n' \
               f'   phi = {self.phi}\n' \
               f'   phi0 = {self.phi0}'

    def __repr__(self):
        return self.__str__()

    def _compute_mu0(self) -> np.ndarray:
        return self._bundle.incidence.cos()

    def _compute_mu(self) -> np.ndarray:
        return self._bundle.emission.cos()

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
        [degrees] (the input to :code:`beam_azimuth`).

        Notes
        -----
        Each element along the observation dimension(s) is named :code:`PHI0` in
        DISORT.

        """
        return self._bundle.beam_azimuth

    @property
    def phi(self) -> np.ndarray:
        r"""Get :math:`\phi`---the azimuth angles [degrees] (the input to
        :code:`azimuth`).

        Notes
        -----
        Each element along the observation dimension(s) is named :code:`PHI` in
        DISORT.

        """
        return self._bundle.azimuth


# TODO: Is there a cleaner way to compute this?
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
        Raised if any values in the input arrays are not numeric.
    ValueError
        Raised if any values of the input arrays are outside their
        mathematically valid range, or if the input arrays do not have the same
        shapes.

    Warnings
    --------
    UserWarning
        Raised if any values in :code:`incidence` are greater than 90 degrees.

    Notes
    -----
    It would almost always be beneficial for all of the inputs to have the same
    shape, but this is not strictly enforced. In any case, the input arrays
    must have compatible shapes.

    Examples
    --------
    For a random assortment of input angles:

    >>> import numpy as np
    >>> incidence_angles = np.array([20, 30, 40])
    >>> emission_angles = np.array([30, 40, 50])
    >>> phase_angles = np.array([25, 30, 35])
    >>> make_azimuth(incidence_angles, emission_angles, phase_angles)
    array([122.74921226, 129.08074256, 131.57329276])

    """
    bundle = _OrbiterAngleBundle(incidence, emission, phase)

    with np.errstate(divide='ignore', invalid='ignore'):
        tmp_arg = (np.true_divide(
            bundle.phase.cos() - bundle.emission.cos() * bundle.incidence.cos(),
            bundle.emission.sin() * bundle.incidence.sin())).to_ndarray()
        tmp_arg[~np.isfinite(tmp_arg)] = -1
        d_phi = np.arccos(np.clip(tmp_arg, -1, 1))

    return 180 - np.degrees(d_phi)


def phase_to_angles(incidence: np.ndarray, emission: np.ndarray,
                    phase: np.ndarray) -> Angles:
    r"""Construct an instance of Angles in the case where the phase angles are
    known but azimuth angles are unknown.

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
        Raised if any values in the input arrays are not numeric.
    ValueError
        Raised if any values of the input arrays are outside their
        mathematically valid range, or if the input arrays do not have the same
        shapes.

    Warnings
    --------
    UserWarning
        Issued if any values in :code:`incidence` are greater than 90 degrees.

    Examples
    --------
    For a random assortment of input angles:

    >>> import numpy as np
    >>> incidence_angles = np.array([20, 30, 40])
    >>> emission_angles = np.array([30, 40, 50])
    >>> phase_angles = np.array([25, 30, 35])
    >>> phase_to_angles(incidence_angles, emission_angles, phase_angles)
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
    """Create an instance of :class:`Angles` from a typical sky image---that is,
    a single incidence and beam azimuth angle are known, and the observational
    geometry defines a 1D array of emission and azimuth angles.

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
    >>> sky_image(incidence_ang, emission_ang, azimuth_ang, beam_az)
    Angles:
       mu = [[0.8660254  0.70710678 0.5       ]]
       mu0 = [0.8660254]
       phi = [[20.  27.5 35.  42.5 50. ]]
       phi0 = [40]

    """
    incidence = _IncidenceAngle(np.array([incidence]))
    emission = _EmissionAngle(np.expand_dims(emission, axis=0))
    azimuth = _AzimuthAngle(np.expand_dims(azimuth, axis=0))
    beam_azimuth = _AzimuthAngle(np.array([beam_azimuth]))
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
    Instantiate this class for a simple set of wavelengths.

    >>> import numpy as np
    >>> from pyRT_DISORT.observation import Spectral
    >>> short = np.array([9, 10])
    >>> long = short + 1
    >>> Spectral(short, long)
    Spectral:
       low_wavenumber = [1000.          909.09090909]
       high_wavenumber = [1111.11111111 1000.        ]

    Instantiate this class for an observation taken at 1 to 30 microns, with
    each channel having a 50 nm spectral width.

    >>> import numpy as np
    >>> center = np.linspace(1, 30, num=30)
    >>> half_width = 0.025
    >>> wavelengths = Spectral(center - half_width, center + half_width)
    >>> wavelengths.low_wavenumber.shape
    (30,)

    For an image of shape (50, 60) with the same 20 wavelengths in each pixel:

    >>> center = np.linspace(1, 20, num=20)
    >>> half_width = 0.025
    >>> wav_grid = np.broadcast_to(center, (50, 60, 20))
    >>> wavelengths = Spectral(wav_grid - half_width, wav_grid + half_width)
    >>> wavelengths.low_wavenumber.shape
    (50, 60, 20)

    """
    def __init__(self, short_wavelength: np.ndarray,
                 long_wavelength: np.ndarray) -> None:

        self._bundle = _WavelengthBundle(short_wavelength, long_wavelength)

        self._high_wavenumber = self.__calculate_high_wavenumber()
        self._low_wavenumber = self.__calculate_low_wavenumber()

    def __str__(self):
        return f'Spectral:\n' \
               f'   low_wavenumber = {self._low_wavenumber}\n' \
               f'   high_wavenumber = {self._high_wavenumber}'

    def __repr__(self):
        return self.__str__()

    def __calculate_high_wavenumber(self) -> np.ndarray:
        return self._bundle.short.to_wavenumber()

    def __calculate_low_wavenumber(self) -> np.ndarray:
        return self._bundle.long.to_wavenumber()

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
        return self._high_wavenumber

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
        return self._low_wavenumber


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


class _Angle(np.ndarray):
    """An abstract base class for designating that an input array represents
    angles.

    Parameters
    ----------
    array
        Any array of angles.
    low
        The lowest value any value in the array is allowed to be.
    high
        The highest value any value in the array is allowed to be.

    Raises
    ------
    TypeError
        Raised if any values in the input array are nonnumerical.
    ValueError
        Raised if any values in the input array are outside the input range.

    """

    def __new__(cls, array: np.ndarray, low: float, high: float):
        obj = np.asarray(array).view(cls)
        obj.low = low
        obj.high = high
        cls.__raise_value_error_if_array_is_not_in_range(obj)
        return obj

    def __array_finalize__(self, obj: np.ndarray):
        if obj is None:
            return
        self.low = getattr(obj, 'low', None)
        self.high = getattr(obj, 'high', None)

    @staticmethod
    def __raise_value_error_if_array_is_not_in_range(obj) -> None:
        if not np.all(((obj.low <= obj) & (obj <= obj.high))):
            message = f'All values in the input angles must be between ' \
                      f'{obj.low} and {obj.high} degrees.'
            raise ValueError(message)

    def sin(self) -> np.ndarray:
        """Compute the sine of the input angles.

        """
        return np.sin(np.radians(self))

    def cos(self) -> np.ndarray:
        """Compute the cosine of the input angles.

        """
        return np.cos(np.radians(self))

    def to_ndarray(self) -> np.ndarray:
        """Turn this object into a generic ndarray.

        """
        return np.array(self)


class _IncidenceAngle(_Angle):
    """Designate that an input array represents incidence (solar zenith) angles.

    Parameters
    ----------
    array
        Any array of incidence angles. Must be between 0 and 180 degrees.

    Raises
    ------
    TypeError
        Raised if any values in the input array are not numeric.
    ValueError
        Raised if any values in the input array are not between 0 and 180
        degrees.

    Warnings
    --------
    UserWarning
        Raised if any values in the input array are greater than 90 degrees.

    """
    def __new__(cls, array: np.ndarray):
        obj = super().__new__(cls, array, 0, 180)
        cls.__warn_if_incidence_angle_is_greater_than_90(obj)
        return obj

    @staticmethod
    def __warn_if_incidence_angle_is_greater_than_90(obj: np.ndarray) -> None:
        if np.any(obj > 90):
            message = 'Some values in incidence are greater than 90 degrees.'
            warnings.warn(message)


class _EmissionAngle(_Angle):
    """Designate that an input array represents emission (emergence) angles.

    Parameters
    ----------
    array
        Any array of emission angles. Must be between 0 and 90 degrees.

    Raises
    ------
    TypeError
        Raised if any values in the input array are not numeric.
    ValueError
        Raised if any values in the input array are not between 0 and 90
        degrees.

    """
    def __new__(cls, array: np.ndarray):
        obj = super().__new__(cls, array, 0, 90)
        return obj


class _PhaseAngle(_Angle):
    """Designate that an input array represents phase angles.

    Parameters
    ----------
    array
        Any array of phase angles. Must be between 0 and 180 degrees.

    Raises
    ------
    TypeError
        Raised if any values in the input array are not numeric.
    ValueError
        Raised if any values in the input array are not between 0 and 180
        degrees.

    """
    def __new__(cls, array: np.ndarray):
        obj = super().__new__(cls, array, 0, 180)
        return obj


class _AzimuthAngle(_Angle):
    """Designate that an input array represents azimuth angles.

    Parameters
    ----------
    array
        Any array of angles. Must be between 0 and 360 degrees.

    Raises
    ------
    TypeError
        Raised if any values in the input array are not numeric.
    ValueError
        Raised if any values in the input array are not between 0 and 360
        degrees.

    """
    def __new__(cls, array: np.ndarray):
        obj = super().__new__(cls, array, 0, 360)
        return obj


class _OrbiterAngleBundle:
    """Designate a collection of angles that represent those found in an
    orbiter observation as being linked.

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
        Raised if any values in the input arrays are not numeric.
    ValueError
        Raised if any values of the input arrays are outside their
        mathematically valid range, or if the input arrays do not have the same
        shapes.

    Warnings
    --------
    UserWarning
        Raised if any values in :code:`incidence_angle` are greater than 90
        degrees.

    """
    def __init__(self, incidence: np.ndarray, emission: np.ndarray,
                 phase: np.ndarray):
        self.incidence = _IncidenceAngle(incidence)
        self.emission = _EmissionAngle(emission)
        self.phase = _PhaseAngle(phase)

        self.__raise_value_error_if_angle_shapes_do_not_match()

    def __raise_value_error_if_angle_shapes_do_not_match(self) -> None:
        if not (self.incidence.shape == self.emission.shape ==
                self.phase.shape):
            message = f'The shapes of the arrays must match. They are ' \
                      f'{self.incidence.shape}, {self.emission.shape}, and ' \
                      f'{self.phase.shape}'
            raise ValueError(message)


class _RoverAngleBundle:
    """Designate a collection of angles that represent those found in a rover
    observation as being linked.

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
        Raised if any values in the input arrays are not numeric.
    ValueError
        Raised if any values of the input arrays are outside their
        mathematically valid range, or if the inputs do not have the same
        observation shape.

    Warnings
    --------
    UserWarning
        Raised if any values in :code:`incidence` are greater than 90 degrees.

    """
    def __init__(self, incidence: np.ndarray, emission: np.ndarray,
                 azimuth: np.ndarray, beam_azimuth: np.ndarray):
        self.incidence = _IncidenceAngle(incidence)
        self.emission = _EmissionAngle(emission)
        self.azimuth = _AzimuthAngle(azimuth)
        self.beam_azimuth = _AzimuthAngle(beam_azimuth)

        self.__raise_value_error_if_observation_dimensions_do_not_match()

    def __raise_value_error_if_observation_dimensions_do_not_match(self) \
            -> None:
        if not (self.incidence.shape == self.emission.shape[:-1] ==
                self.azimuth.shape[: -1] == self.beam_azimuth.shape):
            message = f'The pixel dimension of the arrays must match. They ' \
                      f'are {self.incidence.shape}, ' \
                      f'{self.emission.shape[:-1]}, ' \
                      f'{self.azimuth.shape[:-1]}, and ' \
                      f'{self.beam_azimuth.shape}.'
            raise ValueError(message)


class _Wavelength(np.ndarray):
    """Designate that an input array represents wavelengths.

    Parameters
    ----------
    array
        Array of wavelengths [microns].
    name
        Name of the wavelength.

    Raises
    ------
    ValueError
        Raised if any values in :code`array` are not between 0.1 and 50
        microns (I assume this is the valid range to do retrievals).

    """
    def __new__(cls, array: np.ndarray, name: str):
        obj = np.asarray(array).view(cls)
        obj.name = name
        cls.__raise_value_error_if_array_is_not_in_range(obj)
        return obj

    def __array_finalize__(self, obj: np.ndarray):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)

    @staticmethod
    def __raise_value_error_if_array_is_not_in_range(obj) -> None:
        if not np.all((0.1 <= obj) & (obj <= 50)):
            message = f'All values in {obj.name} must be between 0.1 and 50 ' \
                      'microns.'
            raise ValueError(message)

    def to_ndarray(self) -> np.ndarray:
        """Turn this object into a generic ndarray.

        """
        return np.array(self)

    def to_wavenumber(self) -> np.ndarray:
        return np.array(10 ** 4 / self)


class _WavelengthBundle:
    """Designate the short and long wavelength from an observation as being
    linked.

    Parameters
    ----------
    short_wavelength
        Short wavelengths [microns].
    long_wavelength
        Long wavelengths [microns].

    Raises
    ------
    ValueError
        Raised if any values in the input arrays are not between 0.1 and 50
        microns (I assume this is the valid range to do retrievals).

    """
    def __init__(self, short_wavelength: np.ndarray,
                 long_wavelength: np.ndarray):
        self.short = _Wavelength(short_wavelength, 'short_wavelength')
        self.long = _Wavelength(long_wavelength, 'long_wavelength')

        self.__raise_value_error_if_wavelengths_are_not_same_shape()
        self.__raise_value_error_if_long_wavelength_is_not_larger()

    def __raise_value_error_if_wavelengths_are_not_same_shape(self) -> None:
        if self.short.shape != self.long.shape:
            message = 'short_wavelength and long_wavelength must both have ' \
                      'the same shape.'
            raise ValueError(message)

    def __raise_value_error_if_long_wavelength_is_not_larger(self) -> None:
        if np.any(self.short >= self.long):
            message = 'Some values in long_wavelength are not larger ' \
                      'than the corresponding values in short_wavelength.'
            raise ValueError(message)


if __name__ == '__main__':
    a = np.array([10, 11])
    b = np.array([12, 13])
    c = Spectral(a, b)
    print(type(c.low_wavenumber))
    #incidence_ang = 30
    #beam_azimuth = 40
    #emission_ang = np.linspace(30, 60, num=3)[np.newaxis, :]
    #azimuth_ang = np.linspace(20, 50, num=5)[np.newaxis, :]
    #angles = Angles(incidence_ang, emission_ang, azimuth_ang, beam_azimuth)
    #print(angles)
