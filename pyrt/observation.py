"""This module provides a number of classes (and functions to help create those
classes) useful for computing quantities required by DISORT that are commonly
found in an observation.

"""
# TODO: In Python3.10, they should disable evaluation of type hints. Currently,
#  ArrayLike evaluates to Sequence[Sequence[Sequence[...]. After that, I can
#  remove the __future__ import. See also
#  https://stackoverflow.com/questions/67473396/shorten-display-format-of-python-type-annotations-in-sphinx
# TODO: warn if emission > 90 if orbiter?
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


class Angles:
    r"""A data structure that contains angles required by DISORT.

    It accepts both the incidence and azimuth angles of the incident beam
    as well as emission and azimuth angles from an observation. It holds
    these values and computes both :math:`\mu_0` and :math:`\mu` from these
    angles.

    This class can compute all angular quantities required by DISORT at once,
    even multiple observations. Both :code:`incidence` and :code:`beam_azimuth`
    must have the same shape (the "beam measurement shape"), representing
    measurements at different incident beams. Both :code:`emission` and
    :code:`azimuth` must have that same shape with an additional axis at the
    end, representing the angular grid at the corresponding incident beams. The
    length of this final axis can be different for both of these inputs.
    See the notes section for discussion of multiple cases.

    Parameters
    ----------
    incidence
        Incidence (solar zenith) angles [degrees]. All values must be between 0
        and 90.
    beam_azimuth
        Azimuth angles of the incident beam [degrees]. All values must be
        between 0 and 360.
    emission
        Emission (emergence) angles [degrees]. All values must be between 0 and
        180.
    azimuth
        Azimuth angles [degrees]. All values must be between 0 and 360.

    Raises
    ------
    TypeError
        Raised if the input arrays contain nonnumerical values.
    ValueError
        Raised if any values of the input arrays are outside their
        mathematically valid range, or if the inputs do not have the same beam
        measurement shape.

    See Also
    --------
    phase_to_angles: Create instances of this class if phase angles are known
                     but azimuth angles are unknown.
    sky_image: Create instances of this class from a single sky image.

    Notes
    -----
    In the case of a rover taking a single image from the surface along an MxN
    emission and azimuth angle grid, the image is taken with a single incident
    beam. Thus, :code:`incidence` and :code:`beam_azimuth` should be scalars (or
    single valued arrays) whereas :code:`emission` should have shape (1, M) and
    :code:`azimuth` should have shape (1, N).

    In the case of an orbiter taking an MxN image, each is pixel is illuminated
    by a different incident beam. In this case :code:`incidence` and
    :code:`beam_azimuth` should have shape (M, N) whereas :code:`emission` and
    :code:`azimuth` should have shape (M, N, 1).

    DISORT wants a float for :code:`UMU0` and :code:`PHI0`; it wants a 1D array
    for :code:`UMU` and :code:`PHI`. Selecting the proper indices from the
    beam measurement axes is necessary to get these data types.

    Examples
    --------
    Import the relevant modules

    >>> import numpy as np
    >>> from pyrt.observation import Angles

    Instantiate this class for a (3, 5) sky image taken along an emission and
    azimuth angle grid, with a single incident beam.

    >>> incidence = 30
    >>> beam_azimuth = 40
    >>> emission = np.linspace(30, 60, num=3)[np.newaxis, :]
    >>> azimuth = np.linspace(20, 50, num=5)[np.newaxis, :]
    >>> Angles(incidence, beam_azimuth, emission, azimuth)
    Angles:
       mu0 = [0.8660254]
       phi0 = [40]
       mu = [[0.8660254  0.70710678 0.5       ]]
       phi = [[20.  27.5 35.  42.5 50. ]]

    Instantiate this class for a sequence of 50 (3, 5) images taken from a fixed
    position over a period time where the incidence angle and beam azimuth angle
    varied from image to image.

    >>> incidence = np.linspace(30, 35, num=50)
    >>> beam_azimuth = np.linspace(40, 50, num=50)
    >>> emission = np.broadcast_to(np.linspace(30, 60, num=3), (50, 3))
    >>> azimuth = np.broadcast_to(np.linspace(20, 50, num=5), (50, 5))
    >>> angles = Angles(incidence, beam_azimuth, emission, azimuth)
    >>> angles.mu0.shape, angles.phi0.shape, angles.mu.shape, angles.phi.shape
    ((50,), (50,), (50, 3), (50, 5))

    Instantiate this class for a (40, 50) image where each pixel was illuminated
    by a unique incident beam.

    >>> ang = np.outer(np.linspace(1, 2, num=40), np.linspace(10, 40, num=50))
    >>> angles = Angles(ang, ang, ang[..., np.newaxis], ang[..., np.newaxis])
    >>> angles.mu0.shape, angles.phi0.shape, angles.mu.shape, angles.phi.shape,
    ((40, 50), (40, 50), (40, 50, 1), (40, 50, 1))

    """
    def __init__(self, incidence: ArrayLike, beam_azimuth: ArrayLike,
                 emission: ArrayLike, azimuth: ArrayLike):
        self._bundle = \
            _RoverAngleBundle(incidence, beam_azimuth, emission, azimuth)

        self._mu0 = self._compute_mu0()
        self._mu = self._compute_mu()

    def __str__(self):
        return f'Angles:\n' \
               f'   mu0 = {self.mu0}\n' \
               f'   phi0 = {self.phi0}\n' \
               f'   mu = {self.mu}\n' \
               f'   phi = {self.phi}'

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
def make_azimuth(incidence: ArrayLike, emission: ArrayLike,
                 phase: ArrayLike) -> np.ndarray:
    r"""Construct azimuth angles from a set of incidence, emission, and phase
    angles.

    Parameters
    ----------
    incidence
        Incidence (solar zenith) angles [degrees]. All values must be between 0
        and 90.
    emission
        Emission (emergence) angles [degrees]. All values must be between 0 and
        180.
    phase
        Phase angles [degrees]. All values must be between 0 and 180.

    Raises
    ------
    TypeError
        Raised if any values in the input arrays are nonnumerical.
    ValueError
        Raised if any values of the input arrays are outside their
        mathematically valid range or if the input arrays do not have the same
        shapes.

    Examples
    --------
    Create the azimuth angles from an assortment of angles.

    >>> import numpy as np
    >>> from pyrt.observation import make_azimuth
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


def phase_to_angles(incidence: ArrayLike, emission: ArrayLike,
                    phase: ArrayLike) -> Angles:
    r"""Construct an instance of Angles in the case where phase angles are
    known but azimuth angles are unknown.

    Parameters
    ----------
    incidence
        Incidence (solar zenith) angles [degrees]. All values must be between 0
        and 90.
    emission
        Emission (emergence) angles [degrees]. All values must be between 0 and
        180.
    phase
        Phase angles [degrees]. All values must be between 0 and 180.

    Raises
    ------
    TypeError
        Raised if any values in the input arrays are nonnumerical.
    ValueError
        Raised if any values of the input arrays are outside their
        mathematically valid range or if the input arrays do not have the same
        shapes.

    Examples
    --------
    For a random assortment of input angles:

    >>> import numpy as np
    >>> from pyrt.observation import phase_to_angles
    >>> incidence = [20, 30, 40]
    >>> emission = [30, 40, 50]
    >>> phase = [25, 30, 35]
    >>> phase_to_angles(incidence, emission, phase)
    Angles:
       mu0 = [0.93969262 0.8660254  0.76604444]
       phi0 = [0. 0. 0.]
       mu = [[0.8660254 ]
     [0.76604444]
     [0.64278761]]
       phi = [[122.74921226]
     [129.08074256]
     [131.57329276]]

    """
    bundle = _OrbiterAngleBundle(incidence, emission, phase)
    phi = make_azimuth(bundle.incidence, bundle.emission, bundle.phase)
    phi0 = np.zeros(bundle.phase.shape)
    return Angles(bundle.incidence, phi0,
                  bundle.emission[..., np.newaxis], phi[..., np.newaxis])


def sky_image(incidence: float, beam_azimuth: float, emission: ArrayLike,
              azimuth: ArrayLike) -> Angles:
    """Create an instance of Angles from a typical sky image---that is, a single
    incidence and beam azimuth angle are known, and the observational geometry
    defines a 1D array of emission and azimuth angles.

    Parameters
    ----------
    incidence
        Incidence (solar zenith) angle [degrees]. Value must be between 0 and
        90.
    beam_azimuth
        Azimuth angle of the incident beam [degrees]. Value must be between 0
        and 360.
    emission
        Emission (emergence) angle [degrees]. All values must be between 0 and
        180.
    azimuth
        Azimuthal angle [degrees]. All values must be between 0 and 360.

    Raises
    ------
    TypeError
        Raised if any values in the input arrays are nonnumerical.
    ValueError
        Raised if any values of the input arrays are outside their
        mathematically valid range, or if the inputs do not have the same beam
        measurement shape.

    Examples
    --------
    Create an instance of Angles for a (3, 5) sky image with a single incident
    beam.

    >>> import numpy as np
    >>> from pyrt.observation import sky_image
    >>> incidence = 30
    >>> beam_azimuth = 40
    >>> emission = np.linspace(30, 60, num=3)
    >>> azimuth = np.linspace(20, 50, num=5)
    >>> sky_image(incidence, beam_azimuth, emission, azimuth)
    Angles:
       mu0 = [0.8660254]
       phi0 = [40]
       mu = [[0.8660254  0.70710678 0.5       ]]
       phi = [[20.  27.5 35.  42.5 50. ]]

    """
    incidence = _IncidenceAngles(incidence)
    beam_azimuth = _AzimuthAngles(beam_azimuth)
    emission = _EmissionAngles(emission)[np.newaxis, ...]
    azimuth = _AzimuthAngles(azimuth)[np.newaxis, ...]

    return Angles(incidence, beam_azimuth, emission, azimuth)


class Spectral:
    """A data structure that contains spectral information required by DISORT.

    It accepts the short and long wavelengths from an observation and computes
    their corresponding wavenumbers. It can accommodate arrays of any shape as
    long as they have the same shape.

    Parameters
    ----------
    short_wavelength
        The short wavelength [microns] of each spectral bin.
    long_wavelength
        The long wavelength [microns] of each spectral bin.

    Raises
    ------
    TypeError
        Raised if any values in the input arrays are nonnumerical.
    ValueError
        Raised if either of the input arrays are not the same shape, if any
        values in ``short_wavelength`` are not larger than the corresponding
        values in ``long_wavelength``, or if either of the input arrays
        contain values outside of 0.1 to 50 microns (I assume this is the
        valid range to do retrievals).

    Notes
    -----
    If you do not plan to use thermal emission, there is probably little benefit
    to making an instance of this class. See
    :py:class:`~pyrt.radiation.ThermalEmission` for a discussion on thermal
    radiation in DISORT.

    See Also
    --------
    constant_width: Create instances of this class if the wavelengths are
                    equally spaced.

    Examples
    --------
    Import the relevant modules

    >>> import numpy as np
    >>> from pyrt.observation import Spectral

    Instantiate this class for a simple set of wavelengths.

    >>> short = np.array([9, 10])
    >>> long = short + 1
    >>> Spectral(short, long)
    Spectral:
       low_wavenumber = [1000.          909.09090909]
       high_wavenumber = [1111.11111111 1000.        ]

    Instantiate this class for measurements taken at 1 to 30 microns in 1 micron
    increments, with each channel having a 50 nm spectral width.

    >>> center = np.linspace(1, 30, num=30)
    >>> half_width = 0.05 / 2
    >>> wavelengths = Spectral(center - half_width, center + half_width)
    >>> wavelengths.low_wavenumber.shape
    (30,)

    Instantiate this class for an image of shape (50, 60), where each pixel
    contains the same 20 wavelengths---wavelengths that span 1 to 20 microns in
    1 micron increments with each channel having a 50 nm spectral width. Note
    that this is not memory efficient, as each pixel along the "measurement
    shape" contains the same information, but it may be useful for keeping
    as many arrays as possible equally shaped.

    >>> center = np.linspace(1, 20, num=20)
    >>> half_width = 0.05 / 2
    >>> wav_grid = np.broadcast_to(center, (50, 60, 20))
    >>> wavelengths = Spectral(wav_grid - half_width, wav_grid + half_width)
    >>> wavelengths.low_wavenumber.shape
    (50, 60, 20)
    >>> np.array_equal(wavelengths.low_wavenumber[0, 0, :],
    ...                wavelengths.low_wavenumber[9, 17, :])
    True

    """
    def __init__(self, short_wavelength: ArrayLike,
                 long_wavelength: ArrayLike):

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
        :py:attr:`~pyrt.radiation.ThermalEmission.thermal_emission` is set to
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
        :py:attr:`~pyrt.radiation.ThermalEmission.thermal_emission` is set to
        :code:`True`.

        """
        return self._low_wavenumber


def constant_width(center_wavelength: ArrayLike, width: float) -> Spectral:
    """Create an instance of Spectral assuming the wavelengths all have a
    constant spectral width.

    Parameters
    ----------
    center_wavelength
        The center wavelength [microns] of each spectral bin.
    width
        The width of each spectral bin.

    Raises
    ------
    TypeError
        Raised if any values in the input arrays are nonnumerical.
    ValueError
        Raised if the inputs contain values outside of 0.1 to 50 microns (I
        assume this is the valid range to do retrievals).

    Examples
    --------
    Create an instance of Spectral for an observation taken at 1 to 3 microns
    in 1 micron increments, with each channel having a 50 nm spectral width.

    >>> import numpy as np
    >>> from pyrt.observation import constant_width
    >>> center = [1, 2, 3]
    >>> width = 0.05
    >>> constant_width(center, width)
    Spectral:
       low_wavenumber = [9756.09756098 4938.27160494 3305.78512397]
       high_wavenumber = [10256.41025641  5063.29113924  3361.34453782]

    """
    half = width / 2
    center = _Wavelength(center_wavelength, 'center_wavelength')
    return Spectral(center - half, center + half)


class _Angles(np.ndarray):
    """An abstract base class for designating that an input represents angles.

    Parameters
    ----------
    array
        Any array of angles.
    name
        The name of the angular array.
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

    def __new__(cls, array: ArrayLike, name: str, low: float, high: float):
        obj = np.asarray(array).view(cls)
        obj.name = name
        obj.low = low
        obj.high = high
        obj = cls.__add_dimension_if_array_is_shapeless(obj)
        cls.__raise_value_error_if_array_is_not_in_input_range(obj)
        return obj

    def __array_finalize__(self, obj: np.ndarray):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)
        self.low = getattr(obj, 'low', None)
        self.high = getattr(obj, 'high', None)

    @staticmethod
    def __add_dimension_if_array_is_shapeless(obj):
        if obj.shape == ():
            obj = obj[None]
        return obj

    @staticmethod
    def __raise_value_error_if_array_is_not_in_input_range(obj) -> None:
        if not np.all(((obj.low <= obj) & (obj <= obj.high))):
            message = f'All values in {obj.name} must be between ' \
                      f'{obj.low} and {obj.high} degrees.'
            raise ValueError(message)

    def sin(self) -> _Angles:
        """Compute the sine of the input angles.

        """
        return np.sin(np.radians(self))

    def cos(self) -> _Angles:
        """Compute the cosine of the input angles.

        """
        return np.cos(np.radians(self))

    def to_ndarray(self) -> np.ndarray:
        """Turn this object into a generic ndarray.

        """
        return np.array(self)


class _IncidenceAngles(_Angles):
    """Designate that an input array represents incidence (solar zenith) angles.

    Parameters
    ----------
    array
        Any array of incidence angles [degrees]. Must be between 0 and 90.

    Raises
    ------
    TypeError
        Raised if any values in the input array are nonnumerical.
    ValueError
        Raised if any values in the input array are not between 0 and 90.

    """
    def __new__(cls, array: ArrayLike):
        obj = super().__new__(cls, array, 'incidence', 0, 90)
        return obj


class _EmissionAngles(_Angles):
    """Designate that an input array represents emission (emergence) angles.

    Parameters
    ----------
    array
        Any array of emission angles [degrees]. Must be between 0 and 180.

    Raises
    ------
    TypeError
        Raised if any values in the input array are nonnumerical.
    ValueError
        Raised if any values in the input array are not between 0 and 180.

    """
    def __new__(cls, array: ArrayLike):
        obj = super().__new__(cls, array, 'emission', 0, 180)
        return obj


class _PhaseAngles(_Angles):
    """Designate that an input array represents phase angles.

    Parameters
    ----------
    array
        Any array of phase angles [degrees]. Must be between 0 and 180.

    Raises
    ------
    TypeError
        Raised if any values in the input array are nonnumerical.
    ValueError
        Raised if any values in the input array are not between 0 and 180.

    """
    def __new__(cls, array: ArrayLike):
        obj = super().__new__(cls, array, 'phase', 0, 180)
        return obj


class _AzimuthAngles(_Angles):
    """Designate that an input array represents azimuth angles.

    Parameters
    ----------
    array
        Any array of angles [degrees]. Must be between 0 and 360.

    Raises
    ------
    TypeError
        Raised if any values in the input array are nonnumerical.
    ValueError
        Raised if any values in the input array are not between 0 and 360.

    """
    def __new__(cls, array: ArrayLike):
        obj = super().__new__(cls, array, 'azimuth', 0, 360)
        return obj


class _OrbiterAngleBundle:
    """Designate a collection of angles that represent those found in an
    orbiter observation as being linked.

    Parameters
    ----------
    incidence
        Incidence (solar zenith) angles [degrees]. All values must be between 0
        and 90.
    emission
        Emission (emergence) angles [degrees]. All values must be between 0 and
        180.
    phase
        Phase angles [degrees]. All values must be between 0 and 180.

    Raises
    ------
    TypeError
        Raised if any values in the input arrays are nonnumerical.
    ValueError
        Raised if any values of the input arrays are outside their
        mathematically valid range, or if the input arrays do not have the same
        shapes.

    """
    def __init__(self, incidence: ArrayLike, emission: ArrayLike,
                 phase: ArrayLike):
        self.incidence = _IncidenceAngles(incidence)
        self.emission = _EmissionAngles(emission)
        self.phase = _PhaseAngles(phase)

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
        and 90.
    emission
        Emission (emergence) angles [degrees]. All values must be between 0 and
        180.
    azimuth
        Azimuth angles [degrees]. All values must be between 0 and 360.
    beam_azimuth
        Azimuth angles of the incident beam [degrees]. All values must be
        between 0 and 360.

    Raises
    ------
    TypeError
        Raised if any values in the input arrays are not numeric.
    ValueError
        Raised if any values of the input arrays are outside their
        mathematically valid range, or if the inputs do not have the same
        observation shape.

    """
    def __init__(self, incidence: ArrayLike, beam_azimuth: ArrayLike,
                 emission: ArrayLike, azimuth: ArrayLike):
        self.incidence = _IncidenceAngles(incidence)
        self.beam_azimuth = _AzimuthAngles(beam_azimuth)
        self.emission = _EmissionAngles(emission)
        self.azimuth = _AzimuthAngles(azimuth)

        self.__raise_value_error_if_beam_measurement_dimensions_do_not_match()

    def __raise_value_error_if_beam_measurement_dimensions_do_not_match(self) \
            -> None:
        if not (self.incidence.shape == self.beam_azimuth.shape ==
                self.emission.shape[:-1] == self.azimuth.shape[: -1]):
            message = f'The incident beam measurement dimension of the ' \
                      f'arrays must match. They are {self.incidence.shape}, ' \
                      f'{self.beam_azimuth.shape},' \
                      f'{self.emission.shape[:-1]}, and ' \
                      f'{self.azimuth.shape[:-1]}.'
            raise ValueError(message)


class _Wavelength(np.ndarray):
    """Designate that an input array represents wavelengths.

    Parameters
    ----------
    array
        Array of wavelengths. Must be between 0.1 and 50 microns.
    name
        Name of the wavelength.

    Raises
    ------
    ValueError
        Raised if any values in :code`array` are not between 0.1 and 50
        microns (I assume this is the valid range to do retrievals).

    """
    def __new__(cls, array: ArrayLike, name: str):
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
        """Convert this object into an array of its corresponding wavenumbers.

        """
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
        microns (I assume this is the valid range to do retrievals), or if
        :code:`long_wavelength` are not larger than the corresponding values in
        :code:`short_wavelength`.

    """
    def __init__(self, short_wavelength: ArrayLike,
                 long_wavelength: ArrayLike):
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
