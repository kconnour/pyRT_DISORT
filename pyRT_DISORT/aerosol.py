"""The :code:`aerosol` module contains structures to make the aerosol properties
required by DISORT.
"""
# TODO: add parameter validators if these classes are ok
import numpy as np
from pyRT_DISORT.eos import _Altitude
from scipy.interpolate import interp2d


class Conrath:
    r"""A structure to compute a Conrath profile(s).

    Conrath creates Conrath profile(s) on an input grid of altitudes given a
    volumetric mixing ratio and Conrath parameters.

    """

    def __init__(self, altitude: np.ndarray, q0: np.ndarray,
                 scale_height: np.ndarray, nu: np.ndarray) -> None:
        r"""
        Parameters
        ----------
        altitude
            The altitude [km] at which to construct a Conrath profile.
        q0
            The surface mixing ratio for each of the Conrath profiles.
        scale_height
            The scale height [km] of each of the Conrath profiles.
        nu
            The nu parameter of each of the Conrath profiles.

        Raises
        ------
        TypeError
            Raised if any of the inputs are not a numpy.ndarray.
        ValueError
            Raised if many things...

        Notes
        -----
        The Conrath profile is defined as

        .. math::

           q(z) = q_0 * e^{\nu(1 - e^{z/H})}

        where :math:`q` is a volumetric mixing ratio, :math:`z` is the altitude,
        :math:`\nu` is the Conrath nu parameter, and :math:`H` is the scale
        height.

        For an MxN array of pixels, :code:`q0`, :code:`scale_height`, and
        :code:`nu` should be of shape MxN. :code:`altitude` should have shape
        ZxMxN, where Z is the number of altitudes. Additionally, the units of
        :code:`altitude` and :code:`scale_height` should be the same.

        """
        self.__altitude = _Altitude(altitude, 'altitude')
        self.__q0 = _ConrathParameter(q0, 'q0', 0, np.inf)
        self.__H = _ConrathParameter(scale_height, 'scale_height', 0, np.inf)
        self.__nu = _ConrathParameter(nu, 'nu', 0, np.inf)

        self.__raise_value_error_if_inputs_have_differing_shapes()

        self.__profile = self.__make_profile()

    def __raise_value_error_if_inputs_have_differing_shapes(self) -> None:
        if not (self.__altitude.shape[1:] == self.__q0.shape ==
                self.__H.shape == self.__nu.shape):
            message = 'All inputs must have the same shape along the pixel ' \
                      'dimension.'
            raise ValueError(message)

    def __make_profile(self) -> np.ndarray:
        altitude_scaling = self.__altitude.val / self.__H.val
        return self.__q0.val * np.exp(self.__nu.val *
                                      (1 - np.exp(altitude_scaling)))

    @property
    def profile(self) -> np.ndarray:
        """Get the Conrath profile(s). This will have the same shape as
        ``altitude``.

        """
        return self.__profile


class _ConrathParameter:
    def __init__(self, parameter: np.ndarray, name: str,
                 low: float, high: float) -> None:
        self.__param = parameter
        self.__name = name
        self.__low = low
        self.__high = high

        self.__raise_error_if_param_is_bad()

    def __raise_error_if_param_is_bad(self) -> None:
        self.__raise_type_error_if_not_ndarray_or_float()
        self.__raise_value_error_if_not_in_range()

    def __raise_type_error_if_not_ndarray_or_float(self) -> None:
        if not isinstance(self.__param, (int, float, np.ndarray)):
            message = f'{self.__name} must be a float or a numpy.ndarray.'
            raise TypeError(message)

    def __raise_value_error_if_not_in_range(self) -> None:
        if not (np.all(self.__low <= self.__param) and
                (np.all(self.__param <= self.__high))):
            message = f'{self.__name} must be between {self.__low} and ' \
                      f'{self.__high}.'
            raise ValueError(message)

    def __getattr__(self, method):
        return getattr(self.val, method)

    @property
    def val(self) -> np.ndarray:
        """Get the value of the input wavelength.

        """
        return np.squeeze(np.array([self.__param]))


class Uniform:
    """A structure to create a uniform profile(s).

    Uniform creates uniform volumetric mixing ratio profile(s) on an input
    grid of altitudes given a set of top and bottom altitudes.

    """

    def __init__(self, altitude: np.ndarray, bottom: np.ndarray,
                 top: np.ndarray) -> None:
        """
        Parameters
        ----------
        altitude
            The altitude *boundaries* at which to construct uniform profile(s).
            These are assumed to be decreasing to keep with DISORT's convention.
        bottom
            The bottom altitudes of each of the profiles.
        top
            The top altitudes of each of the profiles.

        Raises
        ------
        TypeError
            Raised if any inputs are not a numpy.ndarray.
        ValueError
            Raised if a number of things...

        """
        self.__altitude = _Altitude(altitude, 'altitude')
        self.__bottom = _UniformParameter(bottom, 'bottom')
        self.__top = _UniformParameter(top, 'top')

        self.__profile = self.__make_profile()

    def __raise_error_if_inputs_are_bad(self) -> None:
        self.__raise_value_error_if_inputs_have_differing_shapes()
        self.__raise_value_error_if_top_is_not_larger()

    def __raise_value_error_if_inputs_have_differing_shapes(self) -> None:
        if not (self.__altitude.shape[1:] == self.__top.shape ==
                self.__bottom.shape):
            message = 'All inputs must have the same shape along the pixel ' \
                      'dimension.'
            raise ValueError(message)

    def __raise_value_error_if_top_is_not_larger(self) -> None:
        if not np.all(self.__top > self.__bottom):
            message = 'All values in top must be larger than the ' \
                      'corresponding values in bottom.'
            raise ValueError(message)

    def __make_profile(self) -> np.ndarray:
        alt_dif = np.diff(self.__altitude.val, axis=0)
        top_prof = np.clip((self.__top.val - self.__altitude.val[1:]) /
                           np.abs(alt_dif), 0, 1)
        bottom_prof = np.clip((self.__altitude.val[:-1] - self.__bottom.val) /
                              np.abs(alt_dif), 0, 1)
        return top_prof + bottom_prof - 1

    @property
    def profile(self) -> np.ndarray:
        """Get the uniform profile(s). This will have the same shape as
        ``altitude``.

        """
        return self.__profile


# TODO: right now this just copies ConrathParameter. Think of a more general
#  name or make the classes different.
class _UniformParameter:
    def __init__(self, parameter: np.ndarray, name: str) -> None:
        self.__param = parameter
        self.__name = name

        self.__raise_error_if_param_is_bad()

    def __raise_error_if_param_is_bad(self) -> None:
        self.__raise_type_error_if_not_ndarray()

    def __raise_type_error_if_not_ndarray_or_float(self) -> None:
        if not isinstance(self.__param, (float, np.ndarray)):
            message = f'{self.__name} must be a float or numpy.ndarray.'
            raise TypeError(message)

    def __getattr__(self, method):
        return getattr(self.val, method)

    @property
    def val(self) -> np.ndarray:
        """Get the value of the input wavelength.

        """
        return np.squeeze(np.array([self.__param]))


class _ForwardScatteringProperty:
    """An abstract class to check something is a forward scattering property.

    """

    def __init__(self, parameter: np.ndarray, particle_size: np.ndarray,
                 wavelength: np.ndarray, name: str) -> None:
        self.__parameter = parameter
        self.__particle_size = particle_size
        self.__wavelength = wavelength
        self.__name = name

        self.__raise_error_if_inputs_are_bad()

    def __raise_error_if_inputs_are_bad(self) -> None:
        self.__raise_type_error_if_parameter_is_not_ndarray()
        self.__raise_value_error_if_shapes_do_not_match()

    def __raise_type_error_if_parameter_is_not_ndarray(self) -> None:
        if not isinstance(self.__parameter, np.ndarray):
            message = f'{self.__name} must be a numpy.ndarray.'
            raise TypeError(message)

    def __raise_value_error_if_shapes_do_not_match(self) -> None:
        if np.ndim(self.__parameter) != 2:
            message = f'{self.__name} must be a 2D array.'
            raise ValueError(message)

        if self.__parameter.shape[0] != self.__particle_size.shape[0]:
            raise ValueError('parameter does not match particle size shape')

        if self.__parameter.shape[1] != self.__wavelength.shape[0]:
            raise ValueError('parameter does not match wavelength shape.')

    @property
    def parameter(self) -> np.ndarray:
        return self.__parameter

    @property
    def particle_size(self) -> np.ndarray:
        return self.__particle_size

    @property
    def wavelength(self) -> np.ndarray:
        return self.__wavelength


class _Interpolator:
    """A structure to get values over a grid.

    Interpolator accepts an array of coefficients, along with the 1D particle
    size and wavelength grid over which they're defined. It contains methods
    to interpolate onto a new grid.

    """
    def __init__(self, coefficient: np.ndarray,
                 particle_size_grid: np.ndarray, wavelength_grid: np.ndarray,
                 particle_size: np.ndarray, wavelength: np.ndarray) -> None:
        """
        Parameters
        ----------
        coefficient
            ND array of coefficients to interpolate over. axis=-1 is assumed
            to be the wavelength dimension and axis=-2 is assumed to be the
            particle size dimension.
        particle_size_grid
            1D array of particle sizes over which the above properties are
            defined.
        wavelength_grid
            1D array of wavelengths over which the above properties are defined.
        particle_size
            1D array of particle sizes to regrid the properties onto.
        wavelength
            1D array of wavelengths to regrid the properties onto.

        """
        self.__coeff = coefficient
        self.__pgrid = particle_size_grid
        self.__wgrid = wavelength_grid
        self.__particle_size = particle_size
        self.__wavelength = wavelength

    def nearest_neighbor(self) -> np.ndarray:
        """Get the input coefficients at the input particle sizes and
        wavelengths using nearest neighbor interpolation to the input particle
        size and wavelength grid.

        """
        size_indices = self.__get_nearest_indices(self.__pgrid,
                                                  self.__particle_size)
        wavelength_indices = self.__get_nearest_indices(self.__wgrid,
                                                        self.__wavelength)
        return np.take(np.take(self.__coeff, size_indices, axis=-2),
                       wavelength_indices, axis=-1)

    def linear(self) -> np.ndarray:
        intpf = np.zeros((self.__coeff.shape[0], self.__particle_size.shape[0], self.__wavelength.shape[0]))
        for i in range(len(intpf)):
            pf = interp2d(self.__pgrid, self.__wgrid, self.__coeff[i, :, :].T)
            intpf[i, :, :] = pf(self.__particle_size, self.__wavelength).T
        return intpf

    @staticmethod
    def __get_nearest_indices(grid: np.ndarray, values: np.ndarray) \
            -> np.ndarray:
        # grid should be 1D; values can be ND
        return np.abs(np.subtract.outer(grid, values)).argmin(0)


class ForwardScattering:
    """A structure to store forward scattering properties.

    ForwardScattering will simply hang on to forward scattering properties and
    provides methods to regrid them into the input particle size profile and
    wavelengths.

    """
    def __init__(self, scattering_cross_section: np.ndarray,
                 extinction_cross_section: np.ndarray,
                 particle_size_grid: np.ndarray, wavelength_grid: np.ndarray,
                 particle_size: np.ndarray,
                 wavelength: np.ndarray,
                 reference_wavelength: float) -> None:
        """
        Parameters
        ----------
        scattering_cross_section
            2D array of the scattering cross section (the 0th axis is assumed to
            be the particle size axis, while the 1st axis is assumed to be the
            wavelength axis).
        extinction_cross_section
            2D array of the extinction cross section with same dims as above.
        particle_size_grid
            1D array of particle sizes over which the above properties are
            defined.
        wavelength_grid
            1D array of wavelengths over which the above properties are defined.
        particle_size
            1D array of particle sizes to regrid the properties onto.
        wavelength
            1D array of wavelengths to regrid the properties onto.

        """
        self.__c_sca = _ForwardScatteringProperty(
            scattering_cross_section, particle_size_grid, wavelength_grid,
            'scattering')
        self.__c_ext = _ForwardScatteringProperty(
            extinction_cross_section, particle_size_grid, wavelength_grid,
            'extinction')
        self.__particle_size = particle_size
        self.__wavelength = wavelength
        self.__wave_ref = reference_wavelength

        self.__gridded_c_scattering = np.array([])
        self.__gridded_c_extinction = np.array([])
        self.__gridded_ssa = np.array([])
        self.__extinction = np.array([])

    def make_nn_properties(self) -> None:
        """Make the forward scattering properties at the nearest neighbor
        particle sizes and wavelengths.

        """
        self.__gridded_c_scattering = self.__make_gridded_c_scattering('nn')
        self.__gridded_c_extinction = self.__make_gridded_c_extinction('nn')
        self.__gridded_ssa = self.__make_gridded_ssa()
        self.__extinction = self.__make_extinction_grid('nn')

    def make_linear_properties(self) -> None:
        """Make the forward scattering properties at the input particle sizes
        and wavelengths by linearly interpolating them onto the input grid.

        .. warning::
           Due to a known bug, this does not work!

        """
        self.__gridded_c_scattering = self.__make_gridded_c_scattering('linear')
        self.__gridded_c_extinction = self.__make_gridded_c_extinction('linear')
        self.__gridded_ssa = self.__make_gridded_ssa()
        self.__extinction = self.__make_extinction_grid('linear')

    def __make_gridded_c_scattering(self, kind: str) -> np.ndarray:
        interp = _Interpolator(self.__c_sca.parameter,
                               self.__c_sca.particle_size,
                               self.__c_sca.wavelength, self.__particle_size,
                               self.__wavelength)

        return self.__interpolate_by_kind(interp, kind)

    def __make_gridded_c_extinction(self, kind: str) -> np.ndarray:
        interp = _Interpolator(self.__c_ext.parameter,
                               self.__c_ext.particle_size,
                               self.__c_ext.wavelength, self.__particle_size,
                               self.__wavelength)

        return self.__interpolate_by_kind(interp, kind)

    def __make_gridded_ssa(self):
        return self.__gridded_c_scattering / self.__gridded_c_extinction

    def __make_extinction_grid(self, kind: str) -> np.ndarray:
        """Make a grid of the extinction cross section at a reference wavelength.
        This is the extinction cross section at the input wavelengths divided by
        the extinction cross section at the reference wavelength.

        """
        interp = _Interpolator(self.__c_ext.parameter,
                               self.__c_ext.particle_size,
                               self.__c_ext.wavelength, self.__particle_size,
                               np.array([self.__wave_ref]))
        extinction_profile = np.squeeze(self.__interpolate_by_kind(interp, kind))
        return (self.__gridded_c_extinction.T / extinction_profile).T

    @staticmethod
    def __interpolate_by_kind(interp: _Interpolator, kind: str) -> np.ndarray:
        # TODO: in python3.10 do a switch case
        if kind == 'nn':
            return interp.nearest_neighbor()
        elif kind == 'linear':
            return interp.linear()

    @property
    def scattering_cross_section(self) -> np.ndarray:
        """Get the scattering cross section on the new grid.

        """
        return self.__gridded_c_scattering

    @property
    def extinction_cross_section(self) -> np.ndarray:
        """Get the extinction cross section on the new grid.

        """
        return self.__gridded_c_extinction

    @property
    def single_scattering_albedo(self) -> np.ndarray:
        """Get the single scattering albedo on the new grid.

        """
        return self.__gridded_ssa

    @property
    def extinction(self) -> np.ndarray:
        """Get the extinction coefficient at the reference wavelength on the new
        grid.

        """
        return self.__extinction


class OpticalDepth:
    """A structure to compute the optical depth given profiles.

    OpticalDepth accepts a mixing ratio profile and atmospheric column density
    profile to compute the optical depth at each wavelength. It also accepts
    an extinction profile to scale the optical depth to a reference wavelength,
    and ensures that the sum over the layers is equal to the total column
    integrated optical depth.

    """
    def __init__(self, mixing_ratio_profile: np.ndarray,
                 column_density_layers: np.ndarray,
                 extinction: np.ndarray,
                 column_integrated_optical_depth: float) -> None:
        """
        Parameters
        ----------
        mixing_ratio_profile
            1D array of mixing ratio.
        column_density_layers
            1D array of the column density in the layers.
        extinction
            2D array of extinction coefficients.
        column_integrated_optical_depth
            The total column integrated optical depth.

        """
        self.__q_prof = mixing_ratio_profile
        self.__colden = column_density_layers
        self.__extinction = extinction
        self.__OD = column_integrated_optical_depth

        self.__optical_depth = self.__calculate_total_optical_depth()

    def __calculate_total_optical_depth(self) -> np.ndarray:
        normalization = np.sum(self.__q_prof * self.__colden)
        profile = self.__q_prof * self.__colden * self.__OD / normalization
        return (profile * self.__extinction.T).T

    @property
    def total(self) -> np.ndarray:
        """Get the total optical depth in each of the layers.

        """
        return self.__optical_depth


class TabularLegendreCoefficients:
    """Create a grid of Legendre coefficients (particle size, wavelength).

    TabularLegendreCoefficients accepts a 3D array of Legendre polynomial
    coefficients of the phase function that depend on particle size and
    wavelength and casts them to an array of the proper shape for use in DISORT
    given a vertical particle size gradient.

    """

    def __init__(self, coefficients: np.ndarray, particle_size_grid: np.ndarray,
                 wavelength_grid: np.ndarray, particle_size_profile: np.ndarray,
                 wavelengths: np.ndarray, max_moments: int = None) -> None:
        """
        Parameters
        ----------
        coefficients
            3D array of Legendre coefficients that depend on particle
            size and wavelength. It is assumed to have shape [n_moments,
            particle_size_grid, wavelength_grid]
        particle_size_grid
            1D array of particle sizes associated with the coefficients matrix.
        wavelength_grid
            1D array of wavelengths associated with the coefficients matrix.
        wavelengths
            ND array of wavelengths where to cast coefficients to.
        particle_size_profile
            1D array of particle sizes.
        max_moments
            The maximum number of coefficients to use in the array. If None,
            all available coefficients are used.

        """
        self.__coefficients = coefficients
        self.__particle_grid = particle_size_grid
        self.__wavelength_grid = wavelength_grid
        self.__particle_profile = particle_size_profile
        self.__wavelengths = wavelengths

        self.__n_layers = self.__make_n_layers()
        self.__n_moments = self.__make_n_moments(max_moments)

        self.__phase_function = np.array([])

    def __make_n_layers(self) -> int:
        return len(self.__particle_grid)

    def __make_n_moments(self, max_moments) -> int:
        return self.__coefficients.shape[0] if max_moments is None else max_moments

    def __normalize_coefficients(self, unnormalized_coefficients):
        coeff = unnormalized_coefficients[:self.__n_moments, :]
        moment_indices = np.linspace(0, self.__n_moments-1, num=self.__n_moments)
        normalization = moment_indices * 2 + 1
        return (coeff.T / normalization).T

    def make_nn_phase_function(self) -> None:
        """Make the phase function at the nearest neighbor particle sizes and
        wavelengths.

        """
        self.__make_phase_function('nn')

    def make_linear_phase_function(self) -> None:
        """Make the phase function by linearly interpolating between the input
        grid of particle sizes and wavelengths.

        """
        self.__make_phase_function('linear')

    def __make_phase_function(self, kind: str) -> None:
        interp = _Interpolator(self.__coefficients, self.__particle_grid,
                               self.__wavelength_grid, self.__particle_profile,
                               self.__wavelengths)

        # TODO: in python3.10 do a switch case
        if kind == 'nn':
            pf = interp.nearest_neighbor()
        elif kind == 'linear':
            pf = interp.linear()

        self.__phase_function = self.__normalize_coefficients(pf)

    @property
    def phase_function(self) -> np.ndarray:
        """Get the Legendre coefficients cast to the proper grid.

        """
        return self.__phase_function


class HenyeyGreenstein:
    """Hold the HenyeyGreenstein asymmetry parameters.

    HenyeyGreenstein holds the asymmetry parameters and ensures they're valid.
    It provides a method to convert these to Legendre polynomials.

    """

    def __init__(self, asymmetry: np.ndarray) -> None:
        r"""
        Parameters
        ----------
        asymmetry
            The asymmetry parameter.

        Raises
        ------
        ValueError
            Raised if the asymmetry parameter is not between -1 and 1.

        Notes
        -----
        The Henyey-Greenstein phase function is defined as follows

        .. math::
           p(\theta) = \frac{1}{4\pi} \frac{1 - g^2}{[1 + g^2 -2g \cos(\theta)]^{3/2}}

        """
        self.__g = asymmetry
        _HGAsymmetryParameter(asymmetry)

    def legendre_decomposition(self, n_moments: int) -> np.ndarray:
        r"""Get the Legendre decomposition of the asymmetry parameters up to
        a given number of moments.

        Parameters
        ----------
        n_moments
            The maximum number of moments to get the coefficients for (not
            including the 0th moment).

        Notes
        -----
        The Legendre decomposition is actually quite simple. The phase function
        can be decomposed as follows

        .. math::
           p(\mu) = \sum_{n=0}^{\infty} (2n + 1)g^n P_n(\mu)

        where :math:`n` is the moment number, :math:`g` is the asymmetry
        parameter, and :math:`P_n(\mu)` is the :math:`n`:sup:`th` Legendre
        polynomial. I'm not a mathematician so if :math:`g=0` I still assume
        the 0 :sup:`th` coefficient is 1.

        """
        moments = np.linspace(0, n_moments, num=n_moments+1)
        coeff = (2 * moments + 1) * np.power.outer(self.__g, moments)
        return np.moveaxis(coeff, -1, 0)


class _HGAsymmetryParameter:
    def __init__(self, asymmetry: np.ndarray) -> None:
        self.__g = asymmetry
        self.__raise_value_error_if_parameter_not_in_valid_range()

    def __raise_value_error_if_parameter_not_in_valid_range(self) -> None:
        if np.any(self.__g > 1) or np.any(self.__g < -1):
            raise ValueError('asymmetry must be in range [-1, 1].')
