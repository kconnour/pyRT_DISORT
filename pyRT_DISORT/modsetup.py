"""modsetup.py contains data structures for creating DISORT inputs not related
to atmospheric radiative properties.
"""
from warnings import warn
import numpy as np


class Control:
    """Control holds variables which modify how DISORT runs.

    Control simply holds booleans that determine which options are used when
    DISORT is run, and performs checks that the variables have values and types
    expected by DISORT.

    """
    def __init__(self, accuracy: float = 0.0, delta_m_plus: bool = False,
                 do_pseudo_sphere: bool = False, header: str = '',
                 only_fluxes: bool = False, print_variables=None,
                 radius: float = 6371., user_angles: bool = True,
                 user_optical_depths: bool = False) -> None:
        """
        Parameters
        ----------
        accuracy: float, optional
            The convergence criterion for azimuthal series. Default is 0.0.
        delta_m_plus: bool, optional
            Denote whether to do delta-M+ scaling. Default is False.
        do_pseudo_sphere: bool, optional
            Denote whether to apply a pseudo-spherical correction. Default is
            False.
        header: str, optional
            Specify what characters appear in the DISORT banner. Default is ''.
        only_fluxes: bool, optional
            Denote whether to include only fluxes. Default is False.
        print_variables
        radius
        user_angles
        user_optical_depths

        Raises
        ------
        TypeError
            Raised if any inputs are not of the correct type.
        ValueError
            Raised if header is too long.

        Warnings
        --------
        UserWarning
            Raised if accuracy is not between 0 and 0.01.

        """

        if print_variables is None:
            print_variables = [False, False, False, False, False]
        self.__accuracy = accuracy
        self.__delta_m_plus = delta_m_plus
        self.__do_pseudo_sphere = do_pseudo_sphere
        self.__header = header
        self.__only_fluxes = only_fluxes
        self.__print_variables = print_variables
        self.__radius = radius
        self.__user_angles = user_angles
        self.__user_optical_depths = user_optical_depths

        self.__raise_error_if_inputs_are_bad()
        self.__warn_if_inputs_are_bad()

    def __raise_error_if_inputs_are_bad(self) -> None:
        self.__raise_type_error_if_accuracy_is_not_float()
        self.__raise_type_error_if_delta_m_plus_is_not_bool()
        self.__raise_type_error_if_do_pseudo_sphere_is_not_bool()
        self.__raise_type_error_if_header_is_not_str()
        self.__raise_value_error_if_header_is_too_long()
        self.__raise_type_error_if_only_fluxes_is_not_bool()

    def __warn_if_inputs_are_bad(self) -> None:
        self.__warn_if_accuracy_is_outside_expected_range()

    def __raise_type_error_if_accuracy_is_not_float(self) -> None:
        if not isinstance(self.__accuracy, float):
            raise TypeError('accuracy must be a float.')

    def __raise_type_error_if_delta_m_plus_is_not_bool(self) -> None:
        self.__raise_type_error_if_not_bool(self.__delta_m_plus, 'delta_m_plus')

    def __raise_type_error_if_do_pseudo_sphere_is_not_bool(self) -> None:
        self.__raise_type_error_if_not_bool(
            self.__do_pseudo_sphere, 'do_pseudo_sphere')

    def __raise_type_error_if_header_is_not_str(self) -> None:
        if not isinstance(self.__header, str):
            raise TypeError('header must be a str.')

    def __raise_value_error_if_header_is_too_long(self):
        if len(self.__header) > 127:
            raise ValueError('header must be 127 characters or less.')

    def __raise_type_error_if_only_fluxes_is_not_bool(self):
        self.__raise_type_error_if_not_bool(self.__only_fluxes, 'only_fluxes')

    @staticmethod
    def __raise_type_error_if_not_bool(inp: bool, name: str) -> None:
        if not isinstance(inp, bool):
            raise TypeError(f'{name} must be a boolean.')

    def __warn_if_accuracy_is_outside_expected_range(self):
        if not 0 <= self.__accuracy <= 0.01:
            warn('accuracy is expected to be in range [0, 0.01].')

    @property
    def accuracy(self) -> float:
        """Get the accuracy---the convergence criterion for azimuthal series.

        Returns
        -------
        float
            The convergence criterion.

        Notes
        -----
        In DISORT, this variable is named "ACCUR". From the documentation:
        Convergence criterion for azimuthal (Fourier cosine)
        series.  Will stop when the following occurs twice:
        largest term being added is less than ACCUR times
        total series sum.  (Twice because there are cases where
        terms are anomalously small but azimuthal series has
        not converged.)  Should be between 0 and 0.01 to avoid
        risk of serious non-convergence.  Has no effect on
        problems lacking a beam source, since azimuthal series
        has only one term in that case.

        """
        return self.__accuracy

    @property
    def delta_m_plus(self) -> bool:
        """Get whether to perform delta-M+ scaling.

        Returns
        -------
        bool
            True if delta-M+ is requested; False otherwise.

        Notes
        -----
        In DISORT, this variable is named "DELTAMPLUS". From the
        documentation: The delta-M+ method of Lin et al. (2018),
        which is a much improved version of delta-M is implemented in
        version 4.0.

        """
        return self.__delta_m_plus

    @property
    def do_pseudo_sphere(self) -> bool:
        """Get whether to perform a pseudo-spherical correction.

        Returns
        -------
        bool
            True if a pseudo-spherical correction is requested; False otherwise.

        Notes
        -----
        In DISORT, this variable is named "DO_PSEUDO_SPHERE". There is no
        documentation on this variable.

        """
        return self.__do_pseudo_sphere

    @property
    def header(self) -> str:
        """Get the characters that will appear in the DISORT banner.

        Returns
        -------
        str
            The characters in the DISORT banner.

        Notes
        -----
        In DISORT, this variable is named "HEADER". From the documentation:
        A 127- (or less) character header for prints, embedded in
        a DISORT banner;  setting HEADER = '' (the null string)
        will eliminate both the banner and the header, and this
        is the only way to do so (HEADER is not controlled by any
        of the PRNT flags);  HEADER can be used to mark the
        progress of a calculation in which DISORT is called
        many times, while leaving all other printing turned off.

        """
        return self.__header

    @property
    def only_fluxes(self) -> bool:
        """Get whether DISORT should only return fluxes.

        Returns
        -------
        bool
            True if only fluxes are requested; False if fluxes and intensities
            are requested.

        Notes
        -----
        From the documentation: If True, return fluxes, flux divergences, and
        mean intensities; if False, return all those and intensities. In
        addition, if True the number of polar angles can be 0, the number of
        azimuthal angles can be 0, phi is not used, and all values of
        intensity (UU) will be set to 0.
        
        """
        return self.__only_fluxes


class Output:
    """Output is a data structure to hold the output arrays required by DISORT.

    Output creates all 8 arrays that get populated with values as DISORT is run.

    """

    def __init__(self, size) -> None:
        """
        Parameters
        ----------
        size: Size
            Data structure holding the model computational variables.

        Raises
        ------
        TypeError
            Raised if input is not an instance of Size.

        """
        self.__size = size

        self.__albedo_medium = self.__make_albedo_medium()
        self.__diffuse_up_flux = self.__make_diffuse_up_flux()
        self.__diffuse_down_flux = self.__make_diffuse_down_flux()
        self.__direct_beam_flux = self.__make_direct_beam_flux()
        self.__flux_divergence = self.__make_flux_divergence()
        self.__intensity = self.__make_intensity()
        self.__mean_intensity = self.__make_mean_intensity()
        self.__transmissivity_medium = self.__make_transmissivity_medium()

    def __raise_type_error_if_input_is_not_size(self) -> None:
        if not isinstance(self.__size, Size):
            raise TypeError('size must be an instance of Size.')

    def __make_albedo_medium(self) -> np.ndarray:
        return np.zeros(self.__size.n_umu)

    def __make_diffuse_up_flux(self) -> np.ndarray:
        return np.zeros(self.__size.n_user_levels)

    def __make_diffuse_down_flux(self) -> np.ndarray:
        return np.zeros(self.__size.n_user_levels)

    def __make_direct_beam_flux(self) -> np.ndarray:
        return np.zeros(self.__size.n_user_levels)

    def __make_flux_divergence(self) -> np.ndarray:
        return np.zeros(self.__size.n_user_levels)

    def __make_intensity(self) -> np.ndarray:
        return np.zeros((self.__size.n_umu, self.__size.n_user_levels,
                         self.__size.n_phi))

    def __make_mean_intensity(self) -> np.ndarray:
        return np.zeros(self.__size.n_user_levels)

    def __make_transmissivity_medium(self) -> np.ndarray:
        return np.zeros(self.__size.n_umu)

    @property
    def albedo_medium(self) -> np.ndarray:
        """Get the albedo of the medium output array.

        Returns
        -------
        np.ndarray
            The albedo of the medium.

        Notes
        -----
        In DISORT, this variable is named "ALBMED".

        """
        return self.__albedo_medium

    @property
    def diffuse_up_flux(self) -> np.ndarray:
        """Get the diffuse upward flux output array.

        Returns
        -------
        np.ndarray
            The diffuse upward flux.

        Notes
        -----
        In DISORT, this variable is named "FLUP".

        """
        return self.__diffuse_up_flux

    @property
    def diffuse_down_flux(self) -> np.ndarray:
        """Get the diffuse downward flux output array.

        Returns
        -------
        np.ndarray
            The diffuse downward flux.

        Notes
        -----
        In DISORT, this variable is named "RFLDN" (total minus direct-beam).

        """
        return self.__diffuse_down_flux

    @property
    def direct_beam_flux(self) -> np.ndarray:
        """Get the direct beam flux output array.

        Returns
        -------
        np.ndarray
            The direct beam flux.

        Notes
        -----
        In DISORT, this variable is named "RFLDIR".

        """
        return self.__direct_beam_flux

    @property
    def flux_divergence(self) -> np.ndarray:
        """Get the flux divergence output array.

        Returns
        -------
        np.ndarray
            The flux divergence.

        Notes
        -----
        In DISORT, this variable is named "DFDT", which is
        (d(net_flux) / d(optical_depth)). This is an exact result.

        """
        return self.__flux_divergence

    @property
    def intensity(self) -> np.ndarray:
        """Get the intensity output array.

        Returns
        -------
        np.ndarray
            The intensity.

        Notes
        -----
        In DISORT, this variable is named "UU".

        """
        return self.__intensity

    @property
    def mean_intensity(self) -> np.ndarray:
        """Get the mean intensity output array.

        Returns
        -------
        np.ndarray
            The mean intensity.

        Notes
        -----
        In DISORT, this variable is named "UAVG".

        """
        return self.__mean_intensity

    @property
    def transmissivity_medium(self) -> np.ndarray:
        """Get the transmissivity of the medium output array.

        Returns
        -------
        np.ndarray
            The transmissivity of the medium.

        Notes
        -----
        In DISORT, this variable is named "TRNMED".

        """
        return self.__transmissivity_medium


class Size:
    """Size is a data structure to hold the computational variables for DISORT.

    Size holds the number of model layers, streams, moments, and angles. It also
    performs basic checks that these values are plausible. Objects of this class
    are meant to be be used as inputs to other classes.

    """

    # TODO: I'd like a better name than umu and phi and better description of
    #  user levels
    def __init__(self, n_layers: int, n_moments: int, n_streams: int,
                 n_umu: int, n_phi: int, n_user_levels: int) -> None:
        """
        Parameters
        ----------
        n_layers: int
            The number of layers to use in the model.
        n_moments: int
            The number of polynomial moments to use in the model.
        n_streams: int
            The number of streams to use in the model.
        n_umu: int
            The number of umu to use in the model.
        n_phi: int
            The number of azimuthal angles to use in the model.
        n_user_levels: int
            The number of user levels to use in the model.

        Raises
        ------
        TypeError
            Raised if any of the inputs are not positive integers, if n_streams
            is not even, or if n_streams is greater than n_moments.

        """
        self.__n_layers = n_layers
        self.__n_moments = n_moments
        self.__n_streams = n_streams
        self.__n_umu = n_umu
        self.__n_phi = n_phi
        self.__n_user_levels = n_user_levels

        self.__raise_error_if_inputs_are_unrealistic()

    def __raise_error_if_inputs_are_unrealistic(self) -> None:
        self.__raise_error_if_n_layers_is_unrealistic()
        self.__raise_error_if_n_moments_is_unrealistic()
        self.__raise_error_if_n_streams_is_unrealistic()
        self.__raise_error_if_n_umu_is_unrealistic()
        self.__raise_error_if_n_phi_is_unrealistic()
        self.__raise_error_if_n_user_levels_is_unrealistic()

    def __raise_error_if_n_layers_is_unrealistic(self) -> None:
        self.__raise_type_error_if_input_is_not_positive_int(
            self.__n_layers, 'n_layers')

    def __raise_error_if_n_moments_is_unrealistic(self) -> None:
        self.__raise_type_error_if_input_is_not_positive_int(
            self.__n_moments, 'n_moments')

    def __raise_error_if_n_streams_is_unrealistic(self) -> None:
        self.__raise_type_error_if_input_is_not_positive_int(
            self.__n_streams, 'n_streams')
        self.__raise_value_error_if_n_streams_is_more_than_n_moments()
        self.__raise_value_error_if_n_streams_is_not_even()

    def __raise_error_if_n_umu_is_unrealistic(self) -> None:
        self.__raise_type_error_if_input_is_not_positive_int(
            self.__n_umu, 'n_umu')

    def __raise_error_if_n_phi_is_unrealistic(self) -> None:
        self.__raise_type_error_if_input_is_not_positive_int(
            self.__n_phi, 'n_phi')

    def __raise_error_if_n_user_levels_is_unrealistic(self) -> None:
        self.__raise_type_error_if_input_is_not_positive_int(
            self.__n_user_levels, 'n_user_levels')

    def __raise_type_error_if_input_is_not_positive_int(
            self, inp: int, name: str) -> None:
        self.__raise_type_error_if_not_int(inp, name)
        self.__raise_type_error_if_not_positive(inp, name)

    @staticmethod
    def __raise_type_error_if_not_int(inp: int, name: str) -> None:
        if not isinstance(inp, int):
            raise ValueError(f'{name} must be an int.')

    @staticmethod
    def __raise_type_error_if_not_positive(inp: int, name: str) -> None:
        if not inp > 0:
            raise ValueError(f'{name} must be positive.')

    def __raise_value_error_if_n_streams_is_more_than_n_moments(self) -> None:
        if self.__n_streams > self.__n_moments:
            err_msg = f'There should not be more input streams ' \
                      f'({self.__n_streams}) than moments ({self.__n_moments}).'
            raise ValueError(err_msg)

    def __raise_value_error_if_n_streams_is_not_even(self) -> None:
        if self.__n_streams % 2 != 0:
            raise ValueError('n_streams must be even.')

    @property
    def n_layers(self) -> int:
        """Get the input number of layers.

        Returns
        -------
        int
            The number of layers.

        """
        return self.__n_layers

    @property
    def n_moments(self) -> int:
        """Get the input number of moments.

        Returns
        -------
        int
            The number of moments.

        """
        return self.__n_moments

    @property
    def n_phi(self) -> int:
        """Get the number of phis.

        Returns
        -------
        int
            The number of azimuthal angles.

        """
        return self.__n_phi

    @property
    def n_streams(self) -> int:
        """Get the input number of streams.

        Returns
        -------
        int
            The number of streams.

        """
        return self.__n_streams

    @property
    def n_umu(self) -> int:
        """Get the number of umus.

        Returns
        -------
        int
            The number of umus.

        """
        return self.__n_umu

    @property
    def n_user_levels(self) -> int:
        """Get the number of user levels.

        Returns
        -------
        int
            The number of user levels.

        """
        return self.__n_user_levels


# TODO: Become sure of what this does and remove it
class Unsure:
    """This class makes the variable "h_lyr". I don't really know what this
    variable does since it can be all 0s.

    """
    def __init__(self, size):
        self.size = size
        self.__h_lyr = np.zeros(self.size.n_layers+1)

    @property
    def h_lyr(self):
        return self.__h_lyr
