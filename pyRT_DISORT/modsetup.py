"""modsetup.py contains data structures for creating DISORT inputs not related
to atmospheric radiative properties.
"""
import numpy as np


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
