"""modsetup.py contains structures for creating DISORT inputs not related to
the physical model---things like creating output arrays, setting the number of
streams, etc.
"""


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
