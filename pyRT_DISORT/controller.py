"""The controller module holds miscellaneous classes responsible for creating
arrays that generally control how DISORT runs.
"""
from typing import Any
import numpy as np


class ComputationalParameters:
    """Create a data structure for holding the size of computational parameters.

    ComputationalParameters holds the number of model layers, streams, moments,
    and angles. It also performs basic checks that these values are plausible.
    Objects of this class are meant to be be used as inputs to other classes.

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
        ValueError
            Raised if any input is not positive finite, if n_streams is not
            even, or if n_streams is greater than n_moments.

        """
        self.__n_layers = self.__make_parameter(n_layers, 'n_layers')
        self.__n_moments = self.__make_parameter(n_moments, 'n_moments')
        self.__n_streams = self.__make_n_streams(n_streams)
        self.__n_umu = self.__make_parameter(n_umu, 'n_umu')
        self.__n_phi = self.__make_parameter(n_phi, 'n_phi')
        self.__n_user_levels = self.__make_parameter(
            n_user_levels, 'n_user_levels')

    @staticmethod
    def __make_parameter(param: Any, name: str) -> int:
        try:
            if param <= 0 or np.isinf(param):
                raise ValueError(f'{name} must be positive, finite')
            return int(param)
        except TypeError as te:
            raise TypeError(f'{name} must be an int.') from te

    def __make_n_streams(self, n_streams) -> int:
        try:
            if n_streams <= 0 or np.isinf(n_streams):
                raise ValueError('n_streams must be positive, finite')
            elif n_streams % 2 != 0:
                raise ValueError('n_streams must be an even number.')
            elif n_streams > self.__n_moments:
                raise ValueError('n_moments must be greater than n_streams.')
            return int(n_streams)
        except TypeError as te:
            raise TypeError('n_streams must be an int.') from te

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
