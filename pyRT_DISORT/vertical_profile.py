"""The vertical profile module contains structures to create vertical profiles
for use in DISORT.
"""
import numpy as np


# TODO: In Python3.10, these can be float | np.ndarray
class Conrath:
    """Compute Conrath profile(s).

    Conrath creates Conrath profile(s) on an input grid of altitudes given a
    set of input parameters.

    """

    def __init__(self, altitude_grid: np.ndarray, q0: np.ndarray,
                 scale_height: np.ndarray, nu: np.ndarray) -> None:
        r"""
        Parameters
        ----------
        altitude_grid
            The altitudes at which to construct a Conrath profile. These are
            assumed to be decreasing to keep with DISORT's convention. If this
            is an MxN array, it will construct N profiles.
        q0
            The surface mixing ratio for each of the Conrath profiles. If all
            profiles have the same q0, this can be a float; otherwise, for an
            MxN altitude_grid, this should be an array of length N.
        scale_height
            The scale height of each of the Conrath profiles. If all profiles
            have the same scale height, this can be a float; otherwise, for an
            MxN altitude_grid, this should be an array of length N. It should
            also have the same units as the altitudes in altitude_grid.
        nu
            The nu parameter of each of the Conrath profiles. If all profiles
            have the same nu, this can be a float; otherwise, for an
            MxN altitude_grid, this should be an array of length N.

        Raises
        ------
        ValueError
            Raised if the inputs cannot be broadcast to the correct shape
            for computations.

        Notes
        -----
        A Conrath profile is defined as
        :math:`q(z) = q_0 * e^{\nu(1 - e^{z/H})}`

        """
        self.__profile = \
            self.__make_profile(altitude_grid, q0, scale_height, nu)

    @staticmethod
    def __make_profile(altitude_grid: np.ndarray, q0: np.ndarray,
                       scale_height: np.ndarray, nu: np.ndarray) -> np.ndarray:
        try:
            altitude_scaling = np.true_divide(altitude_grid, scale_height)
            return q0 * np.exp(nu * (1 - np.exp(altitude_scaling)))
        except ValueError as ve:
            raise ValueError('The input arrays have incompatible shapes.') \
                from ve

    @property
    def profile(self) -> np.ndarray:
        """Get the Conrath profile(s).

        Returns
        -------
        np.ndarray
            The Conrath profile(s).

        """
        return self.__profile


# TODO: It'd be nice to add "subgrid" fixes. If a cloud is 80% in a grid, the
#  profile should be 0.8
class Uniform:
    """Compute uniform volumetric mixing ratio profile(s).

    Uniform creates uniform volumetric mixing ratio profile(s) on an input
    grid of altitudes given a set of top and bottom altitudes.

    """

    def __init__(self, altitude_grid: np.ndarray, bottom_altitude: np.ndarray,
                 top_altitude: np.ndarray) -> None:
        """
        Parameters
        ----------
        altitude_grid
            The altitudes at which to construct a uniform profile. These are
            assumed to be decreasing to keep with DISORT's convention. If this
            is an MxN array, it will construct N profiles.
        bottom_altitude
            The bottom altitudes of each of the profiles. If all profiles have
            the same bottom altitudes, this can be a float; otherwise, for an
            MxN altitude_grid, this should be an array of length N.
        top_altitude
            The top altitudes of each of the profiles. If all profiles have
            the same top altitudes, this can be a float; otherwise, for an
            MxN altitude_grid, this should be an array of length N.

        Raises
        ------
        ValueError
            Raised if the inputs cannot be broadcast to the correct shape
            for computations.

        Notes
        -----
        Right now this only creates a uniform profile if an aerosol is
        completely within a grid point and cuts off the aerosol otherwise.

        """
        self.__profile = \
            self.__make_profile(altitude_grid, bottom_altitude, top_altitude)

    @staticmethod
    def __make_profile(altitude_grid: np.ndarray, bottom_altitude: np.ndarray,
                       top_altitude: np.ndarray) -> np.ndarray:
        try:
            return np.where((bottom_altitude <= altitude_grid) &
                            (altitude_grid <= top_altitude), 1, 0)
        except ValueError as ve:
            raise ValueError(f'The input arrays have incompatible shapes.') \
                from ve

    @property
    def profile(self) -> np.ndarray:
        """Get the uniform profile(s).

        Returns
        -------
        np.ndarray
            The uniform profile(s).

        """
        return self.__profile
