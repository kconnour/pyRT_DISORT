"""The vertical profile module contains structures to create vertical profiles
for use in DISORT.
"""
import numpy as np


class Conrath:
    """Compute a Conrath profile on a grid.

    Conrath creates Conrath profile(s) on an input grid of altitudes given a
    set of input parameters.

    """

    def __init__(self, altitude_grid: np.ndarray, q0: np.ndarray,
                 scale_height: np.ndarray, nu: np.ndarray) -> None:
        r"""
        Parameters
        ----------
        altitude_grid: np.ndarray
            The altitudes which to construct a Conrath profile. These are
            assumed to be decreasing to keep with DISORT's convention. If this
            is an MxN array, it will construct N profiles.
        q0: np.ndarray
            The surface mixing ratio for each of the Conrath profiles. If all
            profiles have the same q0, this can be a float; otherwise, for an
            MxN altitude_grid, this should be of length N.
        scale_height: np.ndarray
            The scale height of each of the Conrath profiles. If all profiles
            have the same scale height, this can be a float; otherwise, for an
            MxN altitude_grid, this should be of length N. It should also have
            the same units as the altitudes in altitude_grid.
        nu: np.ndarray
            The nu parameter of each of the Conrath profiles. If all profiles
            have the same nu, this can be a float; otherwise, for an
            MxN altitude_grid, this should be of length N.

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
            altitude_scale = np.true_divide(altitude_grid, scale_height)
            return q0 * np.exp(nu * (1 - np.exp(altitude_scale)))
        except ValueError as ve:
            raise ValueError(f'Cannot broadcast a {altitude_grid.shape} array '
                             f'with a {scale_height} array') from ve

    @property
    def profile(self) -> np.ndarray:
        """Get the Conrath profile(s).

        Returns
        -------
        np.ndarray
            The Conrath profile(s).

        """
        return self.__profile


class Uniform:
    def __init__(self, altitude_grid: np.ndarray, bottom_altitude: float, top_altitude: float) -> None:
        self.__profile = self.__make_profile(altitude_grid, bottom_altitude, top_altitude)

    def __make_profile(self, altitude_grid, bottom_altitude, top_altitude) -> np.ndarray:
        bottom_profile = self.__make_bottom_profile(altitude_grid, bottom_altitude)
        print(bottom_profile)

    def __make_bottom_profile(self, altitude_grid, bottom_altitude) -> np.ndarray:
        bottom_index = np.argmax(altitude_grid <= bottom_altitude)
        grid_delta = altitude_grid[bottom_index - 1] - altitude_grid[bottom_index]
        alt_delta = altitude_grid[bottom_index - 1] - bottom_altitude
        subgrid = alt_delta / grid_delta

        bottom_profile = np.zeros(altitude_grid.shape)
        bottom_profile[:bottom_index-1] = 1
        bottom_profile[bottom_index] = subgrid
        return bottom_profile


if __name__ == '__main__':
    a = np.linspace(80, 0, num=17)
    print(a)
    u = Uniform(a, 21, 51)

