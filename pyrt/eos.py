"""This module provides functions for working with equation of state variables.
"""
import numpy as np
from numpy.typing import ArrayLike
from scipy.constants import Boltzmann
from scipy.integrate import quadrature as quad


def column_density(pressure: ArrayLike, temperature: ArrayLike,
                   altitude: ArrayLike) -> np.ndarray:
    """Create the column density from a given grid of equation of state
    variables, assuming each grid point can be hydrostatically approximated.

    Parameters
    ----------
    pressure: ArrayLike
        The 1-dimensional pressure [Pa] of the grid.
    temperature: ArrayLike
        The 1-dimensional temperature [K] of the grid.
    altitude: ArrayLike
        The 1-dimensional altitude [km] of each point in the grid. It must be
        monotonically decreasing, otherwise this function will give nonsense.

    Returns
    -------
    The column density of each point in the grid.

    Notes
    -----
    This function assumes the ideal gas law applies to each point in the grid.
    Additionally, it linearly interpolates each grid point vertically so that it
    can do the vertical integration, which may or may not be desirable if the
    pressure varies exponentially. The finer the vertical resolution, the more
    accurate this function will be.

    Examples
    --------
    Get the column density of each 1 km layer of an 80-boundary atmosphere
    representing Mars where the surface pressure is 600 Pa, the temperature is
    180 K everywhere, and the scale height is 10 km.

    >>> import numpy as np
    >>> from scipy.constants import Boltzmann
    >>> import pyrt
    >>> altitude = np.linspace(80, 0, num=80)
    >>> pressure = 600 * np.exp(-altitude / 10)
    >>> temperature = np.ones((80,)) * 180
    >>> colden = column_density(pressure, temperature, altitude)
    >>> colden.shape
    (79,)

    Get the column-integrated column density.
    >>> total_colden = np.sum(colden)
    >>> total_colden
    2.415575739216228e+27

    Compute the analytical result of this scenario.
    >>> analytical_result = 600 * 10000 * (1 - np.exp(-80/10)) / 180 / Boltzmann
    >>> analytical_result
    2.4135135900389294e+27
    """
    pressure = np.flip(np.asarray(pressure))
    temperature = np.flip(np.asarray(temperature))
    altitude = np.flip(np.asarray(altitude))

    def hydrostatic_profile(alt):
        return linear_grid(alt, pressure) / linear_grid(alt, temperature) / \
            Boltzmann

    def linear_grid(alt, grid):
        return np.interp(alt, altitude, grid)

    n = [quad(hydrostatic_profile, altitude[i], altitude[i + 1])[0]
         for i in range(len(altitude) - 1)]

    return np.array(n) * 1000


def scale_height(temperature: ArrayLike, mass: ArrayLike, gravity: float) -> np.ndarray:
    """Compute the scale height of each model layer.

    Parameters
    ----------
    temperature: ArrayLike
        The 1-dimensional temperature [K] of the grid.
    mass: ArrayLike
        The 1-dimensional particle mass [kg] of the grid.
    gravity: float
        The gravitational constant.

    Returns
    -------
    The scale height in each model layer.

    """
    return Boltzmann * temperature / (mass * gravity)
