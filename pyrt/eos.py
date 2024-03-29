"""This module provides functions for working with equation of state variables.
"""
import numpy as np
from numpy.typing import ArrayLike
from scipy.constants import Boltzmann
from scipy.integrate import quad


def column_density(pressure: ArrayLike, temperature: ArrayLike,
                   altitude: ArrayLike) -> np.ndarray:
    r"""Create the column density from a given grid of equation of state
    variables, assuming each grid point can be hydrostatically approximated.

    Parameters
    ----------
    pressure: ArrayLike
        The 1-dimensional pressure [Pa] of the grid.
    temperature: ArrayLike
        The 1-dimensional temperature [K] of the grid.
    altitude: ArrayLike
        The 1-dimensional altitude [m] of each point in the grid. It must be
        monotonically decreasing, otherwise this function will give nonsense.

    Returns
    -------
    The column density [:math:`\frac{\text{particles}}{m^2}`] of each point in
    the vertical column.

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
    >>> altitude = np.linspace(80000, 0, num=81)
    >>> pressure = 600 * np.exp(-altitude / 10000)
    >>> temperature = np.ones((81,)) * 180
    >>> colden = column_density(pressure, temperature, altitude)
    >>> colden.shape
    (80,)

    Get the column-integrated column density.

    >>> total_colden = np.sum(colden)
    >>> f'{total_colden:.2e}'
    '2.42e+27'

    Compute the analytical result of this scenario.

    >>> analytical_result = 600 * 10000 * (1 - np.exp(-80000/10000)) / 180 / Boltzmann
    >>> f'{analytical_result:.2e}'
    '2.41e+27'

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

    return np.flip(np.array(n))


def scale_height(temperature: ArrayLike, mass: ArrayLike, gravity: float) -> np.ndarray:
    r"""Compute the scale height of each model layer.

    Parameters
    ----------
    temperature: ArrayLike
        The 1-dimensional temperature [K] of the grid.
    mass: ArrayLike
        The 1-dimensional particle mass [kg] of the grid.
    gravity: float
        The gravitational acceleration [:math:`\frac{m}{s^2}`].

    Returns
    -------
    The scale height [m] in each model layer.

    Examples
    --------
    Get the scale height of Mars's atmosphere, where the temperature is 210 K.
    Note the atmosphere is primarily carbon dioxide and the gravitational
    acceleration is 3.71 :math:`\frac{m}{s^2}`.

    >>> import numpy as np
    >>> from scipy.constants import m_u
    >>> import pyrt
    >>> int(np.rint(pyrt.scale_height(210, 44*m_u, 3.71)))
    10696

    """
    return Boltzmann * temperature / (mass * gravity)
