import numpy as np
from scipy.constants import Boltzmann
from scipy.integrate import quadrature as quad


def column_density(pressure: np.ndarray, temperature: np.ndarray, altitude: np.ndarray) -> np.ndarray:
    """Create the column density from a given grid of equation of state variables,
    assuming each grid point can be hydrostatically approximated.

    Parameters
    ----------
    pressure
        The pressure [Pa] of the grid.
    temperature
        The temperature [K] of the grid.
    altitude
        The altitude [km] of each point in the grid. It must be decreasing,
        otherwise this function will give nonsense.

    Returns
    -------
    The column density of each point in the grid.

    Notes
    -----
    The altitude must be decreasing so I can ensure I give np.interp a valid
    input. I wish it raised a warning or error if it's given junk, but alas it
    doesn't.

    Examples
    --------
    Get the column density of each 1 km layer of an 80 boundary atmosphere
    representing Mars, where the surface pressure is 600 Pa, the temperature is
    180 K everywhere, and the scale height is 10 km.

    >>> import numpy as np
    >>> from scipy.constants import Boltzmann
    >>> import pyrt
    >>> altitude = np.linspace(80, 0, num=80)
    >>> pressure = np.flip(600 * np.exp(-altitude / 10))
    >>> temperature = np.ones((80,)) * 180
    >>> colden = column_density(pressure, temperature, altitude)
    >>> colden.shape
    (79,)

    Get the total column density given by this function.
    >>> total_colden = np.sum(colden)
    >>> total_colden
    2.4155757392162294e+27

    Compute the analytical result of this scenario.
    >>> analytical_result = 600 * 10000 * (1 - np.exp(-80/10)) / 180 / Boltzmann
    >>> analytical_result
    2.4135135900389294e+27
    """

    pressure = np.flip(pressure)
    temperature = np.flip(temperature)
    altitude = np.flip(altitude)

    def hydrostatic_profile(alt):
        return linear_grid(alt, pressure) / linear_grid(alt, temperature) / Boltzmann

    def linear_grid(alt, grid):
        return np.interp(alt, altitude, grid)

    n = [quad(hydrostatic_profile, altitude[i], altitude[i + 1])[0] for i in range(len(altitude) - 1)]
    return np.array(n) * 1000
