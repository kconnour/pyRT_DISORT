import numpy as np


def rayleigh_co2(wavelength):
    """ Get the Rayleigh scattering cross section for pure CO2. Taken from the paper:
    Sneep and Ubachs 2005, JQSRT, 92, 293-310.  Note: in their equation 4, 1.1427e6 SHOULD BE 1.1427e3.

    See also Ityaksov, Linnartz, Ubachs 2008, Chemical Physics Letters, 462, 31-34, for UV measurements.
    This has same problem with the factor of 1e3.

    Returns
    -------
    cross_section: float
        The scattering cross section in m**2
    """

    number_density = 2.5475605 * 10**19

    # Everyone uses cm for some reason... convert microns to cm
    wavelength = wavelength * 10**-4

    # Get the king factor from Ityaksov et al
    king_factor = 1.14 + (25.3*10**(-12) / wavelength**2)

    nu2 = 1 / wavelength**2
    term1 = 5799.3 / (16.618 * 10**9 - nu2) + 120.05 / (7.9609 * 10**9 - nu2) + 5.3334 / (5.6306 * 10**9 - nu2) + \
        4.3244 / (4.602 * 10**9 - nu2) + 1.2181 * 10**(-5) / (5.84745 * 10**6 - nu2)

    # Refractive index
    n = 1 + 1.1427 * 10**3 * term1
    factor1 = ((n**2 - 1) / (n**2 + 2))**2
    cross_section = 24 * np.pi**3 / (wavelength**4) / number_density**2 * factor1 * king_factor

    # cross_section is in cm**2 so convert to m**2
    cross_section /= 10**4
    return cross_section
