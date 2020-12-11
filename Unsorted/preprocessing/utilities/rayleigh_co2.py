import numpy as np


def calculate_rayleigh_co2_optical_depths(wavelengths, column_density_layers):
    """ Calculate the Rayleigh CO2 optical depth at a given wavelength

    Parameters
    ----------
    wavelengths: (n_wavelengths)
        The wavelengths of the observation
    column_density_layers: np.ndarray (n_layers)
        The column densities in each layer

    Returns
    -------
    tau_rayleigh_co2: np.ndarray(n_layers, n_wavelengths)
        The Rayleigh optical depths in each layer at each wavelength
    """

    cross_section = calculate_molecular_cross_section(wavelengths)
    tau_rayleigh_co2 = np.outer(column_density_layers, cross_section)
    return tau_rayleigh_co2


def calculate_molecular_cross_section(wavelengths):
    """ Calculate the molecular cross section (m**2 / molecule)

    Parameters
    ----------
    wavelengths: np.ndarray (n_wavelengths)
        The wavelengths

    Returns
    -------
    cross_section: np.ndarray (n_layers, n_wavelengths)
        The molecular cross section in each grid point, in m**2 / molecule
    """
    # It'll be much easier at the end if I deal with multi-dimensional arrays from the get-go
    # Make an array (n_layers, n_wavelengths) and convert microns to 1/cm
    wavenumbers = 1 / (wavelengths * 10 ** -4)
    # Make an array (n_layers, n_wavelengths) convert molecules/m**3 to molecules/cm**3
    number_density = 25.47 * 10**18  # molecules / cm**3

    king_factor = 1.1364 + 25.3 * 10 ** -12 * wavenumbers ** 2
    index_of_refraction = co2_index_of_refraction(wavenumbers)

    cross_section = scattering_cross_section(wavenumbers, number_density, king_factor, index_of_refraction) * 10**-4
    return cross_section


def co2_index_of_refraction(wavenumbers):
    """ Calculate the index of refraction for CO2 using equation 13 and changing the coefficient to 10**3

    Parameters
    ----------
    wavenumbers: np.ndarray (n_layers, n_wavelengths)
        The wavenumbers

    Returns
    -------
    n: np.ndarray (n_layers, n_wavelengths)
        The indices of refraction in each grid point
    """
    n = 1 + 1.1427*10**3 * (5799.25 / (128908.9**2 - wavenumbers**2) + 120.05 / (89223.8**2 - wavenumbers**2) +
                            5.3334 / (75037.5**2 - wavenumbers**2) + 4.3244 / (67837.7**2 - wavenumbers**2) +
                            0.1218145*10**-4 / (2418.136**2 - wavenumbers**2))
    return n


def scattering_cross_section(wavenumbers, number_density, king_factor, index_of_refraction):
    """ Calculate the scattering cross section using equation 2

    Parameters
    ----------
    wavenumbers: np.ndarray (n_layers, n_wavelengths)
        The wavenumbers
    number_density: float
        The number densities at which this was calculated
    king_factor: np.ndarray (n_layers, n_wavelengths)
        The King factor
    index_of_refraction: np.ndarray (n_layers, n_wavelengths)
        The indices of refraction

    Returns
    -------
    np.ndarray (n_layers, n_wavelengths) in cm**2 / molecule
    """
    coefficient = 24 * np.pi**3 * wavenumbers**4 / number_density**2
    middle_term = ((index_of_refraction**2 - 1) / (index_of_refraction**2 + 2))**2
    return coefficient * middle_term * king_factor


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
