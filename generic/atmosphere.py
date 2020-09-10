import numpy as np
from scipy.constants import Boltzmann
from scipy.integrate import quadrature
from generic.aerosol import Aerosol
from generic.aerosol_column import Column

# Local imports
from utilities.rayleigh_co2 import rayleigh_co2, make_rayleigh_phase_function


class Atmosphere:
    def __init__(self, atmosphere_file, n_layers=14, z_bottom=0, z_top=100, p_surface=492, scale_height=10,
                 constant_pressure=True, top_temperature=160, bottom_temperature=212):
        # Needed for creating an atmosphere from user-defined values
        self.atm_file = atmosphere_file

        # Read in as much of the atmosphere as possible
        self.z, self.P, self.T, self.n, self.N = self.read_atmosphere()

        # If no z is provided, make the altitude and pressure grid at constant altitude
        if np.all(self.z == 0) and not constant_pressure:
            self._n_layers = n_layers
            self._z_bottom = z_bottom
            self._z_top = z_top
            self._H = scale_height
            self.z = self.make_constant_altitude_boundaries()
            self.P = self.make_pressure_profile(self.z) * p_surface

        # Otherwise make the altitudes and pressures at constant pressures
        elif np.all(self.z == 0) and constant_pressure:
            self._n_layers = n_layers
            self._z_bottom = z_bottom
            self._z_top = z_top
            self._H = scale_height
            self.z, unscaled_pressures = self.make_constant_pressure_boundaries()
            self.P = unscaled_pressures * p_surface

        # If temperature isn't provided, make it here
        if np.all(self.T == 0):
            self._T_bottom = bottom_temperature
            self._T_top = top_temperature
            self.T = self.make_constant_temperature_boundaries()

        self.z_midpoints = self.calculate_altitude_midpoints()

        # If number density isn't provided, make it here
        if np.all(self.n == 0):
            self.n = self.calculate_number_density(self.z_midpoints)

        # If column density isn't provided, make it here
        if np.all(self.N == 0):
            self.N = self.calculate_column_density()

        # Make a list to hold any aerosol columns that can be added to the atmosphere
        self.columns = []

        # Add Rayleigh scattering
        self.tau_rayleigh = 0

    def read_atmosphere(self):
        """ Read in the atmospheric quantities from atmosphere_file

        Returns
        -------
        altitudes: np.ndarray
            The altitudes from atmosphere_file
        pressures: np.ndarray
            The pressures from atmosphere_file
        temperatures: np.ndarray
            The temperatures from atmosphere_file
        number_density: np.ndarray
            The number density from atmosphere_file
        column_density: np.ndarray
            The column density from atmosphere_file
        """
        atmosphere = np.load(self.atm_file, allow_pickle=True)
        altitudes = atmosphere[:, 0]
        pressures = atmosphere[:, 1]
        temperatures = atmosphere[:, 2]
        number_density = atmosphere[:, 3]
        column_density = atmosphere[:, 4]
        return altitudes, pressures, temperatures, number_density, column_density

    def make_constant_altitude_boundaries(self):
        """ Make the boundaries for constant altitude layers

        Returns
        -------
        altitudes: np.ndarray
            An array of equally spaced boundaries of the altitudes
        """
        altitudes = np.linspace(self._z_bottom, self._z_top, num=self._n_layers + 1)
        return altitudes

    def make_constant_pressure_boundaries(self):
        """Make the boundaries for layers equally spaced in pressure. Assume an exponential pressure profile:
        P(z) = P_o * np.exp(-z/H)

        Returns
        -------
        boundaries: np.ndarray
            The boundaries
        """
        top_pressure = self.make_pressure_profile(self._z_top)
        bottom_pressure = self.make_pressure_profile(self._z_bottom)
        fractional_pressures = np.linspace(bottom_pressure, top_pressure, num=self._n_layers + 1, endpoint=True)
        boundaries = -self._H * np.log(fractional_pressures)
        return boundaries, fractional_pressures

    def make_pressure_profile(self, z):
        """Create a pressure profile for an exponential atmosphere

        Parameters
        ----------
        z: np.ndarray
            The altitudes to create the profile for

        Returns
        -------
        frac: np.ndarray or float
            The fraction of the surface pressure at a given altitude
        """
        frac = np.exp(-z / self._H)
        return frac

    def make_constant_temperature_boundaries(self):
        """ Make the boundaries for constant temperature layers

        Returns
        -------
        temperatures: np.ndarray
            An array of equally spaced boundaries of the temperatures
        """
        temperatures = np.linspace(self._T_bottom, self._T_top, num=self._n_layers + 1)
        return temperatures

    def calculate_altitude_midpoints(self):
        """ Compute the altitudes at the midpoint between each layer

        Returns
        -------
        midpoints: np.ndarray
            The midpoint altitudes, such that len(midpoint) = len(self.z) - 1
        """
        midpoints = (self.z[:-1] + self.z[1:]) / 2
        return midpoints

    def calculate_number_density(self, altitudes):
        """Calculate number density (particles / unit volume) at any altitude.
        Assume the atmosphere obeys the ideal gas law.

        Parameters
        ----------
        altitudes: np.ndarray
            The altitudes at which to compute the number density

        Returns
        -------
        number_density: np.ndarray
            The number density at the input altitudes
        """
        interpolated_pressure = np.interp(altitudes, self.z, self.P)
        interpolated_temperature = np.interp(altitudes, self.z, self.T)
        number_density = interpolated_pressure / interpolated_temperature / Boltzmann
        return number_density

    def calculate_column_density(self):
        """Calculate the column density (particles / unit area) by integrating number density with Gaussian quadrature.

        Returns
        -------
        column_density: np.ndarray
            The column density of each layer
        """
        column_density = np.zeros(len(self.z_midpoints))
        for i in range(len(column_density)):
            integral, absolute_error = quadrature(self.calculate_number_density, self.z[i] * 1000,
                                                  self.z[i + 1] * 1000)
            column_density[i] = integral * 1000

        return column_density

    def add_rayleigh_co2_optical_depth(self, wavelength):
        """ Add the optical depth from Rayleigh scattering of CO2 to the total Rayleigh optical depth

        Parameters
        ----------
        wavelength:
            The wavelength of the observation

        Returns
        -------
        None
        """
        self.tau_rayleigh += self.calculate_rayleigh_co2_optical_depth(wavelength)

    def calculate_rayleigh_co2_optical_depth(self, wavelength):
        """ Calculate the Rayleigh CO2 optical depth at a given wavelength

        Parameters
        ----------
        wavelength:
            The wavelength of the observation

        Returns
        -------

        """
        tau_rayleigh_co2 = np.outer(self.N, rayleigh_co2(wavelength))
        return tau_rayleigh_co2

    def add_column(self, column):
        """ Add a column of an aerosol or gas to the atmosphere

        Parameters
        ----------
        column: Column
            A column instance of an aerosol or gas

        Returns
        -------
        None
        """
        self.columns.append(column)

    def calculate_column_optical_depth(self, optical_depth_minimum=10**-7):
        """ Calculate the optical depth of each layer in a column

        Returns
        -------
        column_optical_depth: np.ndarray
            The optical depths in each layer
        """
        # Add in Rayleigh scattering
        column_optical_depth = np.copy(self.tau_rayleigh)

        # Add in the optical depths of each column
        for i in range(len(self.columns)):
            column_optical_depth += self.columns[i].calculate_aerosol_optical_depths(self.z_midpoints, self.N)

        # Make sure ODs cannot be 0 to avoid dividing by 0 later on
        column_optical_depth = np.where(column_optical_depth < optical_depth_minimum, optical_depth_minimum, column_optical_depth)
        return column_optical_depth

    def calculate_single_scattering_albedo(self):
        """ Calculate the single scattering albedo of each layer in a column

        Returns
        -------
        single_scattering_albedo: np.ndarray
            The SSAs in each layer
        """

        # Add in Rayleigh scattering
        single_scattering_albedo = np.copy(self.tau_rayleigh)

        # Add in the single scattering albedo of each aerosol
        for i in range(len(self.columns)):
            scattering_ratio = self.columns[i].aerosol.scattering_coeff
            optical_depths = self.columns[i].calculate_aerosol_optical_depths(self.z_midpoints, self.N)
            single_scattering_albedo += scattering_ratio * optical_depths

        column_optical_depth = self.calculate_column_optical_depth()
        return single_scattering_albedo / column_optical_depth

    def calculate_polynomial_moments(self):
        """ Calculate the polynomial moments for the atmosphere

        Returns
        -------
        polynomial_moments: np.ndarray
            An array of the polynomial moments
        """
        # Get info I'll need
        n_moments = self.columns[0].aerosol.n_moments
        n_layers = len(self.z_midpoints)
        n_wavelengths = len(self.columns[0].aerosol.wavelengths)
        rayleigh_moments = make_rayleigh_phase_function(n_moments, n_layers, n_wavelengths)
        tau_rayleigh = np.copy(self.tau_rayleigh)

        # Start by populating PMOM with Rayleigh scattering
        polynomial_moments = tau_rayleigh * rayleigh_moments

        # Add in the moments for each column
        for i in range(len(self.columns)):
            scattering = self.columns[i].aerosol.scattering_coeff
            column_optical_depths = self.columns[i].calculate_aerosol_optical_depths(self.z_midpoints, self.N)
            moments = self.columns[i].aerosol.phase_function
            moments_holder = np.zeros(polynomial_moments.shape)
            for j in range(len(self.z_midpoints)):
                moments_holder[:, j, :] = moments
            polynomial_moments += scattering * column_optical_depths * moments_holder

        column_optical_depth = self.calculate_column_optical_depth()
        single_scattering_albedo = self.calculate_single_scattering_albedo()
        scaling = column_optical_depth * single_scattering_albedo
        scaling = np.where(scaling == 0, 10**10, scaling)
        return polynomial_moments / scaling


atm = Atmosphere('/home/kyle/repos/pyRT_DISORT/planets/mars/aux/mars_atm.npy')
atm.add_rayleigh_co2_optical_depth(np.array([1, 2, 5, 15, 24, 49]))
dust = Aerosol(128, 'emp', 1, np.array([1, 2, 5, 15, 24, 49]), g=0.5,
               aerosol_file='/home/kyle/repos/pyRT_DISORT/planets/mars/aux/dust.npy',
               legendre_file='/home/kyle/repos/pyRT_DISORT/planets/mars/aux/legendre_coeff_dust.npy')
c = Column(dust, 10, 0.5, 1)
atm.add_column(c)
m = atm.calculate_polynomial_moments()
print(m[:, 7, 3])
