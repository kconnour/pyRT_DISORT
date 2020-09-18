# 3rd-party imports
import numpy as np

# Local imports
from preprocessing.model.atmosphere import Atmosphere
from preprocessing.model.aerosol_column import Column
from preprocessing.model.phase_function import RayleighPhaseFunction


class ModelAtmosphere:
    def __init__(self, atmosphere, n_wavelengths, n_moments):
        self.atmosphere = atmosphere
        self.n_wavelengths = n_wavelengths
        self.n_moments = n_moments
        self.columns = []
        self.rayleigh_optical_depths = []
        self.tau_rayleigh = np.zeros((atmosphere.n_layers, n_wavelengths))
        self.total_optical_depths = np.nan
        self.total_single_scattering_albedo = np.nan
        self.polynomial_moments = np.nan

        assert isinstance(self.atmosphere, Atmosphere), 'atmosphere needs to be an instance of Atmosphere'

    def add_column(self, column):
        """ Add a column to the atmosphere

        Parameters
        ----------
        column: Column
            A column class

        Returns
        -------
        None
        """
        assert isinstance(column, Column), 'Try adding a Column instead.'
        self.columns.append(column)

    def add_rayleigh_constituent(self, constituent):
        """ Add the optical depths from Rayleigh scattering of a constituent

        Parameters
        ----------
        constituent: np.ndarray (n_layers, n_wavelengths)
            An array of optical depths in each grid point

        Returns
        -------
        None
        """
        self.rayleigh_optical_depths.append(constituent)

    def compute_model(self, optical_depth_minimum=10**-7):
        """ Compute all the relevant quantities for this model atmosphere

        Returns
        -------
        None
        """
        self.check_wavelengths()
        self.tau_rayleigh = self.calculate_rayleigh_optical_depths()
        self.total_optical_depths = self.calculate_column_optical_depths(optical_depth_minimum)
        self.total_single_scattering_albedo = self.calculate_single_scattering_albedos()
        self.polynomial_moments = self.calculate_polynomial_moments()

    def check_wavelengths(self):
        for col in range(len(self.columns)):
            wavelengths = self.columns[col].aerosol.wavelengths
            if not col:
                old_wavelengths = wavelengths
            else:
                if not np.array_equal(old_wavelengths, wavelengths):
                    raise SystemExit('The wavelengths of the aerosols don\'t match. Exiting...')

    def calculate_rayleigh_optical_depths(self):
        """ Calculate the Rayleigh optical depths

        Returns
        -------
        total_tau_rayleigh: np.ndarray(n_layers, n_wavelengths)
            The summed optical depths
        """
        total_tau_rayleigh = sum(self.rayleigh_optical_depths)
        return total_tau_rayleigh

    def calculate_column_optical_depths(self, optical_depth_minimum=10**-7):
        """ Calculate the optical depth of each layer in a column

        Parameters
        ----------
        optical_depth_minimum: float, optional
            The minimum optical depth allowable in each grid point. Default is 10**-7

        Returns
        -------
        column_optical_depths: np.ndarray (n_layers, n_wavelengths)
            The optical depths in each layer
        """
        # Add in the Rayleigh scattering contribution to the column optical depths
        column_optical_depths = np.copy(self.tau_rayleigh)

        # Add in the optical depths of each column
        for i in range(len(self.columns)):
            column_optical_depths += self.columns[i].calculate_aerosol_optical_depths(
                self.atmosphere.altitude_layers, self.atmosphere.column_density_layers)

        # Make sure ODs cannot be 0 to avoid dividing by 0 later on
        column_optical_depths = np.where(column_optical_depths < optical_depth_minimum, optical_depth_minimum,
                                         column_optical_depths)
        return column_optical_depths

    def calculate_single_scattering_albedos(self):
        """ Calculate the single scattering albedo of each layer in a column

        Returns
        -------
        single_scattering_albedo: np.ndarray (n_layers, n_wavelengths)
            The SSAs in each layer
        """

        # Add in the Rayleigh scattering contribution to the single scattering albedos
        single_scattering_albedos = np.copy(self.tau_rayleigh)

        # Add in the single scattering albedo of each aerosol
        for i in range(len(self.columns)):
            ssa = self.columns[i].aerosol.single_scattering_albedo
            aerosol_column_optical_depths = self.columns[i].calculate_aerosol_optical_depths(
                self.atmosphere.altitude_layers, self.atmosphere.column_density_layers)
            single_scattering_albedos += ssa * aerosol_column_optical_depths

        total_column_optical_depths = self.calculate_column_optical_depths()
        column_single_scattering_albedos = single_scattering_albedos / total_column_optical_depths
        return column_single_scattering_albedos

    def calculate_polynomial_moments(self):
        """ Calculate the polynomial moments for the atmosphere

        Returns
        -------
        polynomial_moments: np.ndarray (n_moments, n_layers, n_wavelengths)
            An array of the polynomial moments
        """

        n_layers = self.atmosphere.n_layers
        rayleigh = RayleighPhaseFunction(self.n_moments)
        rayleigh_moments = rayleigh.make_phase_function(n_layers, self.n_wavelengths)
        tau_rayleigh = np.copy(self.tau_rayleigh)
    
        # Add in the Rayleigh scattering contribution to the polynomial moments
        polynomial_moments = tau_rayleigh * rayleigh_moments
    
        # Add in the polynomial moments for each column
        for i in range(len(self.columns)):
            ssa = self.columns[i].aerosol.single_scattering_albedo
            aerosol_column_optical_depths = self.columns[i].calculate_aerosol_optical_depths(
                self.atmosphere.altitude_layers, self.atmosphere.column_density_layers)
            aerosol_moments = self.columns[i].aerosol.phase_function.make_phase_function(n_layers, self.n_wavelengths,
                                                                                         self.n_moments)
            polynomial_moments += ssa * aerosol_column_optical_depths * aerosol_moments

        total_column_optical_depths = self.calculate_column_optical_depths()
        total_single_scattering_albedos = self.calculate_single_scattering_albedos()
        return polynomial_moments / (total_column_optical_depths * total_single_scattering_albedos)
