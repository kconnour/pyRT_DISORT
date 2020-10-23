# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.preprocessing.model.aerosol import Aerosol
from pyRT_DISORT.preprocessing.model.atmosphere import Layers


class Column:
    """Create an aerosol column to hold the aerosol's properties in each layer. Right now it constructs a column
    using Conrath parameters. Will be extended to allow user input vertical distributions. """
    def __init__(self, aerosol, layers, aerosol_scale_height, conrath_nu, particle_sizes,
                 column_integrated_optical_depths):
        """ Initialize the class

        Parameters
        ----------
        aerosol: Aerosol
            The aerosol for which this class will construct a column
        layers: Layers
            The equations of state in each of the layers in the plane parallel atmosphere
        aerosol_scale_height: float
            The scale height of the aerosol [km]
        conrath_nu: float
            The Conrath nu parameter
        particle_sizes: np.ndarray
            The effective radii of this aerosol [microns]
        column_integrated_optical_depths: np.ndarray
            The column-integrated optical depth at the effective radii
        """
        self.aerosol = aerosol
        self.layers = layers
        self.H = aerosol_scale_height
        self.nu = conrath_nu
        self.particle_sizes = particle_sizes
        self.column_integrated_optical_depths = column_integrated_optical_depths
        self.__check_input_types()

        self.multisize_hyperspectral_total_optical_depths = \
            self.__calculate_multisize_hyperspectral_total_optical_depths()
        self.multisize_hyperspectral_scattering_optical_depths = \
            self.__calculate_multisize_hyperspectral_scattering_optical_depths()
        self.hyperspectral_total_optical_depths = self.__reduce_total_optical_depths_size_dim()
        self.hyperspectral_scattering_optical_depths = self.__reduce_scattering_optical_depths_size_dim()

    def __check_input_types(self):
        assert isinstance(self.aerosol, Aerosol), 'aerosol needs to be an instance of Aerosol.'
        assert isinstance(self.layers, Layers), 'layers needs to be an instance of Layers.'
        assert isinstance(self.H, (int, float)), 'aerosol_scale_height needs to be an int or float.'
        assert isinstance(self.nu, (int, float)), 'conratu_nu needs to be an int or float.'
        self.__check_conrath_parameters_do_not_suck()
        assert isinstance(self.particle_sizes, np.ndarray), 'particle_sizes needs to be a numpy array.'
        assert isinstance(self.column_integrated_optical_depths, np.ndarray), \
            'column_integrated_optical_depths needs to be a numpy array'
        assert len(self.particle_sizes) == len(self.column_integrated_optical_depths), \
            'particle_sizes and column_integrated_optical_depths need to be the same length.'

    def __check_conrath_parameters_do_not_suck(self):
        if np.isnan(self.H):
            print('You wily bastard, you input a nan as the Conrath dust scale height. Fix that.')
        elif np.isnan(self.nu):
            print('You wily bastard, you input a nan as the Conrath nu parameter. Fix that.')
        if self.H < 0:
            raise SystemExit('Bad Conrath aerosol scale height: it cannot be negative. Fix...')
        elif self.nu < 0:
            raise SystemExit('Bad Conrath nu parameter: it cannot be negative. Fix...')

    def __make_conrath_profile(self):
        """Calculate the vertical dust distribution assuming a Conrath profile, i.e.
        q(z) / q(0) = exp( nu * (1 - exp(z / H)))
        where q is the mass mixing ratio

        Returns
        -------
        fractional_mixing_ratio: np.ndarray (len(altitude_layer))
            The fraction of the mass mixing ratio at the midpoint altitudes
        """

        fractional_mixing_ratio = np.exp(self.nu * (1 - np.exp(self.layers.altitude_layers / self.H)))
        return fractional_mixing_ratio

    def __calculate_multisize_hyperspectral_total_optical_depths(self, optical_depth_minimum=10**-7):
        vertical_mixing_ratio = self.__make_conrath_profile()
        dust_scaling = np.sum(self.layers.column_density_layers * vertical_mixing_ratio)
        multisize_hyperspectral_total_optical_depths = np.multiply.outer(
            np.outer(vertical_mixing_ratio * self.layers.column_density_layers, self.column_integrated_optical_depths),
            self.aerosol.extinction_ratios) / dust_scaling

        # Make sure each grid point is at least optical_depth_minimum
        return np.where(multisize_hyperspectral_total_optical_depths < optical_depth_minimum, optical_depth_minimum,
                        multisize_hyperspectral_total_optical_depths)

    def __calculate_multisize_hyperspectral_scattering_optical_depths(self):
        return self.multisize_hyperspectral_total_optical_depths * self.aerosol.hyperspectral_single_scattering_albedos

    def __reduce_total_optical_depths_size_dim(self):
        return np.sum(self.multisize_hyperspectral_total_optical_depths, axis=1)

    def __reduce_scattering_optical_depths_size_dim(self):
        return np.average(self.multisize_hyperspectral_scattering_optical_depths, axis=1,
                          weights=self.column_integrated_optical_depths)
