# 3rd-party imports
import numpy as np

# Local imports
#from pyRT_DISORT.preprocessing.model.atmosphere import Layers
from pyRT_DISORT.preprocessing.model.atmosphere import ModelGrid
from pyRT_DISORT.preprocessing.utilities.external_files import ExternalFile
from pyRT_DISORT.preprocessing.model.aerosol_column import Column as NewColumn
from pyRT_DISORT.preprocessing.model.aerosol import Aerosol


class Column:
    """Create an aerosol column to hold the aerosol's properties in each layer. Right now it constructs a column
    using Conrath parameters. Will be extended to allow user input vertical distributions. """
    def __init__(self, aerosol, layers, profile, particle_sizes, column_integrated_optical_depths):
        """ Initialize the class
        Parameters
        ----------
        aerosol: Aerosol
            The aerosol for which this class will construct a column
        layers: Layers
            The equations of state in each of the layers in the plane parallel atmosphere
        particle_sizes: np.ndarray
            The effective radii of this aerosol [microns]
        column_integrated_optical_depths: np.ndarray
            The column-integrated optical depth at the effective radii
        """
        self.aerosol = aerosol
        self.layers = layers
        self.profile = profile
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
        assert isinstance(self.aerosol, OldAerosol), 'aerosol needs to be an instance of Aerosol.'
        assert isinstance(self.particle_sizes, np.ndarray), 'particle_sizes needs to be a numpy array.'
        assert isinstance(self.profile, (Conrath, GCMProfile)), 'profile needs to be Conrath or GCMProfile'
        assert isinstance(self.column_integrated_optical_depths, np.ndarray), \
            'column_integrated_optical_depths needs to be a numpy array'
        assert len(self.particle_sizes) == len(self.column_integrated_optical_depths), \
            'particle_sizes and column_integrated_optical_depths need to be the same length.'

    def __calculate_multisize_hyperspectral_total_optical_depths(self):
        vertical_mixing_ratio = self.profile.get_profile()

        dust_scaling = np.sum(self.layers.column_density_layers * vertical_mixing_ratio)
        print(dust_scaling)
        multisize_hyperspectral_total_optical_depths = np.multiply.outer(
            np.outer(vertical_mixing_ratio * self.layers.column_density_layers, self.column_integrated_optical_depths),
            self.aerosol.extinction_ratios) / dust_scaling

        return multisize_hyperspectral_total_optical_depths

    def __calculate_multisize_hyperspectral_scattering_optical_depths(self):
        return self.multisize_hyperspectral_total_optical_depths * self.aerosol.hyperspectral_single_scattering_albedos

    def __reduce_total_optical_depths_size_dim(self):
        return np.sum(self.multisize_hyperspectral_total_optical_depths, axis=1)

    def __reduce_scattering_optical_depths_size_dim(self):
        return np.average(self.multisize_hyperspectral_scattering_optical_depths, axis=1,
                          weights=self.column_integrated_optical_depths)


class Conrath:
    def __init__(self, model_atm, aerosol_scale_height, conrath_nu):
        self.H = aerosol_scale_height
        self.nu = conrath_nu
        self.layers = model_atm

        self.__check_input_types()
        self.__check_conrath_parameters_do_not_suck()

        self.vertical_profile = self.__make_conrath_profile()

    def __check_input_types(self):
        assert isinstance(self.H, (int, float)), 'aerosol_scale_height needs to be an int or float.'
        assert isinstance(self.nu, (int, float)), 'conratu_nu needs to be an int or float.'
        assert isinstance(self.layers, ModelGrid), 'layers needs to be an instance of Layers.'

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

    def get_profile(self):
        return self.vertical_profile


class GCMProfile:
    def __init__(self, layers, profile):
        self.layers = layers
        self.profile = profile

    def __check_input_types(self):
        assert isinstance(self.layers, ModelGrid), 'layers needs to be an instance of Layers'
        assert isinstance(self.profile, np.ndarray), 'profiles needs to be a numpy array'

    def get_profile(self):
        return self.profile


class OldAerosol:
    """ Create a class to hold all of the information about an aerosol"""

    def __init__(self, aerosol_file, wavelengths, reference_wavelength):
        """ Initialize the class to hold all the aerosol's properties
        Parameters
        ----------
        aerosol_file: np.ndarray
            An array containing the aerosol's properties
        wavelengths: np.ndarray
            The wavelengths at which this aerosol was observed
        reference_wavelength: float
            The wavelength at which to scale the wavelengths
        """

        self.aerosol_file = aerosol_file
        self.wavelengths = wavelengths
        self.reference_wavelength = reference_wavelength

        assert isinstance(self.aerosol_file, np.ndarray), 'aerosol_file must be a numpy array.'
        assert isinstance(self.wavelengths, np.ndarray), 'wavelengths needs to be a numpy array.'
        assert isinstance(self.reference_wavelength, float), 'reference_wavelength needs to be a float.'

        # Make sure the aerosol knows its properties
        self.wavelengths_quantities, self.c_extinction, self.c_scattering, self.kappa, self.g, \
            self.p_max, self.theta_max = self.__read_aerosol_file()
        self.__inform_if_outside_wavelength_range()
        self.extinction_ratios = self.__calculate_wavelength_extinction_ratios()
        self.hyperspectral_single_scattering_albedos = self.__calculate_hyperspectral_single_scattering_albedos()
        self.hyperspectral_asymmetry_parameters = self.__calculate_hyperspectral_asymmetry_parameters()

    def __read_aerosol_file(self):
        wavelengths = self.aerosol_file[:, 0]
        c_extinction = self.aerosol_file[:, 1]
        c_scattering = self.aerosol_file[:, 2]
        kappa = self.aerosol_file[:, 3]
        g = self.aerosol_file[:, 4]

        if self.aerosol_file.shape[1] == 5:
            p_max = np.array([])
            theta_max = np.array([])
        else:
            p_max = self.aerosol_file[:, 5]
            theta_max = self.aerosol_file[:, 6]

        return wavelengths, c_extinction, c_scattering, kappa, g, p_max, theta_max

    def __inform_if_outside_wavelength_range(self):
        if np.size((too_short := self.wavelengths[self.wavelengths < self.wavelengths_quantities[0]]) != 0):
            print('The following input wavelengths: {} microns are shorter than {:.3f} microns---the shortest '
                  'wavelength in the file. Using properties from that wavelength.'
                  .format(too_short, self.wavelengths_quantities[0]))
        if np.size((too_long := self.wavelengths[self.wavelengths > self.wavelengths_quantities[-1]]) != 0):
            print('The following input wavelengths: {} microns are longer than {:.1f} microns---the shortest '
                  'wavelength in the file. Using properties from that wavelength.'
                  .format(too_long, self.wavelengths_quantities[-1]))

    def __calculate_wavelength_extinction_ratios(self):
        reference_c_ext = np.interp(self.reference_wavelength, self.wavelengths_quantities, self.c_extinction)
        wavelengths_c_ext = np.interp(self.wavelengths, self.wavelengths_quantities, self.c_extinction)
        return wavelengths_c_ext / reference_c_ext

    def __calculate_hyperspectral_single_scattering_albedos(self):
        interpolated_extinction = np.interp(self.wavelengths, self.wavelengths_quantities, self.c_extinction)
        interpolated_scattering = np.interp(self.wavelengths, self.wavelengths_quantities, self.c_scattering)
        return interpolated_scattering / interpolated_extinction

    def __calculate_hyperspectral_asymmetry_parameters(self):
        return np.interp(self.wavelengths, self.wavelengths_quantities, self.g)


atmFile = ExternalFile('/home/kyle/repos/pyRT_DISORT/pyRT_DISORT/data/planets/mars/aux/mars_atm.npy')
ma = ModelGrid(atmFile.array, atmFile.array[:, 0])

dustFile = ExternalFile('/home/kyle/repos/pyRT_DISORT/pyRT_DISORT/data/planets/mars/aux/dust.npy')
wavs = np.array([1, 9.3])
oldAero = OldAerosol(dustFile.array, wavs, 9.3)
con = Conrath(ma, 10, 0.5)

#raise SystemExit(9)
dcol = Column(oldAero, ma, con, np.array([1]), np.array([1]))
print(np.sum(dcol.hyperspectral_total_optical_depths, axis=0))
print(np.sum(dcol.hyperspectral_scattering_optical_depths, axis=0))
