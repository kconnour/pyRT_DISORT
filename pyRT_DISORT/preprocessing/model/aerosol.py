# 3rd-party imports
import numpy as np


class AerosolProperties:
    def __init__(self, aerosol_properties, particle_size_grid=None, wavelength_grid=None):
        self.aerosol_properties = aerosol_properties
        self.particle_size_grid = particle_size_grid
        self.wavelength_grid = wavelength_grid

        self.__check_input_types_and_dimensions()

        self.__c_ext_ind = 0    # I define these 3 because I may loosen the ordering restriction in the future
        self.__c_sca_ind = 1
        self.__g_ind = 2
        self.c_extinction, self.c_scattering, self.g = self.__read_aerosol_file()
        self.__check_aerosol_properties_are_plausible()

    def __check_input_types_and_dimensions(self):
        self.__check_inputs_are_arrays()
        self.__check_arrays_are_numeric()
        self.__check_grids_are_plausible()
        self.__check_input_array_dimensions()
        self.__check_grids_match_aerosol_dimensions()

    def __check_inputs_are_arrays(self):
        assert isinstance(self.aerosol_properties, np.ndarray), 'aerosol_file must be a np.ndarray'
        assert isinstance(self.particle_size_grid, (np.ndarray, type(None))), 'particle_size_grid must be a np.ndarray'
        assert isinstance(self.wavelength_grid, (np.ndarray, type(None))), 'wavelength_grid must be a np.ndarray'

    def __check_arrays_are_numeric(self):
        assert np.issubdtype(self.aerosol_properties.dtype, np.number), 'aerosol_properties must only contain numbers'
        if self.particle_size_grid is not None:
            assert np.issubdtype(self.particle_size_grid.dtype, np.number), \
                'particle_size_grid must only contain numbers'
        if self.wavelength_grid is not None:
            assert np.issubdtype(self.wavelength_grid.dtype, np.number), 'wavelength_grid must only contain numbers'

    def __check_grids_are_plausible(self):
        if self.particle_size_grid is not None:
            assert np.all(np.isfinite(self.particle_size_grid)), 'particle_size_grid contains non-finite values'
            assert np.all(self.particle_size_grid > 0), 'particle_size_grid contains non-positive sizes'
        if self.wavelength_grid is not None:
            assert np.all(np.isfinite(self.wavelength_grid)), 'wavelength_grid contains non-finite values'
            assert np.all(self.wavelength_grid > 0), 'wavelength_grid contains non-positive wavelengths'

    def __check_input_array_dimensions(self):
        assert np.ndim(self.aerosol_properties) == 2 or np.ndim(self.aerosol_properties) == 3, \
            'aerosol_file must be a 2D or 3D array'
        assert np.shape(self.aerosol_properties)[-1] == 3, \
            'aerosol_properties must be an (M)xNx3 array for the 3 parameters'
        if self.particle_size_grid is not None:
            assert np.ndim(self.particle_size_grid) == 1, 'particle_size_grid must be a 1D array'
        if self.wavelength_grid is not None:
            assert np.ndim(self.wavelength_grid) == 1, 'wavelength_grid must be a 1D array'

    def __check_grids_match_aerosol_dimensions(self):
        if np.ndim(self.aerosol_properties) == 2:
            assert (self.particle_size_grid is not None) ^ (self.wavelength_grid is not None), \
                'For 2D aerosol_properties, provide one and only one of particle_size_grid and wavelength_grid'
            if self.particle_size_grid is not None:
                assert self.aerosol_properties.shape[0] == len(self.particle_size_grid), \
                    'For 2D files, aerosol_properties\' first dimension must be the same length as particle_size_grid'
            if self.wavelength_grid is not None:
                assert self.aerosol_properties.shape[0] == len(self.wavelength_grid), \
                    'For 2D files, aerosol_properties\' first dimension must be the same length as wavelength_gird'
        if np.ndim(self.aerosol_properties) == 3:
            assert (self.particle_size_grid is not None) and (self.wavelength_grid is not None), \
                'You need to include both particle_size_grid and wavelength_grid'
            assert self.aerosol_properties.shape[0] == len(self.particle_size_grid), \
                'For 3D files, aerosol_properties\' first dimension must be the same length as particle_size_grid'
            assert self.aerosol_properties.shape[1] == len(self.wavelength_grid), \
                'For 3D files, aerosol_properties\' second dimension must be the same length as wavelength_grid'

    def __read_aerosol_file(self):
        c_extinction = np.take(self.aerosol_properties, self.__c_ext_ind, axis=-1)
        c_scattering = np.take(self.aerosol_properties, self.__c_sca_ind, axis=-1)
        g = np.take(self.aerosol_properties, self.__g_ind, axis=-1)
        return c_extinction, c_scattering, g

    def __check_aerosol_properties_are_plausible(self):
        assert np.all(np.isfinite(self.c_extinction)), 'c_extinction contains non-finite values'
        assert np.all(np.isfinite(self.c_scattering)), 'c_scattering contains non-finite values'
        assert np.all(self.c_extinction >= 0), 'c_extinction contains negative values'
        assert np.all(self.c_scattering >= 0), 'c_scattering contains negative values'
        assert np.all(self.g > 0) and np.all(self.g < 1), 'g must be in range [0, 1]'


class Aerosol(AerosolProperties):
    """ Create a class to hold all of the information about an aerosol"""

    def __init__(self, aerosol_properties, wavelengths, particle_sizes, reference_wavelengths, particle_size_grid=None, wavelength_grid=None):
        """ Initialize the class to hold all the aerosol's properties

        Parameters
        ----------
        aerosol_properties: np.ndarray
            An array containing the aerosol's properties
        wavelengths: np.ndarray
            The wavelengths at which this aerosol was observed
        particle_sizes: np.ndarray
            The particle sizes of this aerosol
        reference_wavelengths: np.ndarray
            The wavelength at which to scale the wavelengths
        """
        super().__init__(aerosol_properties, particle_size_grid=particle_size_grid, wavelength_grid=wavelength_grid)
        self.wavelengths = wavelengths
        self.particle_sizes = particle_sizes
        self.reference_wavelengths = reference_wavelengths

        self.__check_input_types()
        self.__check_inputs()
        self.__inform_if_outside_wavelength_range()

        self.wavelength_extinction = self.__calculate_extinction(self.wavelengths)
        self.reference_extinction = self.__calculate_extinction(self.reference_wavelengths)
        self.extinction = self.__div_outer_product(self.wavelength_extinction, self.reference_extinction)
        self.wavelength_scattering = self.__calculate_scattering(self.wavelengths)
        self.single_scattering_albedo = self.wavelength_scattering / self.wavelength_extinction
        self.asymmetry_parameter = self.__calculate_asymmetry_parameter(self.wavelengths)

    def __check_input_types(self):
        assert isinstance(self.aerosol_file, np.ndarray), 'aerosol_file must be a np.ndarray'
        assert isinstance(self.wavelengths, np.ndarray), 'wavelengths must to be a np.ndarray'
        assert isinstance(self.reference_wavelengths, np.ndarray), 'reference_wavelengths must be a np.ndarray'

    def __check_inputs(self):
        assert np.ndim(self.wavelengths) == 1, 'wavelengths must be 1D'
        assert np.ndim(self.particle_sizes) == 1, 'particle_sizes must be 1D'
        assert np.shape(self.particle_sizes) == np.shape(self.reference_wavelengths), \
            'particle_sizes and reference_wavelengths must have the same shape'

    def __inform_if_outside_wavelength_range(self):
        if np.size((too_short := self.wavelengths[self.wavelengths < self.wavelength_grid[0]]) != 0):
            print('The following input wavelengths: {} microns are shorter than {:.3f} microns---the shortest '
                  'wavelength in the file. Using properties from that wavelength.'
                  .format(too_short, self.wavelength_grid[0]))
        if np.size((too_long := self.wavelengths[self.wavelengths > self.wavelength_grid[-1]]) != 0):
            print('The following input wavelengths: {} microns are longer than {:.1f} microns---the shortest '
                  'wavelength in the file. Using properties from that wavelength.'
                  .format(too_long, self.wavelength_grid[-1]))

    def __calculate_extinction(self, array):
        return np.interp(array, self.wavelength_grid, self.c_extinction)

    def __calculate_scattering(self, array):
        return np.interp(array, self.wavelength_grid, self.c_scattering)

    def __calculate_asymmetry_parameter(self, array):
        return np.interp(array, self.wavelength_grid, self.g)

    @staticmethod
    def __div_outer_product(spectral, radial):
        return np.divide.outer(spectral, radial).T
