# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.preprocessing.model.aerosol_column import Column


class PhaseFunction:
    def __init__(self, column):
        self.column = column
        self.__check_input()

    def __check_input(self):
        assert isinstance(self.column, Column), 'column must be an instance of Column'

    def expand_dimensions(self, coefficients):
        """ Expand the dimension of input coefficients to be (n_moments, n_sizes, n_wavelengths)

        Parameters
        ----------
        coefficients: np.ndarray
            A 1D, 2D, or 3D array of coefficients

        Returns
        -------
        coefficients: np.ndarray
            A 3D array of the coefficients
        """
        # Define particle_sizes and wavelengths here if they don't exist... not good practice
        if np.ndim(coefficients) == 1:
            self.particle_sizes = np.linspace(0, 0, num=len(self.column.particle_sizes))
            self.wavelengths = np.linspace(0, 0, num=len(self.column.aerosol.wavelengths))
            return np.broadcast_to(coefficients[:, None, None], (coefficients.shape[0], 1, 1))
        elif np.ndim(coefficients) == 2:
            if self.particle_sizes is not None:
                self.wavelengths = np.linspace(0, 0, num=len(self.column.aerosol.wavelengths))
                return np.broadcast_to(coefficients[:, :, None], (coefficients.shape[0], coefficients.shape[1], 1))
            elif self.wavelengths is not None:
                self.particle_sizes = np.linspace(0, 0, num=len(self.column.particle_sizes))
                return np.broadcast_to(coefficients[:, None, :], (coefficients.shape[0], 1, coefficients.shape[1]))
        elif np.ndim(coefficients) == 3:
            return coefficients

    @staticmethod
    def normalize_coefficients(coefficients):
        """ Normalize the coefficients (divide the k-th moment by 2k + 1)

        Parameters
        ----------
        coefficients: np.ndarray
            A 3D array of the coefficients

        Returns
        -------
        np.ndarray of the normalized coefficients
        """
        n_moments = coefficients.shape[0]
        normalization = np.linspace(0, n_moments-1, num=n_moments)*2 + 1
        return (coefficients.T / normalization).T

    def expand_layers(self, coefficients):
        """ Expand coefficients to include a layer dimension

        Parameters
        ----------
        coefficients: np.ndarray
            A 3D numpy array of Legendre coefficients

        Returns
        -------
        np.ndarray: 4D coefficients
        """
        return np.broadcast_to(coefficients[:, None, :, :], (coefficients.shape[0], self.column.layers.n_layers,
                                                             coefficients.shape[1], coefficients.shape[2]))

    def sum_over_size(self, coefficients):
        """ Perform a weighted sum over the coefficients' size dimension

        Parameters
        ----------
        coefficients: np.ndarray
            A 3D array of Legendre coefficients

        Returns
        -------
        np.ndarray: 3D coefficients
        """
        aerosol_polynomial_moments = coefficients * self.column.multisize_hyperspectral_scattering_optical_depths
        return np.average(aerosol_polynomial_moments, axis=2, weights=self.column.column_integrated_optical_depths)


class HenyeyGreenstein(PhaseFunction):
    """ Make a Henyey-Greenstein phase function"""
    def __init__(self, column, asymmetry, n_moments=1000):
        """ Initialize the class

        Parameters
        ----------
        column: Column
            An aerosol column
        asymmetry: float
            The Henyey-Greenstein asymmetry parameter
        n_moments: int, optional
            The number of moments to make a HG phase function for. Default is 1000
        """
        super().__init__(column)
        self.asymmetry = asymmetry
        self.n_moments = n_moments
        self.__check_inputs()
        self.coefficients = self.__expand_coefficients()

    def __check_inputs(self):
        assert 0 <= self.asymmetry <= 1, 'the asymmetry parameter must be in [0, 1]'
        assert isinstance(self.n_moments, int), 'moments must be an int'

    def __make_legendre_coefficients(self):
        moments = np.linspace(0, self.n_moments - 1, num=self.n_moments)
        return (2 * moments + 1) * self.asymmetry ** moments

    def __expand_coefficients(self):
        coefficients = self.__make_legendre_coefficients()
        radial_spectral_coefficients = self.expand_dimensions(coefficients)
        normalized_rs_coefficients = self.normalize_coefficients(radial_spectral_coefficients)
        layered_coefficients = self.expand_layers(normalized_rs_coefficients)
        return self.sum_over_size(layered_coefficients)


class EmpiricalPhaseFunction(PhaseFunction):
    """ Construct an empirical phase function"""
    def __init__(self, column, phase_function_file, particle_sizes_file=None, wavelengths_file=None):
        """ Initialize the class

        Parameters
        ----------
        column: Column
            An aerosol column
        phase_function_file: str
            The complete path of the phase function file
        particle_sizes_file: str, optional
            The complete path of the particle sizes corresponding to the phase_function_file. Default is None
        wavelengths_file: str, optional
            The complete path of the wavelengths corresponding to the phase_function_file. Default is None
        """
        super().__init__(column)
        self.phase_function_file = phase_function_file
        self.particle_sizes_file = particle_sizes_file
        self.wavelengths_file = wavelengths_file
        self.__check_inputs()

        self.empirical_coefficients = self.__read_in_files(self.phase_function_file)
        self.particle_sizes = self.__read_in_files(self.particle_sizes_file)
        self.wavelengths = self.__read_in_files(self.wavelengths_file)
        self.__check_shapes_match()

        self.coefficients = self.__expand_coefficients()

    def __check_inputs(self):
        assert isinstance(self.phase_function_file, str), 'phase_function_file must be a string'
        assert isinstance(self.particle_sizes_file, (str, type(None))), 'particle_sizes_file must be a string'
        assert isinstance(self.wavelengths_file, (str, type(None))), 'wavelengths_file must be a string'

    @staticmethod
    def __read_in_files(file):
        if file:
            return np.load(file)

    def __check_shapes_match(self):
        if np.ndim(self.empirical_coefficients) == 3:
            if self.particle_sizes is None:
                raise SystemExit('You need to include particle size info!')
            elif self.wavelengths is None:
                raise SystemExit('You need to include wavelength info!')
            n_sizes = self.particle_sizes.shape[0]
            n_wavelengths = self.wavelengths.shape[0]
            if n_sizes != self.empirical_coefficients.shape[1]:
                raise SystemExit('The size dimension provided doesn\'t match the phase function file')
            elif n_wavelengths != self.empirical_coefficients.shape[2]:
                raise SystemExit('The wavelength dimension provided doesn\'t match the phase function file')
        elif np.ndim(self.empirical_coefficients) == 2:
            if self.particle_sizes is None and self.wavelengths is None:
                raise SystemExit('You included too little information... you need to specify one of particle_size or wavelengths')
            elif self.particle_sizes is not None and self.wavelengths is not None:
                raise SystemExit('You included too much information... you need to specify one of particle_size or wavelengths')
            if self.particle_sizes is None:
                n_wavelengths = self.wavelengths.shape[0]
                if n_wavelengths != self.empirical_coefficients.shape[1]:
                    raise SystemExit('The wavelength dimension provided doesn\'t match the phase function file')
            elif self.wavelengths is None:
                n_sizes = self.particle_sizes.shape[0]
                if n_sizes != self.empirical_coefficients.shape[1]:
                    raise SystemExit('The size dimension provided doesn\'t match the phase function file')
        elif np.ndim(self.empirical_coefficients) == 1:
            if self.particle_sizes is not None:
                raise SystemExit('If you only have phase functions independent of particle sizes, don\'t include sizes')
            elif self.wavelengths is not None:
                raise SystemExit('If you only have phase functions independent of wavelengths, don\'t include wavelengths')

    def __expand_coefficients(self):
        radial_spectral_coefficients = self.expand_dimensions(self.empirical_coefficients)
        nearest_neighbor_rs_coefficients = self.__get_nearest_neighbor_phase_functions(radial_spectral_coefficients)
        normalized_rs_coefficients = self.normalize_coefficients(nearest_neighbor_rs_coefficients)
        layered_coefficients = self.expand_layers(normalized_rs_coefficients)
        return self.sum_over_size(layered_coefficients)

    def __get_nearest_neighbor_phase_functions(self, coefficients):
        radius_indices = self.__get_nearest_indices(self.column.particle_sizes, self.particle_sizes)
        wavelength_indices = self.__get_nearest_indices(self.column.aerosol.wavelengths, self.wavelengths)
        return coefficients[:, radius_indices, :][:, :, wavelength_indices]

    @staticmethod
    def __get_nearest_indices(values, array):
        diff = (values.reshape(1, -1) - array.reshape(-1, 1))
        indices = np.abs(diff).argmin(axis=0)
        return indices
