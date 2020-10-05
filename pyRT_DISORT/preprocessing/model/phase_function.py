# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.preprocessing.model.aerosol_column import Column


class EmpiricalPhaseFunction:
    def __init__(self, phase_function_file, particle_sizes_file, wavelengths_file):
        self.phase_function_file = phase_function_file
        self.particle_sizes_file = particle_sizes_file
        self.wavelengths_file = wavelengths_file

        self.phase_functions = np.load(self.phase_function_file)
        self.particle_sizes = np.load(self.particle_sizes_file)
        self.wavelengths = np.load(self.wavelengths_file)

        assert len(self.particle_sizes) == self.phase_functions.shape[1], \
            'The shape of radii doesn\'t match the phase function dimension.'
        assert len(self.wavelengths) == self.phase_functions.shape[2], \
            'The shape of wavelengths doesn\'t match the phase function dimension.'


class NearestNeighborPhaseFunction:
    def __init__(self, aerosol_phase_function, column, n_moments):
        self.aerosol_phase_function = aerosol_phase_function
        self.column = column
        self.n_moments = n_moments

        assert isinstance(self.aerosol_phase_function, EmpiricalPhaseFunction)
        assert isinstance(self.column, Column), 'column must be a Column.'
        assert isinstance(self.n_moments, int), 'n_moments but be an int.'

        self.nearest_neighbor_phase_functions = self.__get_nearest_neighbor_phase_functions()
        self.__make_nearest_neighbor_phase_functions_match_n_moments()
        self.__normalize_nearest_neighbor_phase_functions()
        self.layered_nearest_neighbor_phase_functions = self.__expand_nearest_neighbor_phase_function_layers()
        self.layered_hyperspectral_nearest_neighbor_phase_functions = self.__make_phase_function_without_size()

    def __get_nearest_neighbor_phase_functions(self):
        radius_indices = self.__get_nearest_indices(self.column.particle_sizes,
                                                  self.aerosol_phase_function.particle_sizes)
        wavelength_indices = self.__get_nearest_indices(self.column.aerosol.wavelengths,
                                                      self.aerosol_phase_function.wavelengths)
        all_phase_functions = self.aerosol_phase_function.phase_functions

        # This solution reads awfully
        # https://stackoverflow.com/questions/35607818/index-a-2d-numpy-array-with-2-lists-of-indices
        #moments_inds = np.linspace(0, self.n_moments-1, num=self.n_moments, dtype=int)
        #nearest_neighbor_phase_functions = all_phase_functions[np.ix_(moments_inds, radius_indices, wavelength_indices)]

        # This solution reads cleaner but I'm not sure why I cannot do this on one line...
        nearest_neighbor_phase_functions = all_phase_functions[:, radius_indices, :]
        nearest_neighbor_phase_functions = nearest_neighbor_phase_functions[:, :, wavelength_indices]
        return nearest_neighbor_phase_functions

    @staticmethod
    def __get_nearest_indices(values, array):
        diff = (values.reshape(1, -1) - array.reshape(-1, 1))
        indices = np.abs(diff).argmin(axis=0)
        return indices

    def __make_nearest_neighbor_phase_functions_match_n_moments(self):
        if self.nearest_neighbor_phase_functions.shape[0] < self.n_moments:
            self.__add_moments()
        else:
            self.__trim_moments()

    def __add_moments(self):
        nnpf_shapes = self.nearest_neighbor_phase_functions.shape
        moments = np.zeros((self.n_moments, nnpf_shapes[1], nnpf_shapes[2]))
        moments[:nnpf_shapes[0], :, :] = self.nearest_neighbor_phase_functions
        self.nearest_neighbor_phase_functions = moments

    def __trim_moments(self):
        self.nearest_neighbor_phase_functions = self.nearest_neighbor_phase_functions[:self.n_moments, :, :]

    def __normalize_nearest_neighbor_phase_functions(self):
        # Divide the k-th moment by 2k+1
        normalization = np.linspace(0, self.n_moments-1, num=self.n_moments)*2 + 1
        self.nearest_neighbor_phase_functions = (self.nearest_neighbor_phase_functions.T / normalization).T

    def __expand_nearest_neighbor_phase_function_layers(self):
        nnpf = self.nearest_neighbor_phase_functions
        expanded_phase_function = np.broadcast_to(nnpf[:, None, :, :], (self.n_moments, self.column.layers.n_layers,
                                                                        nnpf.shape[1], nnpf.shape[2]))
        return expanded_phase_function

    def __make_phase_function_without_size(self):
        # Calculate C_sca / C_ext * tau_aerosol * PMOM_aerosol and weight its sum over size
        aerosol_polynomial_moments = self.layered_nearest_neighbor_phase_functions * \
                                     self.column.multisize_hyperspectral_scattering_optical_depths
        return np.average(aerosol_polynomial_moments, axis=2, weights=self.column.column_integrated_optical_depths)
