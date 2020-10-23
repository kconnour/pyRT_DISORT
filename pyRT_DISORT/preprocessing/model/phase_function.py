# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.preprocessing.model.aerosol_column import Column

# Note: by "phase functions" I usually mean the Legendre coefficients of the phase function


class EmpiricalPhaseFunctions:
    def __init__(self, phase_function_file, particle_sizes_file='', wavelengths_file=''):
        self.phase_function_file = phase_function_file
        self.particle_sizes_file = particle_sizes_file
        self.wavelengths_file = wavelengths_file
        self.__assert_inputs_are_good()

        self.phase_functions = self.__read_in_files(self.phase_function_file)
        self.particle_sizes = self.__read_in_files(self.particle_sizes_file)
        self.wavelengths = self.__read_in_files(self.wavelengths_file)
        self.__assert_arrays_are_good()

    def __assert_inputs_are_good(self):
        assert isinstance(self.phase_function_file, str), 'phase_function_file needs to be a string'
        assert isinstance(self.particle_sizes_file, str), 'particle_size_files needs to be a string'
        assert isinstance(self.wavelengths_file, str), 'wavelengths_file needs to be a string'

    @staticmethod
    def __read_in_files(file):
        if file:
            return np.load(file)
        else:
            return None

    def __assert_arrays_are_good(self):
        if np.ndim(self.phase_functions) == 2:
            if not self.particle_sizes:
                assert len(self.particle_sizes) == self.phase_functions.shape[1], \
                    'The shape of radii doesn\'t match the phase function dimension.'
            elif not self.wavelengths:
                assert len(self.wavelengths) == self.phase_functions.shape[1], \
                    'The shape of wavelengths doesn\'t match the phase function dimension.'
        elif np.ndim(self.phase_functions) == 3:
            assert len(self.particle_sizes) == self.phase_functions.shape[1], \
                'The shape of radii doesn\'t match the phase function dimension.'
            assert len(self.wavelengths) == self.phase_functions.shape[2], \
                'The shape of wavelengths doesn\'t match the phase function dimension.'

    def get_phase_function(self):
        return self.phase_functions


class NearestNeighborEmpiricalPhaseFunctions:
    def __init__(self, empirical_phase_function, column):
        self.epf = empirical_phase_function
        self.column = column
        assert isinstance(self.epf, EmpiricalPhaseFunctions), 'input needs to be an instance of EmpiricalPhaseFunction'
        assert isinstance(self.column, Column), 'column needs to be an instance of Column'
        self.nearest_neighbor_phase_functions = self.__get_nearest_neighbor_phase_functions()

    def __get_nearest_neighbor_phase_functions(self):
        radius_indices = self.__get_nearest_indices(self.column.particle_sizes, self.epf.particle_sizes)
        wavelength_indices = self.__get_nearest_indices(self.column.aerosol.wavelengths, self.epf.wavelengths)
        all_phase_functions = self.epf.phase_functions

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

    def get_phase_function(self):
        return self.nearest_neighbor_phase_functions


class ResizedPhaseFunctions:
    def __init__(self, phase_functions, n_moments):
        self.phase_functions = phase_functions.get_phase_function()
        self.n_moments = n_moments
        self.__assert_inputs_are_good()

        self.__phase_function_matching_moments = self.__match_n_moments()
        self.normalized_phase_function = self.__normalize_phase_functions()

    def __assert_inputs_are_good(self):
        #assert isinstance(self.phase_functions, (EmpiricalPhaseFunctions, NearestNeighborEmpiricalPhaseFunctions))
        assert isinstance(self.n_moments, int), 'n_moments needs to be an int'

    def __match_n_moments(self):
        if self.phase_functions.shape[0] < self.n_moments:
            return self.__add_moments()
        else:
            return self.__trim_moments()

    def __add_moments(self):
        starting_inds = np.linspace(self.phase_functions.shape[0], self.phase_functions.shape[0],
                                    num=self.n_moments - self.phase_functions.shape[0], dtype=int)
        return np.insert(self.phase_functions, starting_inds, 0, axis=0)

    def __trim_moments(self):
        return self.phase_functions[:self.n_moments]

    def __normalize_phase_functions(self):
        # Divide the k-th moment by 2k+1
        normalization = np.linspace(0, self.n_moments-1, num=self.n_moments)*2 + 1
        return (self.__phase_function_matching_moments.T / normalization).T


class StaticEmpiricalPhaseFunction:
    def __init__(self, phase_function_file, column, n_moments, particle_sizes_file='', wavelengths_file=''):
        self.phase_function_file = phase_function_file
        self.column = column
        self.n_moments = n_moments
        self.particle_sizes_file = particle_sizes_file
        self.wavelengths_file = wavelengths_file
        self.__rn_phase_function = self.__resize_and_normalize()
        self.__expanded_moments = self.__expand_layers()
        self.phase_function = self.__get_scattering_moments()

    def __resize_and_normalize(self):
        static_phase_function = EmpiricalPhaseFunctions(self.phase_function_file,
                                                        particle_sizes_file=self.particle_sizes_file,
                                                        wavelengths_file=self.wavelengths_file)
        resized_phase_function = ResizedPhaseFunctions(static_phase_function, self.n_moments)
        return resized_phase_function.normalized_phase_function

    def __expand_layers(self):
        return np.broadcast_to(self.__rn_phase_function[:, None, None], (self.n_moments, self.column.layers.n_layers,
                                                                      len(self.column.aerosol.wavelengths)))

    def __get_scattering_moments(self):
        return self.__expanded_moments * self.column.hyperspectral_scattering_optical_depths


class HyperradialHyperspectralEmpiricalPhaseFunction:
    def __init__(self, phase_function_file, column, n_moments, particle_sizes_file='', wavelengths_file=''):
        self.phase_function_file = phase_function_file
        self.column = column
        self.n_moments = n_moments
        self.particle_sizes_file = particle_sizes_file
        self.wavelengths_file = wavelengths_file
        self.__rn_phase_function = self.resize_and_normalize()
        self.hyperspectral_hyperradial_expanded_pf = self.expand_layers()
        self.hyperspectral_expanded_pf = self.__weighted_sum_over_size()

    def resize_and_normalize(self):
        static_phase_function = EmpiricalPhaseFunctions(self.phase_function_file,
                                                        particle_sizes_file=self.particle_sizes_file,
                                                        wavelengths_file=self.wavelengths_file)
        nn_phase_function = NearestNeighborEmpiricalPhaseFunctions(static_phase_function, self.column)
        resized_phase_function = ResizedPhaseFunctions(nn_phase_function, self.n_moments)
        return resized_phase_function.normalized_phase_function

    def expand_layers(self):
        rnpf = self.__rn_phase_function
        expanded_phase_function = np.broadcast_to(rnpf[:, None, :, :], (self.n_moments, self.column.layers.n_layers,
                                                                        rnpf.shape[1], rnpf.shape[2]))
        return expanded_phase_function

    def __weighted_sum_over_size(self):
        # Calculate C_sca / C_ext * tau_aerosol * PMOM_aerosol and weight its sum over size
        aerosol_polynomial_moments = self.hyperspectral_hyperradial_expanded_pf * \
                                     self.column.multisize_hyperspectral_scattering_optical_depths
        return np.average(aerosol_polynomial_moments, axis=2, weights=self.column.column_integrated_optical_depths)
