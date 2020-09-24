# Built-in imports
import os
import fnmatch as fnm

# 3rd-party imports
import numpy as np

# Local imports
from preprocessing.model.aerosol_column import Column


class EmpiricalPhaseFunction:
    def __init__(self, phase_function_file, radii_file, wavelength_file):
        self.phase_function_file = phase_function_file   # shape (n_radii, n_wavelengths, n_moments)
        self.radii_file = radii_file
        self.wavelength_file = wavelength_file

        self.phase_functions = np.load(self.phase_function_file)
        self.radii = np.load(self.radii_file)
        self.wavelengths = np.load(self.wavelength_file)

        assert len(self.radii) == self.phase_functions.shape[1], \
            'The shape of radii doesn\'t match the phase function.'
        assert len(self.wavelengths) == self.phase_functions.shape[2], \
            'The shape of wavelengths doesn\'t match the phase function.'


class NearestNeighborPhaseFunction:
    def __init__(self, all_phase_functions, column, n_moments):
        self.all_phase_functions = all_phase_functions
        self.column = column
        self.n_moments = n_moments

        assert isinstance(self.all_phase_functions, EmpiricalPhaseFunction)
        assert isinstance(self.column, Column)

        self.nearest_neighbor_phase_functions = self.get_nearest_neighbor_phase_functions()
        self.make_phase_function_match_n_moments()
        self.normalize_phase_function()
        self.expanded_nearest_neighbor_phase_functions = self.expand_phase_function_layers()
        self.aerosol_phase_function = self.make_phase_function_without_size()

    def get_nearest_neighbor_phase_functions(self):
        radius_indices = self.get_nearest_indices(self.column.effective_radii, self.all_phase_functions.radii)
        wavelength_indices = self.get_nearest_indices(self.column.aerosol.wavelengths, self.all_phase_functions.wavelengths)
        all_phase_functions = self.all_phase_functions.phase_functions

        # I'm not sure why I cannot do this on one line...
        closest_phase_functions = all_phase_functions[:, radius_indices, :]
        closest_phase_functions = closest_phase_functions[:, :, wavelength_indices]
        return closest_phase_functions

    @staticmethod
    def get_nearest_indices(values, array):
        diff = (values.reshape(1, -1) - array.reshape(-1, 1))
        indices = np.abs(diff).argmin(axis=0)
        return indices

    def make_phase_function_match_n_moments(self):
        if self.nearest_neighbor_phase_functions.shape[0] < self.n_moments:
            self.add_moments()
        else:
            self.trim_moments()

    def add_moments(self):
        nnpf = self.nearest_neighbor_phase_functions
        moments = np.zeros((self.n_moments, nnpf.shape[1], nnpf.shape[2]))
        moments[:nnpf.shape[0], :, :] = nnpf
        self.nearest_neighbor_phase_functions = moments

    def trim_moments(self):
        self.nearest_neighbor_phase_functions = self.nearest_neighbor_phase_functions[:self.n_moments, :, :]

    def normalize_phase_function(self):
        normalization = np.linspace(0, self.n_moments-1, num=self.n_moments)*2 + 1
        self.nearest_neighbor_phase_functions = (self.nearest_neighbor_phase_functions.T / normalization).T

    def expand_phase_function_layers(self):
        nnpf = self.nearest_neighbor_phase_functions
        expanded_phase_function = np.broadcast_to(nnpf[:, None, :, :], (self.n_moments, self.column.layers.n_layers,
                                                                        nnpf.shape[1], nnpf.shape[2]))
        return expanded_phase_function

    def make_phase_function_without_size(self):
        # Calculate C_sca / C_ext * tau_aerosol * PMOM_aerosol
        aerosol_polynomial_moments = self.expanded_nearest_neighbor_phase_functions * self.column.single_scattering_albedos_with_size
        return np.average(aerosol_polynomial_moments, axis=2, weights=self.column.column_integrated_optical_depths)
