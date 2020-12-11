# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.preprocessing.model.atmosphere import ModelGrid
from pyRT_DISORT.preprocessing.utilities.array_checks import ArrayChecker


class SporadicParticleSizes:
    """ This class interpolates particle sizes on a sporadic altitude grid to the model altitude grid"""
    def __init__(self, z_size_grid, model_grid):
        """
        Parameters
        ----------
        z_size_grid: np.ndarray
            An Nx2 array where the first column are altitudes and the second column are the associated particle sizes
        model_grid: ModelGrid
            Model structure to interpolate particle sizes on to

        Attributes
        ----------
        z_size_grid: np.ndarray
            The input z_size_grid
        model_grid: ModelGrid
            The input model_grid
        regridded_particle_sizes: np.ndarray
            The particle sizes regridded to match the number of layers in model_grid
        """
        self.z_size_grid = z_size_grid
        self.model_grid = model_grid

        self.__check_grids()

        self.regridded_particle_sizes = self.__interp_to_new_grid()

    def __check_grids(self):
        z_size_checker = ArrayChecker(self.z_size_grid, 'z_size_grid')
        z_size_checker.check_object_is_array()
        z_size_checker.check_ndarray_is_numeric()
        z_size_checker.check_ndarray_is_non_negative()
        z_size_checker.check_ndarray_is_finite()
        z_size_checker.check_ndarray_is_2d()
        altitude_checker = ArrayChecker(self.z_size_grid[:, 0], 'z_size_grid')
        altitude_checker.check_1d_array_is_monotonically_decreasing()
        if self.z_size_grid.shape[-1] != 2:
            raise IndexError('The second dimension of z_size_grid must be 2')
        if not isinstance(self.model_grid, ModelGrid):
            raise TypeError('model_grid must be an instance of ModelGrid')

    def __interp_to_new_grid(self):
        return np.interp(self.model_grid.altitude_layers, np.flip(self.z_size_grid[:, 0]),
                         np.flip(self.z_size_grid[:, 1]))
