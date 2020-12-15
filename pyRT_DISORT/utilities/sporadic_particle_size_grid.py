# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.model_atmosphere.atmosphere_grid import ModelGrid
from pyRT_DISORT.utilities.array_checks import ArrayChecker


class SporadicParticleSizes:
    """ A SporadicParticleSizes object holds altitude info of particle sizes and can interpolate them onto a model
    grid"""

    def __init__(self, altitude_grid, particle_size_grid, model_grid):
        """
        Parameters
        ----------
        altitude_grid: np.ndarray
            1D array of altitudes where particle sizes are known
        particle_size_grid: np.ndarray
            1D array of particle sizes
        model_grid: ModelGrid
            Model structure to interpolate particle sizes on to

        Attributes
        ----------

        """
        self.altitude_grid = altitude_grid
        self.particle_size_grid = particle_size_grid
        self.model_grid = model_grid

        #self.__check_grids()

        #self.regridded_particle_sizes = self.__interp_to_new_grid()

    def __check_grids_are_physical(self):
        self.__check_altitude_grid_is_physical()

    def __check_altitude_grid_is_physical(self):
        altitude_checker = ArrayChecker(self.altitude_grid, 'altitude_grid')
        altitude_checker.check_object_is_array()
        altitude_checker.check_ndarray_is_numeric()
        altitude_checker.check_ndarray_is_non_negative()
        altitude_checker.check_ndarray_is_finite()
        altitude_checker.check_ndarray_is_1d()

    def __check_particle_size_grid_is_physical(self):

    '''def __check_grids(self):

        altitude_checker = ArrayChecker(self.z_size_grid[:, 0], 'z_size_grid')
        altitude_checker.check_1d_array_is_monotonically_decreasing()
        if self.z_size_grid.shape[-1] != 2:
            raise IndexError('The second dimension of z_size_grid must be 2')
        if not isinstance(self.model_grid, ModelGrid):
            raise TypeError('model_grid must be an instance of ModelGrid')

    def __interp_to_new_grid(self):
        return np.interp(self.model_grid.altitude_layers, np.flip(self.z_size_grid[:, 0]),
                         np.flip(self.z_size_grid[:, 1]))
'''