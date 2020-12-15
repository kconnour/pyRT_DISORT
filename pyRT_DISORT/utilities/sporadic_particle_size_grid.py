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
        altitude_grid: np.ndarray
            The input altitude grid
        particle_size_grid: np.ndarray
            The input particle size grid
        model_grid: ModelGrid
            The input model grid
        regridded_particle_sizes: np.ndarray
            The regridded particle sizes at the altitude layers in model_grid
        """
        self.altitude_grid = altitude_grid
        self.particle_size_grid = particle_size_grid
        self.model_grid = model_grid

        self.__check_grids_are_physical()

        self.regridded_particle_sizes = self.__interp_particle_sizes_to_model_grid()

    def __check_grids_are_physical(self):
        self.__check_altitude_grid_is_physical()
        self.__check_particle_size_grid_is_physical()
        self.__check_altitude_grid_size_grid_have_same_shape()
        self.__check_model_grid_is_ModelGrid()

    def __check_altitude_grid_is_physical(self):
        altitude_checker = ArrayChecker(self.altitude_grid, 'altitude_grid')
        altitude_checker.check_object_is_array()
        altitude_checker.check_ndarray_is_numeric()
        altitude_checker.check_ndarray_is_non_negative()
        altitude_checker.check_ndarray_is_finite()
        altitude_checker.check_ndarray_is_1d()
        altitude_checker.check_1d_array_is_monotonically_decreasing()

    def __check_particle_size_grid_is_physical(self):
        size_checker = ArrayChecker(self.particle_size_grid, 'particle_size_grid')
        size_checker.check_object_is_array()
        size_checker.check_ndarray_is_numeric()
        size_checker.check_ndarray_is_positive_finite()
        size_checker.check_ndarray_is_1d()

    def __check_altitude_grid_size_grid_have_same_shape(self):
        if self.altitude_grid.shape != self.particle_size_grid.shape:
            raise ValueError('altitude_grid and particle_size_grid must have the same shape')

    def __check_model_grid_is_ModelGrid(self):
        if not isinstance(self.model_grid, ModelGrid):
            raise TypeError('model_grid must be an instance of ModelGrid')

    def __interp_particle_sizes_to_model_grid(self):
        # I must flip these since numpy.interp expects monotonically increasing xp
        return np.interp(self.model_grid.layer_altitudes, np.flip(self.altitude_grid), np.flip(self.particle_size_grid))
