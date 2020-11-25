# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.preprocessing.model.aerosol import Aerosol
from pyRT_DISORT.preprocessing.model.atmosphere import ModelGrid
from pyRT_DISORT.preprocessing.utilities.array_checks import CheckArray


class Column:
    """Create an aerosol column to hold the aerosol's properties in each layer.  """
    def __init__(self, aerosol, model_grid, profiles, column_integrated_optical_depths):
        """
        Parameters
        ----------
        aerosol: Aerosol
            The aerosol for which this class will construct columns
        model_grid: ModelGrid
            The model atmosphere
        profiles: np.ndarray
            The vertical profiles at each of the particle sizes
        column_integrated_optical_depths: np.ndarray
            The column-integrated optical depth at the effective radii
        """
        self.aerosol = aerosol
        self.profiles = profiles
        self.model_grid = model_grid
        self.column_integrated_optical_depths = column_integrated_optical_depths
        self.__check_inputs()
        self.multisize_total_optical_depth = self.__calculate_multisize_total_optical_depths()
        self.multisize_scattering_optical_depth = self.__calculate_multisize_scattering_optical_depths()
        self.total_optical_depth = self.__calculate_total_optical_depths()
        self.scattering_optical_depth = self.__calculate_scattering_optical_depths()

    def __check_inputs(self):
        if not isinstance(self.aerosol, Aerosol):
            raise TypeError('aerosol needs to be an instance of Aerosol.')
        if not isinstance(self.profiles, np.ndarray):
            raise TypeError('profiles must be a np.ndarray')
        if not isinstance(self.model_grid, ModelGrid):
            raise TypeError('model_grid must be an instance of ModelGrid')
        column_od_checker = CheckArray(self.column_integrated_optical_depths, 'column_integrated_optical_depths')
        column_od_checker.check_object_is_array()
        column_od_checker.check_ndarray_is_numeric()
        column_od_checker.check_ndarray_is_positive_finite()
        if len(self.column_integrated_optical_depths) != len(self.aerosol.particle_sizes):
            raise ValueError('column_integrated_optical_depths must be the same length as particle_sizes in Aerosol')

    def __calculate_multisize_total_optical_depths(self):
        # Array shapes for this mess, in case I ever need to modify this:
        # self.profiles [nlayers, nsizes]
        # self.model_atmosphere.column_density_layers [nlayers]
        # self.columnODs [nsizes]
        # self.normalization_factor [nsizes]
        # self.aerosol.extinction [nsizes, nwavelengths]
        normalization_factor = np.sum((self.profiles.T * self.model_grid.column_density_layers).T, axis=0)
        dummy_var = self.profiles * self.column_integrated_optical_depths / normalization_factor
        dvar = (dummy_var.T * self.model_grid.column_density_layers).T
        return dvar[:, :, np.newaxis] * self.aerosol.extinction[np.newaxis, :, :]

    def __calculate_multisize_scattering_optical_depths(self):
        return self.multisize_total_optical_depth * self.aerosol.single_scattering_albedo

    def __calculate_total_optical_depths(self):
        return np.sum(self.multisize_total_optical_depth, axis=1)

    def __calculate_scattering_optical_depths(self):
        return np.average(self.multisize_scattering_optical_depth, axis=1,
                          weights=self.column_integrated_optical_depths)
