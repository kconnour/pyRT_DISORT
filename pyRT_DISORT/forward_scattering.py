import numpy as np


class ForwardScatteringProperty:
    """An abstract class to check something is a forward scattering property.

    """

    def __init__(self, parameter: np.ndarray, particle_size: np.ndarray,
                 wavelength: np.ndarray, name: str) -> None:
        self.__parameter = parameter
        self.__particle_size = particle_size
        self.__wavelength = wavelength
        self.__name = name

        self.__raise_error_if_inputs_are_bad()

    def __raise_error_if_inputs_are_bad(self) -> None:
        self.__raise_type_error_if_parameter_is_not_ndarray()
        self.__raise_value_error_if_shapes_do_not_match()

    def __raise_type_error_if_parameter_is_not_ndarray(self) -> None:
        if not isinstance(self.__parameter, np.ndarray):
            message = f'{self.__name} must be a numpy.ndarray.'
            raise TypeError(message)

    def __raise_value_error_if_shapes_do_not_match(self) -> None:
        if np.ndim(self.__parameter) != 2:
            message = f'{self.__name} must be a 2D array.'
            raise ValueError(message)

        if self.__parameter.shape[0] != self.__particle_size.shape[0]:
            raise ValueError('parameter does not match particle size shape')

        if self.__parameter.shape[1] != self.__wavelength.shape[0]:
            raise ValueError('parameter does not match wavelength shape.')

    @property
    def parameter(self) -> np.ndarray:
        return self.__parameter

    @property
    def particle_size(self) -> np.ndarray:
        return self.__particle_size

    @property
    def wavelength(self) -> np.ndarray:
        return self.__wavelength


class NearestNeighborSingleScatteringAlbedo:
    """A class to make the single scattering albedo related variables.

    NearestNeighborSingleScatteringAlbedo accepts the scattering and extinction
    cross sections, along with a 1D wavelength and particle size grid. It then
    finds the nearest neighbor value in the input forward scattering properties
    to the grids, and creates the properties on a particle size and wavelength
    grid.

    """
    def __init__(self, scattering_cross_section: np.ndarray,
                 extinction_cross_section: np.ndarray,
                 particle_size_grid: np.ndarray, wavelength_grid: np.ndarray,
                 particle_size: np.ndarray,
                 wavelength: np.ndarray) -> None:
        """
        Parameters
        ----------
        scattering_cross_section
            2D array of the scattering cross section (the 0th axis is assumed to
            be the particle size axis, while the 1st axis is assumed to be the
            wavelength axis).
        extinction_cross_section
            2D array of the extinction cross section with same dims as above.
        particle_size_grid
            1D array of particle sizes over which the above properties are
            defined.
        wavelength_grid
            1D array of wavelengths over which the above properties are defined.
        particle_size
            1D array of particle sizes to regrid the properties onto.
        wavelength
            1D array of wavelengths to regrid the properties onto.

        """
        self.__c_sca = ForwardScatteringProperty(scattering_cross_section,
                                                 particle_size_grid,
                                  wavelength_grid, 'scattering')
        self.__c_ext = ForwardScatteringProperty(extinction_cross_section,
                                                 particle_size_grid,
                                  wavelength_grid, 'extinction')
        self.__particle_size = particle_size
        self.__wavelength = wavelength

        self.__nnps, self.__nnw = self.__get_nearest_neighbor_values()
        self.__gridded_c_scattering = self.__make_gridded_c_scattering()
        self.__gridded_c_extinction = self.__make_gridded_c_extinction()
        self.__gridded_ssa = self.__make_gridded_ssa()

    def __get_nearest_neighbor_values(self):
        size_indices = self.__get_nearest_indices(
            self.__c_sca.particle_size, self.__particle_size)
        wavelength_indices = self.__get_nearest_indices(
            self.__c_sca.wavelength, self.__wavelength)
        return size_indices, wavelength_indices

    @staticmethod
    def __get_nearest_indices(grid: np.ndarray, values: np.ndarray) \
            -> np.ndarray:
        # grid should be 1D; values can be ND
        return np.abs(np.subtract.outer(grid, values)).argmin(0)

    def __make_gridded_c_scattering(self):
        return self.__c_sca.parameter[self.__nnps, :][:, self.__nnw]

    def __make_gridded_c_extinction(self):
        return self.__c_ext.parameter[self.__nnps, :][:, self.__nnw]

    def __make_gridded_ssa(self):
        return self.__gridded_c_scattering / self.__gridded_c_extinction

    @property
    def scattering_cross_section(self) -> np.ndarray:
        """Get the scattering cross section on the new grid.

        """
        return self.__gridded_c_scattering

    @property
    def extinction_cross_section(self) -> np.ndarray:
        """Get the extinction cross section on the new grid.

        """
        return self.__gridded_c_extinction

    @property
    def single_scattering_albedo(self) -> np.ndarray:
        """Get the single scattering albedo on the new grid.

        """
        return self.__gridded_ssa

    def make_extinction_grid(self, reference_wavelength: float) -> np.ndarray:
        """Make a grid of the extinction cross section at a reference wavelength.
        This is the extinction cross section at the input wavelengths divided by
        the extinction cross section at the reference wavelength.

        """
        size_ind = self.__get_nearest_indices(self.__c_ext.particle_size,
                                              self.__particle_size)
        wav_ind = self.__get_nearest_indices(self.__c_ext.wavelength,
                                             np.array([reference_wavelength]))
        extinction_profile = self.__c_ext.parameter[size_ind, wav_ind]
        return (self.__gridded_c_extinction.T / extinction_profile).T
