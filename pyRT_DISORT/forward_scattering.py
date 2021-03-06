import numpy as np


# TODO: Include an optional g input parameter
class ForwardScatteringProperties:
    def __init__(self, scattering_cross_section: np.ndarray,
                 extinction_cross_section: np.ndarray,
                 particle_size_grid: np.ndarray, wavelength_grid) -> None:
        """
        Parameters
        ----------
        scattering_cross_section
            2D grid of the scattering cross sections. This is assumed to have
            shape [particle_size_grid, wavelength_grid].
        extinction_cross_section
            2D grid of the extinction cross sections. This is assumed to have
            shape [particle_size_grid, wavelength_grid].
        particle_size_grid
            1D grid of the particle sizes.
        wavelength_grid
            1D grid of the wavelengths.

        """

        self.__scattering = scattering_cross_section
        self.__extinction = extinction_cross_section
        self.__sizes = particle_size_grid
        self.__wavelength = wavelength_grid
        self.__ssa = self.__make_single_scattering_albedo()

    def __make_single_scattering_albedo(self) -> np.ndarray:
        try:
            return self.__scattering / self.__extinction
        except ValueError as ve:
            raise ValueError('The scattering and extinction arrays cannot be'
                             'broadcast together.') from ve

    @property
    def scattering_cross_section(self) -> np.ndarray:
        return self.__scattering

    @property
    def extinction_cross_section(self) -> np.ndarray:
        return self.__extinction

    @property
    def particle_size_grid(self) -> np.ndarray:
        return self.__sizes

    @property
    def wavelength_grid(self) -> np.ndarray:
        return self.__wavelength

    @property
    def single_scattering_albedo(self) -> np.ndarray:
        return self.__ssa


class GriddedForwardScatteringProperties:
    def __init__(self,
                 forward_scattering_properties: ForwardScatteringProperties,
                 particle_size_profile: np.ndarray,
                 wavelengths: np.ndarray, reference_wavelength: float) -> None:
        self.__fsp = forward_scattering_properties
        self.__size_inds, self.__wavelength_inds = \
            self.__get_nearest_neighbor_values(particle_size_profile, wavelengths)
        self.__gridded_c_scattering = self.__make_gridded_c_scattering()
        self.__gridded_c_extinction = self.__make_gridded_c_extinction()
        self.__gridded_ssa = self.__make_gridded_single_scattering_albedo()
        self.__extinction = self.__make_extinction_grid(reference_wavelength)

    def __get_nearest_neighbor_values(self, particle_size_profile, wavelengths):
        size_indices = self.__get_nearest_indices(self.__fsp.particle_size_grid,
                                                  particle_size_profile)
        wavelength_indices = self.__get_nearest_indices(
            self.__fsp.wavelength_grid, wavelengths)
        return size_indices, wavelength_indices

    @staticmethod
    def __get_nearest_indices(grid: np.ndarray, values: np.ndarray) \
            -> np.ndarray:
        # grid should be 1D; values can be ND
        return np.abs(np.subtract.outer(grid, values)).argmin(0)

    def __make_gridded_c_scattering(self) -> np.ndarray:
        return self.__fsp.scattering_cross_section[self.__size_inds, :][:, self.__wavelength_inds]

    def __make_gridded_c_extinction(self) -> np.ndarray:
        return self.__fsp.extinction_cross_section[self.__size_inds, :][:, self.__wavelength_inds]

    def __make_gridded_single_scattering_albedo(self) -> np.ndarray:
        return self.__gridded_c_scattering / self.__gridded_c_extinction

    def __make_extinction_grid(self, reference_wavelength) -> np.ndarray:
        size_ind, wav_ind = self.__get_nearest_neighbor_values(self.__fsp.particle_size_grid, np.array([reference_wavelength]))
        extinction_profile = self.__fsp.extinction_cross_section[self.__size_inds, wav_ind]
        return (self.__gridded_c_extinction.T / extinction_profile).T

    @property
    def scattering_cross_section(self) -> np.ndarray:
        return self.__gridded_c_scattering

    @property
    def extinction_cross_section(self) -> np.ndarray:
        return self.__gridded_c_extinction

    @property
    def single_scattering_albedo(self) -> np.ndarray:
        return self.__gridded_ssa

    @property
    def extinction(self) -> np.ndarray:
        return self.__extinction
