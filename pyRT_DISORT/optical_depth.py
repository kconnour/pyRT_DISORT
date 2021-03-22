import numpy as np


class OpticalDepth:
    """A structure to compute the optical depth given profiles.

    OpticalDepth accepts a mixing ratio profile and atmospheric column density
    profile to compute the optical depth at each wavelength. It also accepts
    an extinction profile to scale the optical depth to a reference wavelength,
    and ensures that the sum over the layers is equal to the total column
    integrated optical depth.

    """
    def __init__(self, mixing_ratio_profile: np.ndarray,
                 column_density_layers: np.ndarray,
                 extinction: np.ndarray,
                 column_integrated_optical_depth: float) -> None:
        """
        Parameters
        ----------
        mixing_ratio_profile
            1D array of mixing ratio.
        column_density_layers
            1D array of the column density in the layers.
        extinction
            2D array of extinction coefficients.
        column_integrated_optical_depth
            The total column integrated optical depth.

        """
        self.__q_prof = mixing_ratio_profile
        self.__colden = column_density_layers
        self.__extinction = extinction
        self.__OD = column_integrated_optical_depth

        self.__optical_depth = self.__calculate_total_optical_depth()

    def __calculate_total_optical_depth(self) -> np.ndarray:
        normalization = np.sum(self.__q_prof * self.__colden)
        profile = self.__q_prof * self.__colden * self.__OD / normalization
        return (profile * self.__extinction.T).T

    @property
    def total(self) -> np.ndarray:
        """Get the total optical depth in each of the layers.

        """
        return self.__optical_depth
