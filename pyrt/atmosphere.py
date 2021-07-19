"""The :code:`atmosphere` module contains structures to make the total
atmospheric properties required by DISORT.
"""
import numpy as np


class Atmosphere:
    """A structure to compute the total atmospheric properties.

    Atmosphere accepts a collection of atmospheric properties for each
    constituent in the model. It then computes the total atmospheric arrays
    from these inputs.

    """

    def __init__(self,
                 *args: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """
        Parameters
        ----------
        args
            Tuple of (optical depth, single scattering albedo, phase function)
            for each constituent to add to the atmospheric model.

        """
        self.__properties = args

        self.__check_constituents()

        self.__n_species = len(self.__properties)
        self.__constituent_optical_depth = self.__extract_property(0)
        self.__constituent_single_scattering_albedo = self.__extract_property(1)
        self.__constituent_phase_function = self.__extract_property(2)

        self.__optical_depth = self.__calculate_optical_depth()
        self.__single_scattering_albedo = \
            self.__calculate_single_scattering_albedo()
        self.__legendre_moments = self.__calculate_legendre_coefficients()

    def __check_constituents(self):
        for p in self.__properties:
            if not isinstance(p, tuple):
                raise TypeError('properties must be a tuple')
            if len(p) != 3:
                raise ValueError('properties must be of length 3')
            if not all(isinstance(x, np.ndarray) for x in p):
                raise TypeError('All elements in args must be a np.ndarray')

    def __extract_property(self, index):
        return [f[index] for f in self.__properties]

    def __calculate_optical_depth(self) -> np.ndarray:
        return sum(self.__constituent_optical_depth)

    def __calculate_single_scattering_albedo(self) -> np.ndarray:
        scattering_od = [self.__constituent_single_scattering_albedo[i] *
                         self.__constituent_optical_depth[i] for i in
                         range(self.__n_species)]
        return sum(scattering_od) / self.__optical_depth

    def __calculate_legendre_coefficients(self) -> np.ndarray:
        max_moments = self.__get_max_moments()
        self.__match_moments(self.__constituent_phase_function,
                             max_moments)
        weighted_moments = [self.__constituent_single_scattering_albedo[i] *
                            self.__constituent_optical_depth[i] *
                            self.__constituent_phase_function[i] for i in
                            range(self.__n_species)]
        denom = [self.__constituent_single_scattering_albedo[i] *
                 self.__constituent_optical_depth[i]
                 for i in range(self.__n_species)]
        return sum(weighted_moments) / sum(denom)

    def __get_max_moments(self):
        return max(i.shape[0] for i in self.__constituent_phase_function)

    def __match_moments(self, phase_functions, max_moments):
        for counter, pf in enumerate(phase_functions):
            if pf.shape[0] < max_moments:
                self.__constituent_phase_function[
                    counter] = self.__add_moments(pf, max_moments)

    @staticmethod
    def __add_moments(phase_function, max_moments):
        starting_inds = np.linspace(phase_function.shape[0],
                                    phase_function.shape[0],
                                    num=max_moments - phase_function.shape[0],
                                    dtype=int)
        return np.insert(phase_function, starting_inds, 0, axis=0)

    @property
    def optical_depth(self) -> np.ndarray:
        r"""Get the total optical depth of the atmosphere. This is computed via

        .. math::
           \tau = \Sigma \tau_i

        where :math:`\tau` is the total optical depth, and :math:`\tau_i` is the
        optical depth
        of each of the atmospheric species.

        Notes
        -----
        Each element of this variable along the wavelength dimension is named
        :code:`DTAUC` in DISORT.

        """
        return self.__optical_depth

    @property
    def single_scattering_albedo(self) -> np.ndarray:
        r"""Get the single scattering albedo of the atmosphere. This is computed
        via

        .. math::
           \tilde{\omega} = \frac{\Sigma \tilde{\omega}_i \tau_i}{\tau}

        where :math:`\tilde{\omega}` is the total single scattering albedo,
        :math:`\tilde{\omega}_i` is the single scattering albedo of an
        individual species, and :math:`\tau` is the total optical depth.

        Notes
        -----
        Each element of this variable along the wavelength dimension is named
        :code:`SSALB` in DISORT.

        """
        return self.__single_scattering_albedo

    @property
    def legendre_moments(self) -> np.ndarray:
        r"""Get the total Legendre coefficient array of the atmosphere. This is
        computed via

        .. math::
           P = \frac{\Sigma \tilde{\omega}_i * \tau_i * P_i}
                    {\tilde{\omega} * \tau}

        where :math:`P` is the total phase function array, :math:`P_i` is the
        phase function of each constituent, and the other variables are defined
        in the other properties.

        Notes
        -----
        Each eleemnt of this variable along the wavelength dimension is named
        :code:`PMOM` in DISORT.

        """
        return self.__legendre_moments
