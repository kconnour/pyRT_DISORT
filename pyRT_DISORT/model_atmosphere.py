# 3rd-party imports
import numpy as np


class ModelAtmosphere:
    """A structure to compute the total atmospheric properties.

    ModelAtmosphere accepts the radiative properties of atmospheric constituents
    and holds them. When requested, it computes the total atmospheric quantities
    for use in DISORT.

    """

    def __init__(self):
        self.__constituent_optical_depth = []
        self.__constituent_single_scattering_albedo = []
        self.__constituent_legendre_coefficient = []
        self.__optical_depth = np.array([])
        self.__single_scattering_albedo = np.array([])
        self.__legendre_moments = np.array([])

    def add_constituent(self, properties: tuple) -> None:
        """Add an atmospheric constituent to the model. :code:`properties` must
        be a tuple of the numpy.ndarrays of the optical depth, single scattering
        albedo, and Legendre coefficient decomposed phase function.

        """
        self.__check_constituent_addition(properties)
        self.__constituent_optical_depth.append(properties[0])
        self.__constituent_single_scattering_albedo.append(properties[1])
        self.__constituent_legendre_coefficient.append(properties[2])

    @staticmethod
    def __check_constituent_addition(properties):
        if not isinstance(properties, tuple):
            raise TypeError('properties must be a tuple')
        if len(properties) != 3:
            raise ValueError('properties must be of length 3')
        if not all(isinstance(x, np.ndarray) for x in properties):
            raise TypeError('All elements in properties must be a np.ndarray')

    def compute_model(self) -> None:
        self.__calculate_optical_depth()
        self.__calculate_single_scattering_albedo()
        self.__calculate_legendre_coefficients()

    def __calculate_optical_depth(self):
        self.__optical_depth = sum(self.__constituent_optical_depth)

    def __calculate_single_scattering_albedo(self):
        scattering_od = [self.__constituent_single_scattering_albedo[i] *
                         self.__constituent_optical_depth[i] for i in
                         range(len(self.__constituent_optical_depth))]
        self.__single_scattering_albedo = \
            sum(scattering_od) / self.__optical_depth

    def __calculate_legendre_coefficients(self):
        max_moments = self.__get_max_moments()
        self.__match_moments(self.__constituent_legendre_coefficient, max_moments)
        weighted_moments = [self.__constituent_single_scattering_albedo[i] *
                            self.__constituent_optical_depth[i] *
                            self.__constituent_legendre_coefficient[i] for i in
                            range(len(self.__constituent_optical_depth))]
        self.__legendre_moments = sum(weighted_moments) / (self.__optical_depth * self.__single_scattering_albedo)

    def __get_max_moments(self):
        return max(i.shape[0] for i in self.__constituent_legendre_coefficient)

    def __match_moments(self, phase_functions, max_moments):
        for counter, pf in enumerate(phase_functions):
            if pf.shape[0] < max_moments:
                self.__constituent_legendre_coefficient[counter] = self.__add_moments(pf, max_moments)

    @staticmethod
    def __add_moments(phase_function, max_moments):
        starting_inds = np.linspace(phase_function.shape[0], phase_function.shape[0],
                                    num=max_moments - phase_function.shape[0], dtype=int)
        return np.insert(phase_function, starting_inds, 0, axis=0)

    @property
    def optical_depth(self) -> np.ndarray:
        r"""Get the total optical depth of the atmosphere. This is computed via

        .. math::
           \tau = \Sigma \tau_i

        where :math:`\tau` is the total optical depth, and :math:`\tau_i` is the optical depth
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
           \tilde{\omega} = \frac{\Sigma \tilde{\omega}_i * \tau_i}{\tau}

        where :math:`\tilde{\omega}` is the total single scattering albedo and
        :math:`\tilde{\omega}_i` is the single scattering albedo of an individual
        species.

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
           P = \frac{\Sigma \tilde{\omega}_i * \tau_i * P_i}{\tilde{\omega} * \tau}

        where :math:`P` is the total phase function array.

        Notes
        -----
        Each eleemnt of this variable along the wavelength dimension is named
        :code:`PMOM` in DISORT.

        """
        return self.__legendre_moments
