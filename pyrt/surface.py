"""The :code:`surface` module contains structures for creating arrays related to
DISORT's surface treatment.
"""
import numpy as np
from disort import disobrdf


class Surface:
    """A structure to compute all the surface parameterization.

    Surface accepts parameters that define the surface arrays required by
    DISORT. The methods in Surface will compute those arrays.

    """

    def __init__(self, albedo: float, n_streams: int, n_polar: int,
                 n_azimuth: int, user_angles: bool, only_fluxes: bool,
                 n_mug: int = 200) -> None:
        """
        Parameters
        ----------
        albedo
            The surface albedo. Only used by DISORT if the bottom boundary is
            Lambertian.
        n_streams
            The number of streams.
        n_polar
            The number of polar angles.
        n_azimuth
            The number of azimuth angles.
        n_mug
            The number of angle cosine quadrature points for integrating the
            bidirectional reflectivity.

        Raises
        ------
        TypeError
            Raised if a number of things...
        ValueError
            Raised if albedo is not between 0 and 1.

        """
        self.__albedo = self.__make_albedo(albedo)
        self.__n_streams = n_streams
        self.__n_polar = n_polar
        self.__n_azimuth = n_azimuth
        self.__user_angles = user_angles
        self.__only_fluxes = only_fluxes
        self.__lambertian = False
        self.__n_mug = n_mug

        self.__bemst = self.__make_empty_bemst()
        self.__emust = self.__make_empty_emust()
        self.__rho_accurate = self.__make_empty_rho_accurate()
        self.__rhoq = self.__make_empty_rhoq()
        self.__rhou = self.__make_empty_rhou()

    def __make_albedo(self, albedo: float) -> float:
        albedo = self.__cast_to_float(albedo)
        self.__raise_value_error_if_albedo_is_unphysical(albedo)
        return albedo

    def __make_empty_bemst(self) -> np.ndarray:
        return np.zeros(int(0.5*self.__n_streams))

    def __make_empty_emust(self) -> np.ndarray:
        return np.zeros(self.__n_polar)

    def __make_empty_rho_accurate(self) -> np.ndarray:
        return np.zeros((self.__n_polar, self.__n_azimuth))

    # TODO: the first dimension seems messed up to me...
    def __make_empty_rhoq(self) -> np.ndarray:
        return np.zeros((int(0.5 * self.__n_streams),
                         int(0.5 * self.__n_streams + 1),
                         self.__n_streams))

    def __make_empty_rhou(self) -> np.ndarray:
        return np.zeros((self.__n_streams,
                         int(0.5 * self.__n_streams + 1),
                         self.__n_streams))

    @staticmethod
    def __cast_to_float(albedo: float) -> float:
        try:
            return float(albedo)
        except TypeError as te:
            raise TypeError('albedo cannot be cast into a float.') from te
        except ValueError as ve:
            raise ValueError('albedo cannot be cast into a float.') from ve

    @staticmethod
    def __raise_value_error_if_albedo_is_unphysical(albedo: float) -> None:
        if not 0 <= albedo <= 1:
            raise ValueError('albedo must be between 0 and 1.')

    def __make_output_arrays(self, mu: float, mu0: float, phi: float,
                            phi0: float, beam_flux: float,
                            albedo: float,
                            phase_function_number: int, brdf_arg: np.ndarray,
                            n_mug: int):
        try:
            return self.__call_disobrdf(mu, mu0, phi, phi0, beam_flux, albedo,
                                        phase_function_number, brdf_arg, n_mug)
        except ValueError as ve:
            raise ValueError('problem') from ve

    def __call_disobrdf(
            self, mu: float, mu0: float, phi: float, phi0: float,
            beam_flux: float, albedo, phase_function_number: int,
            brdf_arg: np.ndarray, n_mug: int):
        return disobrdf(self.__user_angles, mu, beam_flux, mu0, False, albedo,
                        self.__only_fluxes, self.__rhoq, self.__rhou, self.__emust,
                        self.__bemst, False, phi, phi0, self.__rho_accurate,
                        phase_function_number, brdf_arg, n_mug,
                        nstr=self.__n_streams, numu=self.__n_polar,
                        nphi=self.__n_azimuth)

    def make_lambertian(self) -> None:
        """Make the surface a Lambertian surface.

        """
        self.__lambertian = True

    def make_hapke(self, b0: float, h: float, w: float, mu: float, mu0: float,
                   phi: float, phi0: float, beam_flux: float) -> None:
        """Make a basic Hapke surface.

        Parameters
        ----------
        b0
            The strength of the opposition surge.
        h
            The width of the opposition surge.
        w
            The surface single scattering albedo.
        mu
            The cosine of emission.
        mu0
            The cosine of incidence.
        phi
            The azimuth angle.
        phi0
            Phi0
        beam_flux
            The incident beam flux.

        See Also
        --------
        :class:`~observation.Angles`, :class:`~radiation.IncidentFlux`

        """
        self.__lambertian = False
        brdf_arg = np.array([b0, h, w, 0, 0, 0])

        outputs = \
            self.__make_output_arrays(mu, mu0, phi, phi0, beam_flux,
                                      self.__albedo, 1, brdf_arg, self.__n_mug)
        self.__rhoq, self.__rhou, self.__emust, self.__bemst, \
        self.__rho_accurate = outputs

    def make_hapkeHG2(self, b0: float, h: float, w: float, asym: float,
                   frac: float, mu: float, mu0: float,
                   phi: float, phi0: float, beam_flux: float) -> None:
        """Make a Hapke surface with 2 lobed Henyey-Greenstein surface phase
        function.

        Parameters
        ----------
        b0
            The strength of the opposition surge.
        h
            The width of the opposition surge.
        w
            The surface single scattering albedo.
        asym
            The asymmetry parameter.
        frac
            The forward scattering fraction.
        mu
            The cosine of emission.
        mu0
            The cosine of incidence.
        phi
            The azimuth angle.
        phi0
            Phi0
        beam_flux
            The incident beam flux.


        """
        self.__lambertian = False
        brdf_arg = np.array([b0, h, w, asym, frac, 0])

        outputs = \
            self.__make_output_arrays(mu, mu0, phi, phi0, beam_flux,
                                      self.__albedo, 5, brdf_arg, self.__n_mug)
        self.__rhoq, self.__rhou, self.__emust, self.__bemst, \
        self.__rho_accurate = outputs

    def make_hapkeHG2_roughness(self, b0: float, h: float, w: float,
                                asym: float, roughness: float,
                   frac: float, mu: float, mu0: float,
                   phi: float, phi0: float, beam_flux: float) -> None:
        """Make a Hapke surface with 2 lobed Henyey-Greenstein surface phase
        function and surface roughness parameter.

        Parameters
        ----------
        b0
            The strength of the opposition surge.
        h
            The width of the opposition surge.
        w
            The surface single scattering albedo.
        asym
            The asymmetry parameter.
        frac
            The forward scattering fraction.
        roughness
            The roughness parameter.
        mu
            The cosine of emission.
        mu0
            The cosine of incidence.
        phi
            The azimuth angle.
        phi0
            Phi0
        beam_flux
            The incident beam flux.

        See Also
        --------
        :class:`~observation.Angles`, :class:`~radiation.IncidentFlux`

        """
        self.__lambertian = False
        brdf_arg = np.array([b0, h, w, asym, frac, roughness])

        outputs = \
            self.__make_output_arrays(mu, mu0, phi, phi0, beam_flux,
                                      self.__albedo, 6, brdf_arg, self.__n_mug)
        self.__rhoq, self.__rhou, self.__emust, self.__bemst, \
        self.__rho_accurate = outputs

    @property
    def albedo(self) -> float:
        """Get the input albedo.

        Notes
        -----
        In DISORT, this variable is named :code:`ALBEDO`.

        """
        return self.__albedo

    @property
    def lambertian(self) -> bool:
        """Get whether the bottom boundary used in the model will be Lambertian.

        Notes
        -----
        In DISORT, this variable is named :code:`LAMBER`.

        """
        return self.__lambertian

    @property
    def bemst(self) -> np.ndarray:
        """Get the directional emissivity at quadrature angles.

        Notes
        -----
        In DISORT, this variable is named :code:`BEMST`.

        """
        return self.__bemst

    @property
    def emust(self) -> np.ndarray:
        """Get the directional emissivity at user angles.

        Notes
        -----
        In DISORT, this variable is named :code:`EMUST`.

        """
        return self.__emust

    @property
    def rho_accurate(self) -> np.ndarray:
        """Get the analytic BRDF results.

        Notes
        -----
        In DISORT, this variable is named :code:`RHO_ACCURATE`.

        """
        return self.__rho_accurate

    @property
    def rhoq(self) -> np.ndarray:
        """Get the quadrature fourier expanded BRDF.

        Notes
        -----
        In DISORT, this variable is named :code:`RHOQ`.

        """
        return self.__rhoq

    @property
    def rhou(self) -> np.ndarray:
        """Get the user defined fourier expanded BRDF.

        Notes
        -----
        In DISORT, this variable is named :code:`RHOU`.

        """
        return self.__rhou
