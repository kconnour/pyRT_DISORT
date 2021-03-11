"""The surface module contains classes for creating arrays related to DISORT's
surface treatment.
"""
import numpy as np
from pyRT_DISORT.controller import ComputationalParameters, ModelBehavior
from pyRT_DISORT.observation import Angles
#from pyRT_DISORT.radiation import IncidentFlux
from radiation import IncidentFlux
from disort import disobrdf


# TODO: I still don't really know what bemst, emust, rhoq, rhou, or rho_accurate
#  are... get better names.
class Surface:
    """An abstract base "surface" class.

    Surface holds properties relevant to all surfaces. It is an abstract base
    class from which all other surface classes are derived; it is not meant to
    be instantiated.

    """

    def __init__(self, albedo: float, cp: ComputationalParameters) -> None:
        """
        Parameters
        ----------
        albedo: float
            The surface albedo. Only used by DISORT if the bottom boundary is
            Lambertian (see the Lambertian class), but always used by
            pyRT_DISORT.
        cp: ComputationalParameters
            The model's computational parameters.

        Raises
        ------
        TypeError
            Raised if albedo cannot be cast to a float, or cp is not an instance
            of ComputationalParameters
        ValueError
            Raised if albedo is not between 0 and 1.

        """
        self.__albedo = self.__make_albedo(albedo)
        self.__raise_type_error_if_cp_is_not_computational_parameters(cp)
        self._cp = cp
        self._lambertian = False

        self._bemst = self.__make_empty_bemst()
        self._emust = self.__make_empty_emust()
        self._rho_accurate = self.__make_empty_rho_accurate()
        self._rhoq = self.__make_empty_rhoq()
        self._rhou = self.__make_empty_rhou()

    def __make_albedo(self, albedo: float) -> float:
        albedo = self.__cast_to_float(albedo)
        self.__raise_value_error_if_albedo_is_unphysical(albedo)
        return albedo

    @staticmethod
    def __raise_type_error_if_cp_is_not_computational_parameters(
            cp: ComputationalParameters) -> None:
        if not isinstance(cp, ComputationalParameters):
            raise TypeError(
                'cp must be an instance of ComputationalParameters.')

    def __make_empty_bemst(self) -> np.ndarray:
        return np.zeros(int(0.5*self._cp.n_streams))

    def __make_empty_emust(self) -> np.ndarray:
        return np.zeros(self._cp.n_polar)

    def __make_empty_rho_accurate(self) -> np.ndarray:
        return np.zeros((self._cp.n_polar, self._cp.n_azimuth))

    # TODO: the first dimension seems messed up to me...
    def __make_empty_rhoq(self) -> np.ndarray:
        return np.zeros((int(0.5 * self._cp.n_streams),
                         int(0.5 * self._cp.n_streams + 1),
                         self._cp.n_streams))

    def __make_empty_rhou(self) -> np.ndarray:
        return np.zeros((self._cp.n_streams,
                         int(0.5 * self._cp.n_streams + 1),
                         self._cp.n_streams))

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

    def _make_output_arrays(self, mb: ModelBehavior, angles: Angles,
                            flux: IncidentFlux, albedo,
                            phase_function_number: int, brdf_arg: np.ndarray,
                            n_mug: int):
        try:
            return self.__call_disobrdf(
                mb, angles, flux, albedo, phase_function_number, brdf_arg,
                n_mug)
        except ValueError as ve:
            raise ValueError('problem') from ve

    def __call_disobrdf(self, mb: ModelBehavior, angles: Angles,
                        flux: IncidentFlux, albedo, phase_function_number: int,
                        brdf_arg: np.ndarray, n_mug: int):
        return disobrdf(mb.user_angles, angles.mu, flux.beam_flux, angles.mu0,
                        False, albedo, mb.only_fluxes, self._rhoq, self._rhou,
                        self._emust, self._bemst, False, angles.phi,
                        angles.phi0, self._rho_accurate, phase_function_number,
                        brdf_arg,
                        n_mug, nstr=self._cp.n_streams, numu=self._cp.n_polar,
                        nphi=self._cp.n_azimuth)

    @property
    def albedo(self) -> float:
        """Get the input albedo.

        Returns
        -------
        float
            The albedo.

        Notes
        -----
        In DISORT, this variable is named "ALBEDO".

        """
        return self.__albedo

    @property
    def lambertian(self) -> bool:
        """Get whether the bottom boundary used in the model will be Lambertian.

        Returns
        -------
        bool
            True if the class is Lambertian, False otherwise.

        Notes
        -----
        In DISORT, this variable is named "LAMBER".

        """
        return self._lambertian

    @property
    def bemst(self) -> np.ndarray:
        """Get the directional emissivity at quadrature angles.

        Returns
        -------
        np.ndarray
            The directional emissivity at quadrature angles.

        Notes
        -----
        In DISORT, this variable is named "BEMST"

        """
        return self._bemst

    @property
    def emust(self) -> np.ndarray:
        """Get the directional emissivity at user angles.

        Returns
        -------
        np.ndarray
            The directional emissivity at user angles.

        Notes
        -----
        In DISORT, this variable is named "EMUST"

        """
        return self._emust

    @property
    def rho_accurate(self) -> np.ndarray:
        """Get the analytic BRDF results.

        Returns
        -------
        np.ndarray
            The analytic BRDF results.

        Notes
        -----
        In DISORT, this variable is named "RHO_ACCURATE".

        """
        return self._rho_accurate

    @property
    def rhoq(self) -> np.ndarray:
        """Get the quadrature fourier expanded BRDF.

        Returns
        -------
        np.ndarray
            The quadrature fourier expanded BRDF.

        Notes
        -----
        In DISORT, this variable is named "RHOQ".

        """
        return self._rhoq

    @property
    def rhou(self) -> np.ndarray:
        """Get the user defined fourier expanded BRDF.

        Returns
        -------
        np.ndarray
            The user-defined fourier expanded BRDF.

        Notes
        -----
        In DISORT, this variable is named "RHOU".

        """
        return self._rhou


class Lambertian(Surface):
    """Create a Lambertian surface.

    Lambertian creates a boolean flag that can notify DISORT to use a Lambertian
    surface. It also creates bi-directional reflectivity arrays of 0s required
    by DISORT.

    """
    def __init__(self, albedo: float, cp: ComputationalParameters):
        """
        Parameters
        ----------
        albedo: float
            The surface albedo.
        cp: ComputationalParameters
            The computational parameters.

        Raises
        ------
        TypeError
            Raised if albedo cannot be cast to a float or if cp is not an
            instance of ComputationalParameters.
        ValueError
            Raised if albedo is not between 0 and 1.

        """
        super().__init__(albedo, cp)
        self._lambertian = True


class Hapke(Surface):
    """Create a basic Hapke surface.

    Hapke creates the bi-directional reflectivity arrays required by DISORT
    based on the input parameterization of a Hapke surface.

    """

    def __init__(self, albedo: float, cp: ComputationalParameters,
                 mb: ModelBehavior, flux: IncidentFlux, angles: Angles,
                 b0: float, h: float, w: float, n_mug: int = 200) -> None:
        """
        Parameters
        ----------
        albedo: float
            The surface albedo.
        cp: ComputationalParameters
            The computational parameters.
        mb: ModelBehavior
            The model behavior.
        flux: IncidentFlux
            The incident flux.
        angles: Angles
            The observation angles.
        b0: float
            The strength of the opposition surge.
        h: float
            The width of the opposition surge.
        w: float
            The surface single scattering albedo.
        n_mug: int, optional
            The number of angle cosine quadrature points for integrating the
            bidirectional reflectivity. Default is 200.

        Raises
        ------
        TypeError
            Raised if inputs cannot be cast to the correct type.
        ValueError
            Raised if albedo is not between 0 and 1.

        """
        super().__init__(albedo, cp)
        brdf_arg = np.array([b0, h, w, 0, 0, 0])

        self._rhoq, self._rhou, self._emust, self._bemst, self._rho_accurate = \
            self._make_output_arrays(mb, angles, flux, albedo, 1, brdf_arg,
                                     n_mug)


class HapkeHG2(Surface):
    """Create a Hapke surface with a 2-lobed Henyey-Greenstein phase function.

    HapkeHG2 creates the bi-directional reflectivity arrays required by DISORT
    based on the input parameterization of a 2-lobed Hapke surface.

    """

    def __init__(self, albedo: float, cp: ComputationalParameters,
                 mb: ModelBehavior, flux: IncidentFlux, angles: Angles,
                 b0: float, h: float, w: float, asym: float, frac: float,
                 n_mug: int = 200) -> None:
        """
        Parameters
        ----------
        albedo: float
            The surface albedo.
        cp: ComputationalParameters
            The computational parameters.
        mb: ModelBehavior
            The model behavior.
        flux: IncidentFlux
            The incident flux.
        angles: Angles
            The observation angles.
        b0: float
            The strength of the opposition surge.
        h: float
            The width of the opposition surge.
        w: float
            The surface single scattering albedo.
        asym: float
            The asymmetry parameter
        frac: float
            The forward scattering fraction
        n_mug: int, optional
            The number of angle cosine quadrature points for integrating the
            bidirectional reflectivity. Default is 200.

        Raises
        ------
        TypeError
            Raised if inputs cannot be cast to the correct type.
        ValueError
            Raised if albedo is not between 0 and 1.

        """
        super().__init__(albedo, cp)
        brdf_arg = np.array([b0, h, w, asym, frac, 0])

        self._rhoq, self._rhou, self._emust, self._bemst, self._rho_accurate = \
            self._make_output_arrays(mb, angles, flux, albedo, 5, brdf_arg,
                                     n_mug)


class HapkeHG2Roughness(Surface):
    """Create a Hapke surface with a 2-lobed Henyey-Greenstein phase function.

    HapkeHG2Roughness creates the bi-directional reflectivity arrays required by
    DISORT based on the input parameterization of a 2-lobed Hapke surface with
    roughness parameter.

    """

    def __init__(self, albedo: float, cp: ComputationalParameters,
                 mb: ModelBehavior, flux: IncidentFlux, angles: Angles,
                 b0: float, h: float, w: float, asym: float, frac: float,
                 roughness: float, n_mug: int = 200) -> None:
        """
        Parameters
        ----------
        albedo: float
            The surface albedo.
        cp: ComputationalParameters
            The computational parameters.
        mb: ModelBehavior
            The model behavior.
        flux: IncidentFlux
            The incident flux.
        angles: Angles
            The observation angles.
        b0: float
            The strength of the opposition surge.
        h: float
            The width of the opposition surge.
        w: float
            The surface single scattering albedo.
        asym: float
            The asymmetry parameter.
        frac: float
            The forward scattering fraction.
        roughness: float
            The roughness parameter.
        n_mug: int, optional
            The number of angle cosine quadrature points for integrating the
            bidirectional reflectivity. Default is 200.

        Raises
        ------
        TypeError
            Raised if inputs cannot be cast to the correct type.
        ValueError
            Raised if albedo is not between 0 and 1.

        """
        super().__init__(albedo, cp)
        brdf_arg = np.array([b0, h, w, asym, frac, roughness])

        self._rhoq, self._rhou, self._emust, self._bemst, self._rho_accurate = \
            self._make_output_arrays(mb, angles, flux, albedo, 6, brdf_arg,
                                     n_mug)
