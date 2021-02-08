"""The surface module contains classes for creating arrays related to DISORT's
surface treatment.
"""
import numpy as np
from pyRT_DISORT.controller import ComputationalParameters


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
            Raised if albedo cannot be cast to a float.
        ValueError
            Raised if albedo is not between 0 and 1.

        """
        self.__albedo = self.__make_albedo(albedo)
        self.__raise_type_error_if_cp_is_not_computational_parameters(cp)
        self.__cp = cp

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
        return np.zeros(int(0.5*self.__cp.n_streams))

    def __make_empty_emust(self) -> np.ndarray:
        return np.zeros(self.__cp.n_polar)

    def __make_empty_rho_accurate(self) -> np.ndarray:
        return np.zeros((self.__cp.n_polar, self.__cp.n_azimuth))

    # TODO: the first dimension seems messed up to me...
    def __make_empty_rhoq(self) -> np.ndarray:
        return np.zeros((self.__cp.n_streams,
                         int(0.5 * self.__cp.n_streams + 1),
                         self.__cp.n_streams))

    def __make_empty_rhou(self) -> np.ndarray:
        return np.zeros((int(0.5 * self.__cp.n_streams),
                         int(0.5 * self.__cp.n_streams + 1),
                         self.__cp.n_streams))

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


class Lambertian(Surface):
    """Create a Lambertian surface.


    Lambertian creates a boolean flag that can notify DISORT to use a Lambertian
    surface. It also creates the bi-directional reflectivity arrays required
    by DISORT.

    """
    def __init__(self, albedo: float, cp: ComputationalParameters):
        """
        Parameters
        ----------
        albedo: float
            The surface albedo.
        cp: ComputationalParameters
            The computational parameters

        Raises
        ------
        TypeError
            Raised if albedo cannot be cast to a float or if cp is not an
            instance of ComputationalParameters.
        ValueError
            Raised if albedo is not between 0 and 1.

        """
        super().__init__(albedo, cp)

    @property
    def lambertian(self) -> bool:
        """Get whether the bottom boundary used in the model will be Lambertian.
        This is necessarily True for this class.

        Returns
        -------
        bool
            True.

        Notes
        -----
        In DISORT, this variable is named "LAMBER".

        """
        return True

    @property
    def bemst(self) -> np.ndarray:
        """Get the directional emissivity at quadrature angles. For a Lambertian
        surface, this array is (presumably) unused and therefore set to all 0s.

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
        """Get the directional emissivity at user angles. For a Lambertian
        surface, this array is (presumably) unused and therefore set to all 0s.

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
        """Get the analytic BRDF results. For a Lambertian surface, this
        array is (presumably) unused and therefore set to all 0s.

        Returns
        -------
        np.ndarray
            The analystic BRDF results.

        Notes
        -----
        In DISORT, this variable is named "RHO_ACCURATE".

        """
        return self._rho_accurate

    @property
    def rhoq(self) -> np.ndarray:
        """Get the quadrature fourier expanded BRDF. For a Lambertian surface,
        this array is (presumably) unused and therefore set to all 0s.

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
        """Get the user defined fourier expanded BRDF. For a Lambertian surface,
        this array is (presumably) unused and therefore set to all 0s.

        Returns
        -------
        np.ndarray
            The user-defined fourier expanded BRDF.

        Notes
        -----
        In DISORT, this variable is named "RHOU".

        """
        return self._rhou
