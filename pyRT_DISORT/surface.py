"""The surface module contains classes for creating arrays related to DISORT's
surface treatment.
"""
from typing import Any


class Surface:
    """An abstract base "surface" class.

    Surface holds properties relevant to all surfaces. It is an abstract base
    class from which all other surface classes are derived; it is not meant to
    be instantiated.

    """
    def __init__(self, albedo: float) -> None:
        """
        Parameters
        ----------
        albedo: float
            The surface albedo. Only used by DISORT if the bottom boundary is
            Lambertian (see the Lambertian class), but always used by
            pyRT_DISORT.

        Raises
        ------
        TypeError
            Raised if albedo cannot be cast to a float.
        ValueError
            Raised if is not between 0 and 1.

        """
        self.__albedo = self.__make_albedo(albedo)

    @staticmethod
    def __make_albedo(albedo: Any) -> float:
        try:
            if not 0 <= albedo <= 1:
                raise ValueError('albedo must be between 0 and 1.')
            return float(albedo)
        except TypeError as te:
            raise TypeError('albedo must be a float.') from te

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
    def __init__(self, albedo):
        super().__init__(albedo)

    # TODO: add rhoq, rhou, bemst, emust, and rho_accurate

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
