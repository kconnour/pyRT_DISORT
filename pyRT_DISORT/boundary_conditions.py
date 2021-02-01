"""boundary_conditions.py contains a data structure to set variables related
to DISORT's boundary conditions.
"""
import numpy as np


class BoundaryConditions:
    """A BoundaryConditions object holds boundary condition data for DISORT.

    Longer description here.
    """
    def __init__(self, albedo: float = 0.0, beam_flux: float = np.pi,
                 bottom_temperature: float = 0.0, fisot: int = 0,
                 ibcnd: int = 0, lambertian_bottom_boundary: bool = True,
                 plank: bool = False, top_emissivity: float = 1.0,
                 top_temperature: float = 0.0):
        """
        Parameters
        ----------
        albedo: float, optional
            The surface albedo. Only required if lambertian_bottom_boundary is
            True. Default is 0.0.
        beam_flux: float, optional
            The intensity of the incident parallel beam the the top boundary. If
            plank == True, this has the same units as the variable "PLKAVG",
            which defaults to W / m**2. Ensure this variable and "fisot" have
            the same units. If plank == False, the radiant output units are the
            same as this quantity and fisot. Note that this is an infinitely
            wide beam. Default is pi.
        bottom_temperature
        fisot
        ibcnd
        lambertian_bottom_boundary
        plank
        top_emissivity
        top_temperature

        Raises
        ------
        TypeError
            Raised if the inputs are not of the correct type.
        ValueError
            Raised if the inputs are unphysical.

        """

        self.__albedo = albedo
        self.__beam_flux = beam_flux

        self.__raise_error_if_input_boundary_conditions_are_bad()

    def __raise_error_if_input_boundary_conditions_are_bad(self):
        self.__raise_error_if_albedo_is_bad()
        self.__raise_error_if_beam_flux_is_bad()

    def __raise_error_if_albedo_is_bad(self):
        self.__raise_type_error_if_input_is_not_float(self.__albedo, 'albedo')
        self.__raise_value_error_if_albedo_is_unphysical()

    def __raise_error_if_beam_flux_is_bad(self):
        self.__raise_type_error_if_input_is_not_float(
            self.__beam_flux, 'beam_flux')
        self.__raise_value_error_if_beam_flux_is_not_positive_finite()

    @staticmethod
    def __raise_type_error_if_input_is_not_float(quantity, name):
        if not isinstance(quantity, float):
            raise TypeError(f'{name} must be a float.')

    def __raise_value_error_if_albedo_is_unphysical(self):
        if not 0 <= self.__albedo <= 1:
            raise ValueError('albedo must be between 0 and 1.')

    def __raise_value_error_if_beam_flux_is_not_positive_finite(self):
        if np.isinf(self.__beam_flux):
            raise ValueError('beam_flux must be finite.')
        if self.__beam_flux <= 0:
            raise ValueError('beam_flux must be positive.')

    @property
    def albedo(self):
        """Get the input surface albedo.

        Returns
        -------
        float
            The surface albedo.

        Notes
        -----
        In DISORT, this variable is named "ALBEDO".

        """
        return self.__albedo

    @property
    def beam_flux(self):
        """Get the input beam flux.

        Returns
        -------
        float
            The beam flux.

        Notes
        -----
        In DISORT, this variable is named "FBEAM".

        """
        return self.__beam_flux
