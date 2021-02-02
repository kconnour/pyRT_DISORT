"""boundary_conditions.py contains a data structure to set variables related
to DISORT's boundary conditions.
"""
from typing import Any
import numpy as np


class BoundaryConditions:
    """A BoundaryConditions object holds boundary condition data for DISORT.

    BoundaryConditions is a data structure to set and hold boundary variables.
    Many of these quantities are only used if another variable is set to True.
    See the constructor docstring for more information on which variables are
    needed in which circumstances.

    """
    def __init__(self, thermal_emission: bool = False,
                 bottom_temperature: float = 0.0, top_temperature: float = 0.0,
                 top_emissivity: float = 1.0, beam_flux: float = np.pi,
                 isotropic_flux: float = 0.0,
                 lambertian_bottom_boundary: bool = True, albedo: float = 0.0,
                 incidence_beam_conditions: bool = False) -> None:
        """
        Parameters
        ----------
        thermal_emission: bool, optional
            Denote whether to use thermal emission. If True, DISORT will include
            thermal emission and will need the following variables:

                - bottom_temperature (from this class)
                - top_temperature (from this class)
                - top_emissivity (from this class)
                - low_wavenumber (from Angles)
                - high_wavenumber (from Angles)
                - temperature_boundaries (from ModelEquationOfState)

            If False, DISORT will save computation time by ignoring all thermal
            emission and all of the aforementioned variables; however,
            pyRT_DISORT will still compute all variables. Default is False.
        bottom_temperature: float, optional
            The temperature of the bottom boundary [K]. Only used by DISORT if
            thermal_emission == True. Default is 0.0.
        top_temperature: float, optional
            The temperature of the top boundary [K]. Only used by DISORT if
            thermal_emission == True. Default is 0.0.
        top_emissivity: float, optional
            The emissivity of the top boundary. Only used by DISORT if
            thermal_emission == True. Default is 1.0.
        beam_flux: float, optional
            The intensity of the incident beam at the top boundary. If
            thermal_emission == True this is assumed to have the same units as
            "PLKAVG" (which defaults to W / m**2) and the corresponding incident
            flux is umu0 * beam_flux. Ensure this variable and isotropic_flux
            have the same units. If thermal_emission == False, this variable and
            isotropic_flux have arbitrary units, and the output fluxes and
            intensities are assumed to have the same units as these variables.
            Note that this is an infinitely wide beam. Default is pi.
        isotropic_flux: float, optional
            The intensity of isotropic illumination at the top boundary. If
            thermal_emission == True this is assumed to have the same units as
            "PLKAVG" (which defaults to W / m**2) and the corresponding incident
            flux is pi * isotropic_flux. Ensure this variable and beam_flux
            have the same units. If thermal_emission == False, this variable and
            beam_flux have arbitrary units, and the output fluxes and
            intensities are assumed to have the same units as these variables.
            Default is 0.0.
        lambertian_bottom_boundary: bool, optional
            Denote whether to use an isotropically reflecting ("Lambertian")
            bottom boundary. If True, DISORT will use the albedo specified in
            this object. If False, DISORT will use a bidirectionally reflecting
            bottom boundary defined in BDREF.f. See the surface module of
            pyRT_DISORT for included surfaces. Default is True.
        albedo: float, optional
            The surface albedo. Only used by DISORT if
            lambertian_bottom_boundary == True. Default is 0.0.
        incidence_beam_conditions: bool, optional
            Denote what functions of the incidence beam angle should be
            included. If True, return the albedo and transmissivity of the
            entire medium as a function of incidence beam angle. In this case,
            the following inputs are the only ones considered by DISORT:

                - NLYR
                - DTAUC
                - SSALB
                - PMOM
                - NSTR
                - USRANG
                - NUMU
                - UMU
                - ALBEDO
                - PRNT
                - HEADER

            PLANK is assumed to be False, LAMBER is assumed to be True, and
            ONLYFL must be False. The only output is ALBMED and TRNMED. The
            intensities are not corrected for delta-M+ correction.

            If False, this is accommodates any general case of boundary
            conditions including beam illumination from the top, isotropic
            illumination from the top, thermal emission from the top, internal
            thermal emission, reflection at the bottom, and/or thermal emission
            from the bottom. Default is False.

        Raises
        ------
        TypeError
            Raised if the inputs are not of the correct type.
        ValueError
            Raised if the inputs are outside their physically allowable range of
            values.

        """
        self.__thermal_emission = thermal_emission
        self.__bottom_temperature = bottom_temperature
        self.__top_temperature = top_temperature
        self.__top_emissivity = top_emissivity
        self.__beam_flux = beam_flux
        self.__isotropic_flux = isotropic_flux
        self.__lambertian_bottom_boundary = lambertian_bottom_boundary
        self.__albedo = albedo
        self.__incidence_beam_conditions = incidence_beam_conditions

        self.__raise_error_if_input_boundary_conditions_are_bad()

    def __raise_error_if_input_boundary_conditions_are_bad(self) -> None:
        self.__raise_error_if_thermal_emission_is_bad()
        self.__raise_error_if_bottom_temperature_is_bad()
        self.__raise_error_if_top_temperature_is_bad()
        self.__raise_error_if_top_emissivity_is_bad()
        self.__raise_error_if_beam_flux_is_bad()
        self.__raise_error_if_isotropic_flux_is_bad()
        self.__raise_error_if_lambertian_bottom_boundary_is_bad()
        self.__raise_error_if_albedo_is_bad()
        self.__raise_error_if_incidence_beam_conditions_is_bad()

    def __raise_error_if_thermal_emission_is_bad(self) -> None:
        self.__raise_type_error_if_quantity_is_not_bool(
            self.__thermal_emission, 'thermal_emission')

    def __raise_error_if_bottom_temperature_is_bad(self) -> None:
        self.__raise_type_error_if_quantity_is_not_float(
            self.__bottom_temperature, 'bottom_temperature')
        self.__raise_value_error_if_quantity_is_not_finite(
            self.__bottom_temperature, 'bottom_temperature')
        self.__raise_value_error_if_quantity_is_negative(
            self.__bottom_temperature, 'bottom_temperature')

    def __raise_error_if_top_temperature_is_bad(self) -> None:
        self.__raise_type_error_if_quantity_is_not_float(
            self.__top_temperature, 'top_temperature')
        self.__raise_value_error_if_quantity_is_not_finite(
            self.__top_temperature, 'top_temperature')
        self.__raise_value_error_if_quantity_is_negative(
            self.__top_temperature, 'top_temperature')

    def __raise_error_if_top_emissivity_is_bad(self) -> None:
        self.__raise_type_error_if_quantity_is_not_float(
            self.__top_emissivity, 'top_emissivity')
        self.__raise_value_error_if_quantity_is_not_between_0_and_1(
            self.__top_emissivity, 'top_emissivity')

    def __raise_error_if_beam_flux_is_bad(self) -> None:
        self.__raise_type_error_if_quantity_is_not_float(
            self.__beam_flux, 'beam_flux')
        self.__raise_value_error_if_quantity_is_not_finite(
            self.__beam_flux, 'beam_flux')
        self.__raise_value_error_if_quantity_is_negative(
            self.__beam_flux, 'beam_flux')

    def __raise_error_if_isotropic_flux_is_bad(self) -> None:
        self.__raise_type_error_if_quantity_is_not_float(
            self.__isotropic_flux, 'isotropic_flux')
        self.__raise_value_error_if_quantity_is_not_finite(
            self.__isotropic_flux, 'isotropic_flux')
        self.__raise_value_error_if_quantity_is_negative(
            self.__isotropic_flux, 'isotropic_flux')

    def __raise_error_if_lambertian_bottom_boundary_is_bad(self) -> None:
        self.__raise_type_error_if_quantity_is_not_bool(
            self.__lambertian_bottom_boundary, 'lambertian_bottom_boundary')

    def __raise_error_if_albedo_is_bad(self) -> None:
        self.__raise_type_error_if_quantity_is_not_float(
            self.__albedo, 'albedo')
        self.__raise_value_error_if_quantity_is_not_between_0_and_1(
            self.__albedo, 'albedo')

    def __raise_error_if_incidence_beam_conditions_is_bad(self) -> None:
        self.__raise_type_error_if_quantity_is_not_bool(
            self.__incidence_beam_conditions, 'incidence_beam_conditions')

    @staticmethod
    def __raise_type_error_if_quantity_is_not_bool(
            quantity: Any, name: str) -> None:
        if not isinstance(quantity, bool):
            raise TypeError(f'{name} must be a bool.')

    @staticmethod
    def __raise_type_error_if_quantity_is_not_float(
            quantity: Any, name: str) -> None:
        if not isinstance(quantity, float):
            raise TypeError(f'{name} must be a float.')

    @staticmethod
    def __raise_value_error_if_quantity_is_not_finite(
            quantity: Any, name: str) -> None:
        if np.isinf(quantity):
            raise ValueError(f'{name} must be finite.')

    @staticmethod
    def __raise_value_error_if_quantity_is_negative(
            quantity: Any, name: str) -> None:
        if quantity < 0:
            raise ValueError(f'{name} must be non-negative.')

    @staticmethod
    def __raise_value_error_if_quantity_is_not_between_0_and_1(
            quantity: Any, name: str) -> None:
        if not 0 <= quantity <= 1:
            raise ValueError(f'{name} must be between 0 and 1.')

    @property
    def albedo(self) -> float:
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
    def beam_flux(self) -> float:
        """Get the input flux of the incident beam at the top boundary.

        Returns
        -------
        float
            The beam flux.

        Notes
        -----
        In DISORT, this variable is named "FBEAM".

        """
        return self.__beam_flux

    @property
    def bottom_temperature(self) -> float:
        """Get the input temperature at the bottom boundary.

        Returns
        -------
        float
            The bottom boundary temperature.

        Notes
        -----
        In DISORT, this variable is named "BTEMP".

        """
        return self.__bottom_temperature

    @property
    def incidence_beam_conditions(self) -> bool:
        """Get whether the model will only return albedo and transmissivity.

        Returns
        -------
        bool
            True if DISORT should only return albedo and transmissivity; False
            otherwise.

        """
        return self.__incidence_beam_conditions

    @property
    def isotropic_flux(self) -> float:
        """Get the input flux of isotropic sources at the top boundary.

        Returns
        -------
        float
            The isotropic flux.

        Notes
        -----
        In DISORT, this variable is named "FISOT".

        """
        return self.__isotropic_flux

    @property
    def lambertian_bottom_boundary(self) -> bool:
        """Get whether the bottom boundary used in the model will be Lambertian.

        Returns
        -------
        bool
            True if a Lambertian boundary is requested; False otherwise.

        Notes
        -----
        In DISORT, this variable is named "LAMBER".

        """
        return self.__lambertian_bottom_boundary

    @property
    def thermal_emission(self) -> bool:
        """Get whether thermal emission will be used in the model.

        Returns
        -------
        bool
            True if thermal emission is requested; False otherwise.

        Notes
        -----
        In DISORT, this variable is named "PLANK".

        """
        return self.__thermal_emission

    @property
    def top_emissivity(self) -> float:
        """Get the input emissivity at the top boundary.

        Returns
        -------
        float
            The top boundary emissivity.

        Notes
        -----
        In DISORT, this variable is named "TEMIS".

        """
        return self.__top_emissivity

    @property
    def top_temperature(self) -> float:
        """Get the input temperature at the top boundary.

        Returns
        -------
        float
            The top boundary temperature.

        Notes
        -----
        In DISORT, this variable is named "TTEMP".

        """
        return self.__top_temperature
