"""boundary_conditions.py contains a data structure to set variables related
to DISORT's boundary conditions.
"""
import numpy as np


# TODO: ibcnd
# TODO: type hints
# TODO: remake documentation
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
                 ibcnd: bool = False):
        """
        Parameters
        ----------
        thermal_emission: bool, optional
            Denote whether to use thermal emission. If True, DISORT will include
            thermal emission and will use the following variables:

                - bottom_temperature (from this class)
                - top_temperature (from this class)
                - top_emissivity (from this class)
                - low_wavenumber (from Angles)
                - high_wavenumber (from Angles)
                - temperature_boundaries (from ModelEquationOfState)

            If False, DISORT will save computation time, ignore all thermal
            emission, and ignore all the aforementioned variables. However,
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
            pyRT_DISORT for included surfaces.
        albedo: float, optional
            The surface albedo. Only used by DISORT if
            lambertian_bottom_boundary == True. Default is 0.0.
        ibcnd: bool, optional

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

        self.__raise_error_if_input_boundary_conditions_are_bad()

    def __raise_error_if_input_boundary_conditions_are_bad(self):
        self.__raise_error_if_thermal_emission_is_bad()
        self.__raise_error_if_bottom_temperature_is_bad()
        self.__raise_error_if_top_temperature_is_bad()
        self.__raise_error_if_top_emissivity_is_bad()
        self.__raise_error_if_beam_flux_is_bad()
        self.__raise_error_if_isotropic_flux_is_bad()
        self.__raise_error_if_lambertian_bottom_boundary_is_bad()
        self.__raise_error_if_albedo_is_bad()

    def __raise_error_if_thermal_emission_is_bad(self):
        self.__raise_type_error_if_quantity_is_not_bool(
            self.__thermal_emission, 'thermal_emission')

    def __raise_error_if_bottom_temperature_is_bad(self):
        self.__raise_type_error_if_input_is_not_float(
            self.__bottom_temperature, 'bottom_temperature')
        self.__raise_value_error_if_quantity_is_not_finite(
            self.__bottom_temperature, 'bottom_temperature')
        self.__raise_value_error_if_quantity_is_negative(
            self.__bottom_temperature, 'bottom_temperature')

    def __raise_error_if_top_temperature_is_bad(self):
        self.__raise_type_error_if_input_is_not_float(
            self.__top_temperature, 'top_temperature')
        self.__raise_value_error_if_quantity_is_not_finite(
            self.__top_temperature, 'top_temperature')
        self.__raise_value_error_if_quantity_is_negative(
            self.__top_temperature, 'top_temperature')

    def __raise_error_if_top_emissivity_is_bad(self):
        self.__raise_type_error_if_input_is_not_float(
            self.__top_emissivity, 'top_emissivity')
        self.__raise_value_error_if_quantity_is_not_between_0_and_1(
            self.__top_emissivity, 'top_emissivity')

    def __raise_error_if_beam_flux_is_bad(self):
        self.__raise_type_error_if_input_is_not_float(
            self.__beam_flux, 'beam_flux')
        self.__raise_value_error_if_quantity_is_not_finite(
            self.__beam_flux, 'beam_flux')
        self.__raise_value_error_if_quantity_is_negative(
            self.__beam_flux, 'beam_flux')

    def __raise_error_if_isotropic_flux_is_bad(self):
        self.__raise_type_error_if_input_is_not_float(
            self.__isotropic_flux, 'isotropic_flux')
        self.__raise_value_error_if_quantity_is_not_finite(
            self.__isotropic_flux, 'isotropic_flux')
        self.__raise_value_error_if_quantity_is_negative(
            self.__isotropic_flux, 'isotropic_flux')

    def __raise_error_if_lambertian_bottom_boundary_is_bad(self):
        self.__raise_type_error_if_quantity_is_not_bool(
            self.__lambertian_bottom_boundary, 'lambertian_bottom_boundary')

    def __raise_error_if_albedo_is_bad(self):
        self.__raise_type_error_if_input_is_not_float(self.__albedo, 'albedo')
        self.__raise_value_error_if_quantity_is_not_between_0_and_1(
            self.__albedo, 'albedo')

    @staticmethod
    def __raise_type_error_if_quantity_is_not_bool(quantity, name):
        if not isinstance(quantity, bool):
            raise TypeError(f'{name} must be a bool.')

    @staticmethod
    def __raise_type_error_if_input_is_not_float(quantity, name):
        if not isinstance(quantity, float):
            raise TypeError(f'{name} must be a float.')

    @staticmethod
    def __raise_value_error_if_quantity_is_not_finite(quantity, name):
        if np.isinf(quantity):
            raise ValueError(f'{name} must be finite.')

    @staticmethod
    def __raise_value_error_if_quantity_is_negative(quantity, name):
        if quantity < 0:
            raise ValueError(f'{name} must be non-negative.')

    @staticmethod
    def __raise_value_error_if_quantity_is_not_between_0_and_1(quantity, name):
        if not 0 <= quantity <= 1:
            raise ValueError(f'{name} must be between 0 and 1.')

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
    def bottom_temperature(self):
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
    def isotropic_flux(self):
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
    def lambertian_bottom_boundary(self):
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
    def thermal_emission(self):
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
    def top_emissivity(self):
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
    def top_temperature(self):
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
