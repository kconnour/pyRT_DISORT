"""boundary_conditions.py contains a data structure to set variables related
to DISORT's boundary conditions.
"""
import numpy as np


# TODO: inputs don't need to be in abc order
class BoundaryConditions:
    """A BoundaryConditions object holds boundary condition data for DISORT.

    BoundaryConditions is a data structure to set and hold the boundary
    quantities. Many of these quantities are only used if another variable
    is set to True. See the constructor docstring for more information on which
    variables are needed in which circumstances.
    """
    def __init__(self, albedo: float = 0.0, beam_flux: float = np.pi,
                 bottom_temperature: float = 0.0, fisot: float = 0.0,
                 ibcnd: int = 0, lambertian_bottom_boundary: bool = True,
                 thermal_emission: bool = False, top_emissivity: float = 1.0,
                 top_temperature: float = 0.0):
        """
        Parameters
        ----------
        albedo: float, optional
            The surface albedo. Only used by DISORT if
            lambertian_bottom_boundary == True and must be between 0 and 1.
            Default is 0.0.
        beam_flux: float, optional
            The intensity of the incident parallel beam the the top boundary. If
            plank == True, this has the same units as the variable "PLKAVG",
            which defaults to W / m**2. Ensure this variable and "fisot" have
            the same units. If plank == False, the radiant output units are the
            same as this quantity and fisot. Note that this is an infinitely
            wide beam. Default is pi.
        bottom_temperature: float, optional
            The temperature of the bottom boundary [K]. Only used by DISORT if
            plank == True. Default is 0.0.
        fisot
        ibcnd
        lambertian_bottom_boundary
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
            pyRT_DISORT will still compute all variables.
        top_emissivity: float, optional
            The emissivity of the top boundary. Only used by DISORT if plank
            == True and must be between 0 and 1. Default is 0.0.
        top_temperature: float, optional
            The temperature of the top boundary [K]. Only used by DISORT if
            plank == True. Default is 0.0.

        Raises
        ------
        TypeError
            Raised if the inputs are not of the correct type.
        ValueError
            Raised if the inputs are unphysical.

        """
        self.__albedo = albedo
        self.__beam_flux = beam_flux
        self.__bottom_temperature = bottom_temperature

        self.__thermal_emission = thermal_emission
        self.__top_emissivity = top_emissivity
        self.__top_temperature = top_temperature

        self.__raise_error_if_input_boundary_conditions_are_bad()

    def __raise_error_if_input_boundary_conditions_are_bad(self):
        self.__raise_error_if_albedo_is_bad()
        self.__raise_error_if_beam_flux_is_bad()
        self.__raise_error_if_bottom_temperature_is_bad()

        self.__raise_error_if_thermal_emission_is_bad()
        self.__raise_error_if_top_emissivity_is_bad()
        self.__raise_error_if_top_temperature_is_bad()

    def __raise_error_if_albedo_is_bad(self):
        self.__raise_type_error_if_input_is_not_float(self.__albedo, 'albedo')
        self.__raise_value_error_if_quantity_is_not_between_0_and_1(
            self.__albedo, 'albedo')

    def __raise_error_if_beam_flux_is_bad(self):
        self.__raise_type_error_if_input_is_not_float(
            self.__beam_flux, 'beam_flux')
        self.__raise_value_error_if_quantity_is_not_finite(
            self.__beam_flux, 'beam_flux')
        self.__raise_value_error_if_quantity_is_negative(
            self.__beam_flux, 'beam_flux')

    def __raise_error_if_bottom_temperature_is_bad(self):
        self.__raise_type_error_if_input_is_not_float(
            self.__bottom_temperature, 'bottom_temperature')
        self.__raise_value_error_if_quantity_is_not_finite(
            self.__bottom_temperature, 'bottom_temperature')
        self.__raise_value_error_if_quantity_is_negative(
            self.__bottom_temperature, 'bottom_temperature')

    def __raise_error_if_thermal_emission_is_bad(self):
        self.__raise_type_error_if_quantity_is_not_bool(
            self.__thermal_emission, 'thermal_emission')

    def __raise_error_if_top_emissivity_is_bad(self):
        self.__raise_type_error_if_input_is_not_float(
            self.__top_emissivity, 'top_emissivity')
        self.__raise_value_error_if_quantity_is_not_between_0_and_1(
            self.__top_emissivity, 'top_emissivity')

    def __raise_error_if_top_temperature_is_bad(self):
        self.__raise_type_error_if_input_is_not_float(
            self.__top_temperature, 'top_temperature')
        self.__raise_value_error_if_quantity_is_not_finite(
            self.__top_temperature, 'top_temperature')
        self.__raise_value_error_if_quantity_is_negative(
            self.__top_temperature, 'top_temperature')

    @staticmethod
    def __raise_type_error_if_input_is_not_float(quantity, name):
        if not isinstance(quantity, float):
            raise TypeError(f'{name} must be a float.')

    @staticmethod
    def __raise_value_error_if_quantity_is_not_between_0_and_1(quantity, name):
        if not 0 <= quantity <= 1:
            raise ValueError(f'{name} must be between 0 and 1.')

    @staticmethod
    def __raise_value_error_if_quantity_is_not_finite(quantity, name):
        if np.isinf(quantity):
            raise ValueError(f'{name} must be finite.')

    @staticmethod
    def __raise_value_error_if_quantity_is_negative(quantity, name):
        if quantity < 0:
            raise ValueError(f'{name} must be non-negative.')

    @staticmethod
    def __raise_type_error_if_quantity_is_not_bool(quantity, name):
        if not isinstance(quantity, bool):
            raise TypeError(f'{name} must be a bool.')

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

    @property
    def bottom_temperature(self):
        """Get the input bottom boundary temperature.

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
    def thermal_emission(self):
        """Get whether to use thermal emission.

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
        """Get the input top boundary emissivity.

        Returns
        -------
        float
            The top boundary emissivity

        Notes
        -----
        In DISORT, this variable is named "TEMIS".

        """
        return self.__top_emissivity

    @property
    def top_temperature(self):
        """Get the input top boundary temperature.

        Returns
        -------
        float
            The top boundary temperature.

        Notes
        -----
        In DISORT, this variable is named "TTEMP".

        """
        return self.__top_temperature
