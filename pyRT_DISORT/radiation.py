"""The flux module contains data structures for holding the flux variables
used in DISORT.
"""
import numpy as np


class IncidentFlux:
    """A data structure for holding the incident fluxes.

    IncidentFlux creates scalars for the incident beam and isotropic fluxes and
    performs checks that the input fluxes are valid inputs to DISORT.

    """

    def __init__(self, beam_flux: float = np.pi,
                 isotropic_flux: float = 0.0) -> None:
        r"""
        Parameters
        ----------
        beam_flux
            The intensity of the incident beam at the top boundary.
            Ensure this variable and :code:`isotropic_flux` have the same units.
            Note that this is an infinitely wide beam. Default is pi.
        isotropic_flux
            The intensity of the incident beam at the top boundary. Ensure this
            variable and :code:`beam_flux` have the same units. Default is 0.0.

        Raises
        ------
        TypeError
            Raised if either input flux cannot be cast into a float.

        Notes
        -----
        If :code:`thermal_emission==True` (defined in :class:`.ThermalEmission`)
        this is assumed to have the same units as :code:`PLKAVG` (which defaults
        to [:math:`\frac{\text{W}}{\text{m}^2}`]) and the corresponding
        incident flux is :code:`umu0 * beam_flux` and
        :code:`pi * isotropic_flux`. If :code:`thermal_emission==False`,
        :code:`beam_flux` and :code:`isotropic_flux` have arbitrary units, and
        the output fluxes and
        intensities are assumed to have the same units as these variables.

        """
        self.__beam_flux = self.__cast_to_float(beam_flux, 'beam_flux')
        self.__isotropic_flux = \
            self.__cast_to_float(isotropic_flux, 'isotropic_flux')

    @staticmethod
    def __cast_to_float(flux: float, name: str) -> float:
        try:
            return float(flux)
        except ValueError as ve:
            raise TypeError(f'{name} could not be cast to a float.') from ve
        except TypeError as te:
            raise TypeError(f'{name} could not be cast to a float.') from te

    @property
    def beam_flux(self) -> float:
        """Get the input flux of the incident beam at the top boundary.

        Notes
        -----
        In DISORT, this variable is named :code:`FBEAM`.

        """
        return self.__beam_flux

    @property
    def isotropic_flux(self) -> float:
        """Get the input flux of isotropic sources at the top boundary.

        Notes
        -----
        In DISORT, this variable is named :code:`FISOT`.

        """
        return self.__isotropic_flux


class ThermalEmission:
    """A data structure for holding thermal emission variables.

    ThermalEmission creates variables needs to include thermal emission in
    DISORT.

    """

    def __init__(self, thermal_emission: bool = False,
                 bottom_temperature: float = 0.0, top_temperature: float = 0.0,
                 top_emissivity: float = 1.0) -> None:
        """
        Parameters
        ----------
        thermal_emission
            Denote whether to use thermal emission. If :code:`True`, DISORT will
            include thermal emission and will need the following variables:

                - :code:`bottom_temperature` (from this class)
                - :code:`top_temperature` (from this class)
                - :code:`top_emissivity` (from this class)
                - :code:`low_wavenumber` (from :class:`observation.Spectral`)
                - :code:`high_wavenumber` (from :class:`observation.Spectral`)
                - :code:`temperature` (from class:`eos.Hydrostatic`)

            If :code:`False`, DISORT will save computation time by ignoring all
            thermal emission and all of the aforementioned variables. Default is
            False.
        bottom_temperature
            The temperature of the bottom boundary [K]. Only used by DISORT if
            :code:`thermal_emission==True`. Default is 0.0.
        top_temperature
            The temperature of the top boundary [K]. Only used by DISORT if
            :code:`thermal_emission==True`. Default is 0.0.
        top_emissivity
            The emissivity of the top boundary. Only used by DISORT if
            :code:`thermal_emission==True`. Default is 1.0.

        Raises
        ------
        TypeError
            Raised if bottom_temperature, top_temperature, or top_emissivity
            cannot be cast to a float.
        ValueError
            Raised if bottom_temperature or top_temperature is negative, or if
            top_emissivity is not between 0 and 1.

        """
        self.__thermal_emission = self.__make_thermal_emission(thermal_emission)
        self.__bottom_temperature = \
            self.__make_temperature(bottom_temperature, 'bottom_temperature')
        self.__top_temperature = \
            self.__make_temperature(top_temperature, 'top_temperature')
        self.__top_emissivity = self.__make_emissivity(top_emissivity)

    def __make_thermal_emission(self, thermal_emission: bool) -> bool:
        return self.__cast_variable_to_bool(
            thermal_emission, 'thermal_emission')

    def __make_temperature(self, temperature: float, name: str) -> float:
        temperature = self.__cast_variable_to_float(temperature, name)
        self.__raise_value_error_if_temperature_is_unphysical(temperature, name)
        return temperature

    def __make_emissivity(self, top_emissivity: float) -> float:
        top_emissivity = self.__cast_variable_to_float(
            top_emissivity, 'top_emissivity')
        self.__raise_value_error_if_emissivity_not_in_range(top_emissivity)
        return top_emissivity

    @staticmethod
    def __cast_variable_to_bool(variable: bool, name: str) -> bool:
        try:
            return bool(variable)
        except TypeError as te:
            raise TypeError(f'{name} cannot be cast into a boolean.') from te

    @staticmethod
    def __cast_variable_to_float(variable: float, name: str) -> float:
        try:
            return float(variable)
        except TypeError as te:
            raise TypeError(f'{name} cannot be cast into a float.') from te
        except ValueError as ve:
            raise ValueError(f'{name} cannot be cast into a float.') from ve

    @staticmethod
    def __raise_value_error_if_temperature_is_unphysical(
            temperature: float, name: str) -> None:
        if temperature < 0 or np.isinf(temperature) or np.isnan(temperature):
            raise ValueError(f'{name} must be non-negative and finite.')

    @staticmethod
    def __raise_value_error_if_emissivity_not_in_range(
            top_emissivity: float) -> None:
        if not 0 <= top_emissivity <= 1:
            raise ValueError('top_emissivity must be between 0 and 1.')

    @property
    def thermal_emission(self) -> bool:
        """Get whether thermal emission will be used in the model.

        Notes
        -----
        In DISORT, this variable is named :code:`PLANK`.

        """
        return self.__thermal_emission

    @property
    def bottom_temperature(self) -> float:
        """Get the input temperature at the bottom boundary.

        Notes
        -----
        In DISORT, this variable is named :code:`BTEMP`.

        """
        return self.__bottom_temperature

    @property
    def top_temperature(self) -> float:
        """Get the input temperature at the top boundary.

        Notes
        -----
        In DISORT, this variable is named :code:`TTEMP`.

        """
        return self.__top_temperature

    @property
    def top_emissivity(self) -> float:
        """Get the input emissivity at the top boundary.

        Notes
        -----
        In DISORT, this variable is named :code:`TEMIS`.

        """
        return self.__top_emissivity
