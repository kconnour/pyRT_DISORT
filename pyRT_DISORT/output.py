"""The output module contains data structures for controlling what DISORT
outputs.
"""
import numpy as np


class OutputArrays:
    """Create a data structure to make the DISORT output arrays.

    OutputArrays creates arrays of 0s that are designed to get populated with
    values as DISORT runs.

    """

    def __init__(self, n_polar: int, n_user_levels: int,
                 n_azimuth: int) -> None:
        """
        Parameters
        ----------
        n_polar
            The number of polar angles to use in the model.
        n_user_levels
            The number of user levels to use in the model.
        n_azimuth
            The number of azimuthal angles to use in the model.

        """
        self.__n_polar = n_polar
        self.__n_user_levels = n_user_levels
        self.__n_azimuth = n_azimuth

        self.__albedo_medium = self.__make_albedo_medium()
        self.__diffuse_up_flux = self.__make_diffuse_up_flux()
        self.__diffuse_down_flux = self.__make_diffuse_down_flux()
        self.__direct_beam_flux = self.__make_direct_beam_flux()
        self.__flux_divergence = self.__make_flux_divergence()
        self.__intensity = self.__make_intensity()
        self.__mean_intensity = self.__make_mean_intensity()
        self.__transmissivity_medium = self.__make_transmissivity_medium()

    def __make_albedo_medium(self) -> np.ndarray:
        return np.zeros(self.__n_polar)

    def __make_diffuse_up_flux(self) -> np.ndarray:
        return np.zeros(self.__n_user_levels)

    def __make_diffuse_down_flux(self) -> np.ndarray:
        return np.zeros(self.__n_user_levels)

    def __make_direct_beam_flux(self) -> np.ndarray:
        return np.zeros(self.__n_user_levels)

    def __make_flux_divergence(self) -> np.ndarray:
        return np.zeros(self.__n_user_levels)

    def __make_intensity(self) -> np.ndarray:
        return np.zeros((self.__n_polar, self.__n_user_levels,
                         self.__n_azimuth))

    def __make_mean_intensity(self) -> np.ndarray:
        return np.zeros(self.__n_user_levels)

    def __make_transmissivity_medium(self) -> np.ndarray:
        return np.zeros(self.__n_polar)

    @property
    def albedo_medium(self) -> np.ndarray:
        """Get the albedo of the medium output array.

        Notes
        -----
        In DISORT, this variable is named :code:`ALBMED`.

        """
        return self.__albedo_medium

    @property
    def diffuse_up_flux(self) -> np.ndarray:
        """Get the diffuse upward flux output array.

        Returns
        -------
        np.ndarray
            The diffuse upward flux.

        Notes
        -----
        In DISORT, this variable is named :code:`FLUP`.

        """
        return self.__diffuse_up_flux

    @property
    def diffuse_down_flux(self) -> np.ndarray:
        """Get the diffuse downward flux output array, which will be the total
        downward flux minus the direct beam flux.

        Notes
        -----
        In DISORT, this variable is named :code:`RFLDN`.

        """
        return self.__diffuse_down_flux

    @property
    def direct_beam_flux(self) -> np.ndarray:
        """Get the direct beam flux output array.

        Notes
        -----
        In DISORT, this variable is named :code:`RFLDIR`.

        """
        return self.__direct_beam_flux

    @property
    def flux_divergence(self) -> np.ndarray:
        """Get the flux divergence output array, which is will represent
        (d(net_flux) / d(optical_depth)). This is an exact result.

        Notes
        -----
        In DISORT, this variable is named :code:`DFDT`.

        """
        return self.__flux_divergence

    @property
    def intensity(self) -> np.ndarray:
        """Get the intensity output array.

        Notes
        -----
        In DISORT, this variable is named :code:`UU`.

        """
        return self.__intensity

    @property
    def mean_intensity(self) -> np.ndarray:
        """Get the mean intensity output array.

        Notes
        -----
        In DISORT, this variable is named ":code:`UAVG`.

        """
        return self.__mean_intensity

    @property
    def transmissivity_medium(self) -> np.ndarray:
        """Get the transmissivity of the medium output array.

        Notes
        -----
        In DISORT, this variable is named :code:`TRNMED`.

        """
        return self.__transmissivity_medium


class OutputBehavior:
    """A data structure that holds DISORT output behavior switches.

    OutputBehavior provides some default output behavior for DISORT. All the
    parameters can be overriden to fit user specifications.

    """
    def __init__(self, incidence_beam_conditions: bool = False,
                 only_fluxes: bool = False, user_angles: bool = True,
                 user_optical_depths: bool = False) -> None:
        """
        Parameters
        ----------
        incidence_beam_conditions
            Denote what functions of the incidence beam angle should be
            included. If :code:`True`, return the albedo and transmissivity of
            the
            entire medium as a function of incidence beam angle. In this case,
            the following inputs are the only ones considered by DISORT:

            - :code:`n_layers` (from :class:`controller.ComputationalParameters`)
            - DTAUC
            - SSALB
            - PMOM
            - :code:`n_streams` (from :class:`controller.ComputationalParameters`)
            - :code:`user_angles` (from this class)
            - :code:`n_polar` (from :class:`controller.ComputationalParameters`)
            - :code:`mu` (from :class:`observation.Spectral`)
            - ALBEDO
            - :code:`print_variables` (from :class:`controller.ComputationalParameters`)
            - :code:`header` (from :class:`controller.ComputationalParameters`)

            PLANK is assumed to be False, LAMBER is assumed to be True, and
            ONLYFL must be False. The only output is ALBMED and TRNMED. The
            intensities are not corrected for delta-M+ correction.

            If False, this is accommodates any general case of boundary
            conditions including beam illumination from the top, isotropic
            illumination from the top, thermal emission from the top, internal
            thermal emission, reflection at the bottom, and/or thermal emission
            from the bottom. Default is False.
        only_fluxes
            Determine if only the fluxes are returned by the model. If True,
            return fluxes, flux divergences, and mean intensities; if False,
            return all those quantities and intensities. In addition, if True
            the number of polar angles can be 0, the number of azimuthal angles
            can be 0, phi is not used, and all values of intensity (UU) will be
            set to 0 (these are defined in ComputationalParameters). Default is
            False.
        user_angles
            Denote whether radiant quantities should be returned at user angles.
            If False, radiant quantities are to be returned at computational
            polar angles. Also, UMU will return the cosines of the computational
            polar angles and n_polar (from ComputationalParameters) will return
            their number ( = n_streams).UMU must
            be large enough to contain n_streams elements. If True,
            radiant quantities are to be returned at user-specified polar
            angles, as follows: NUMU No. of polar angles (zero is a legal value
            only when 'only_fluxes' == True ) UMU(IU) IU=1 to NUMU, cosines of
            output polar angles in increasing order---starting with negative
            (downward) values (if any) and on through positive (upward)
            values; *** MUST NOT HAVE ANY ZERO VALUES ***. Default is True.
        user_optical_depths
            Denote whether radiant quantities are returned at user-specified
            optical depths. Default is False.
        """
        self.__incidence_beam_conditions = \
            self.__make_incidence_beam_conditions(incidence_beam_conditions)
        self.__only_fluxes = self.__make_only_fluxes(only_fluxes)
        self.__user_angles = self.__make_user_angles(user_angles)
        self.__user_optical_depths = self.__make_user_optical_depths(
            user_optical_depths)

    def __make_incidence_beam_conditions(self, ibcnd: bool) -> bool:
        return self.__cast_variable_to_bool(ibcnd, 'incidence_beam_conditions')

    def __make_only_fluxes(self, only_fluxes: bool) -> bool:
        return self.__cast_variable_to_bool(only_fluxes, 'only_fluxes')

    def __make_user_angles(self, user_angles: bool) -> bool:
        return self.__cast_variable_to_bool(user_angles, 'user_angles')

    def __make_user_optical_depths(self, user_optical_depths: bool) -> bool:
        return self.__cast_variable_to_bool(
            user_optical_depths, 'user_optical_depths')

    @staticmethod
    def __cast_variable_to_bool(variable: bool, name: str) -> bool:
        try:
            return bool(variable)
        except TypeError as te:
            raise TypeError(f'{name} cannot be cast into a boolean.') from te

    @property
    def incidence_beam_conditions(self) -> bool:
        """Get whether the model will only return albedo and transmissivity.

        Notes
        -----
        In DISORT, this variable is named :code:`IBCND`.

        """
        return self.__incidence_beam_conditions

    @property
    def only_fluxes(self) -> bool:
        """Get whether DISORT should only return fluxes.

        Notes
        -----
        In DISORT, this variable is named :code:`ONLYFL`.

        """
        return self.__only_fluxes

    @property
    def user_angles(self) -> bool:
        """Get whether radiant quantities should be returned at user angles.

        Notes
        -----
        In DISORT, this variable is named :code:`USRANG`.

        """
        return self.__user_angles

    @property
    def user_optical_depths(self) -> bool:
        """Get whether radiant quantities should be returned at user optical
        depths.

        Notes
        -----
        In DISORT, this variable is named :code:`USRTAU`.

        """
        return self.__user_optical_depths


class UserLevel:
    """Create the user levels at which radiant quantities are returned.

    UserLevel checks that radiant quantities can be returned at user levels.

    """
    def __init__(self, optical_depth_output: np.ndarray = None,
                 n_user_levels: int = 1) -> None:
        """
        Parameters
        ----------
        optical_depth_output
            If radiant quantities are to be returned at user specified optical
            depths (as specified in mb) this is the monotonically increasing 1D
            array of optical depths where they should be output. It must not
            exceed the column integrated optical depth of DTAUC. If radiant
            quantities can simply be returned at the boundary of every
            computational layer (user_optical_depths = False) this can be None,
            and this object will create an array of 0s for the output. Default
            is None.
        n_user_levels
            The number of user levels to instantiate. Only required if
            :code:`optical_depth_output` is not None.

        Notes
        -----
        The documentation suggests if user_optical_depths = False, this array
        is populated with optical depths at the boundaries of the computational
        layers. However, this seems equal to DTAUC to me but with a 0 appended.
        Additionally, making this case work would require changing f2py to
        make this an output array... which I'm not inclined to do if no one will
        use it.

        """
        self.__optical_depth_output = self.__make_optical_depth_output(
            optical_depth_output, n_user_levels)

    @staticmethod
    def __make_optical_depth_output(optical_depth_output, n_user_levels):
        if optical_depth_output is not None:
            # TODO: ensure this is monotonically increasing
            # TODO: ensure the last element does not exceed the total optical
            #  depth of the medium from dtauc
            return optical_depth_output
        else:
            return np.zeros(n_user_levels)

    @property
    def optical_depth_output(self) -> np.ndarray:
        """Get the input user optical depth array at user levels.

        Returns
        -------
        np.ndarray
            The optical depth at user levels.

        Notes
        -----
        In DISORT, this variable is named "UTAU".

        """
        return self.__optical_depth_output
