"""The controller module holds miscellaneous classes responsible for creating
arrays that control how DISORT runs.
"""
from warnings import warn
import numpy as np


class ComputationalParameters:
    """Create a data structure for holding the size of computational parameters.

    ComputationalParameters holds the number of model layers, streams, moments,
    and angles. It also performs basic checks that these values are plausible.
    Objects of this class are meant to be be used as inputs to other classes.

    """

    # TODO: I'd like a better description of n_user_levels. It's not documented
    #  in the DISORT documentation...
    def __init__(self, n_layers: int, n_moments: int, n_streams: int,
                 n_azimuth: int, n_polar: int, n_user_levels: int) -> None:
        """
        Parameters
        ----------
        n_layers: int
            The number of layers to use in the model.
        n_moments: int
            The number of polynomial moments to use in the model. This number
            should be greater than or equal to 'n_streams' in problems with
            scattering. In problems without scattering, this variable is not
            used by DISORT.
        n_streams: int
            The number of streams (i.e. the number of computational polar
            angles) to use in the model. This number should be even and at least
            2. In general, the more streams used the more accurate DISORT's
            computations will be.
        n_azimuth: int
            The number of azimuthal angles where DISORT should return radiant
            quantities.
        n_polar: int
            The number of user-specified polar angles where DISORT should return
            radiant quantities. Only used by DISORT if user_angles (from
            ModelBehavior) == True.
        n_user_levels: int
            The number of user levels to use in the model.

        Raises
        ------
        TypeError
            Raised if any of the inputs cannot be converted to an int.
        ValueError
            Raised if any of the inputs are not positive finite.

        Warnings
        --------
        UserWarning
            Raised if n_streams is not even or if n_streams is greater than
            n_moments.

        """
        self.__n_layers = self.__make_n_layers(n_layers)
        self.__n_moments = self.__make_n_moments(n_moments)
        self.__n_streams = self.__make_n_streams(n_streams)
        self.__n_azimuth = self.__make_n_azimuth(n_azimuth)
        self.__n_polar = self.__make_n_polar(n_polar)
        self.__n_user_levels = self.__make_n_user_levels(n_user_levels)

        self.__warn_if_n_streams_is_not_even()
        self.__warn_if_n_streams_is_greater_than_n_moments()

    def __make_n_layers(self, n_layers: int) -> int:
        n_layers = self.__make_parameter(n_layers, 'n_layers')
        self.__raise_value_error_if_parameter_is_not_positive(
            n_layers, 'n_layers')
        return n_layers

    def __make_n_moments(self, n_moments: int) -> int:
        n_moments = self.__make_parameter(n_moments, 'n_moments')
        self.__raise_value_error_if_parameter_is_not_positive(
            n_moments, 'n_moments')
        return n_moments

    def __make_n_streams(self, n_streams: int) -> int:
        n_streams = self.__make_parameter(n_streams, 'n_streams')
        self.__raise_value_error_if_parameter_is_not_positive(
            n_streams, 'n_streams')
        return n_streams

    def __make_n_azimuth(self, n_azimuth: int) -> int:
        n_azimuth = self.__make_parameter(n_azimuth, 'n_azimuth')
        self.__raise_value_error_if_parameter_is_not_positive(
            n_azimuth, 'n_azimuth')
        return n_azimuth

    def __make_n_polar(self, n_polar: int) -> int:
        n_polar = self.__make_parameter(n_polar, 'n_polar')
        self.__raise_value_error_if_parameter_is_not_positive(
            n_polar, 'n_polar')
        return n_polar

    def __make_n_user_levels(self, n_user_levels: int) -> int:
        n_user_levels = self.__make_parameter(n_user_levels, 'n_user_levels')
        self.__raise_value_error_if_parameter_is_not_positive(
            n_user_levels, 'n_user_levels')
        return n_user_levels

    @staticmethod
    def __make_parameter(param: int, name: str) -> int:
        try:
            return int(param)
        except TypeError as te:
            raise TypeError(f'Cannot convert {name} to an int.') from te
        except ValueError as ve:
            raise ValueError(f'Cannot convert {name} to an int.') from ve
        except OverflowError as oe:
            raise ValueError(f'{name} must be finite.') from oe

    @staticmethod
    def __raise_value_error_if_parameter_is_not_positive(
            param: int, name: str) -> None:
        if param < 1:
            raise ValueError(f'{name} must be positive.')

    def __warn_if_n_streams_is_not_even(self) -> None:
        if self.__n_streams % 2 != 0:
            warn('n_streams should be even.')

    def __warn_if_n_streams_is_greater_than_n_moments(self) -> None:
        if self.__n_streams > self.__n_moments:
            warn('n_streams should not be greater than n_moments.')

    @property
    def n_layers(self) -> int:
        """Get the input number of layers.

        Returns
        -------
        int
            The number of layers.

        """
        return self.__n_layers

    @property
    def n_moments(self) -> int:
        """Get the input number of moments.

        Returns
        -------
        int
            The number of moments.

        """
        return self.__n_moments

    @property
    def n_streams(self) -> int:
        """Get the input number of streams.

        Returns
        -------
        int
            The number of streams.

        """
        return self.__n_streams

    @property
    def n_azimuth(self) -> int:
        """Get the input number of azimuthal angles.

        Returns
        -------
        int
            The number of azimuthal angles.

        """
        return self.__n_azimuth

    @property
    def n_polar(self) -> int:
        """Get the input number of user_specified polar angles.

        Returns
        -------
        int
            The number of polar angles.

        """
        return self.__n_polar

    @property
    def n_user_levels(self) -> int:
        """Get the input number of user levels.

        Returns
        -------
        int
            The number of user levels.

        """
        return self.__n_user_levels


# TODO: fix user_angles docstring. It's a mess
class ModelBehavior:
    """Create a data structure for holding the DISORT control variables.

    ModelBehavior holds the control flags that dictate DISORT's behavior. It
    also performs basic checks that the input control options are plausible.

    """
    def __init__(self, accuracy: float = 0.0, delta_m_plus: bool = False,
                 do_pseudo_sphere: bool = False, header: str = '',
                 print_variables: list[bool] = None,
                 radius: float = 6371.0) -> None:
        """
        Parameters
        ----------
        accuracy: float, optional
            The convergence criterion for azimuthal (Fourier cosine) series.
            Will stop when the following occurs twice: largest term being added
            is less than 'accuracy' times total series sum (twice because
            there are cases where terms are anomalously small but azimuthal
            series has not converged). Should be between 0 and 0.01 to avoid
            risk of serious non-convergence. Has no effect on problems lacking a
            beam source, since azimuthal series has only one term in that case.
            Default is 0.0.
        delta_m_plus: bool, optional
            Denote whether to use the delta-M+ method of Lin et al. (2018).
            Default is False.
        do_pseudo_sphere: bool, optional
            Denote whether to use a pseudo-spherical correction. Default is
            False.
        header: str, optional
            Use a 127- (or less) character header for prints, embedded in the
            DISORT banner. Input headers greater than 127 characters will be
            truncated. Setting header='' will eliminate both the banner and the
            header, and this is the only way to do so ('header' is not
            controlled by any of the 'print' flags); 'header' can be used
            to mark the progress of a calculation in which DISORT is called many
            times, while leaving all other printing turned off. Default is ''.

        print_variables: list[bool], optional
            Make a list of variables that control what DISORT prints. The 5
            booleans control whether each of the following is printed:

            1. Input variables (except PMOM)
            2. Fluxes
            3. Intensities at user levels and angles
            4. Planar transmissivity and planar albedo as a function of solar zenith angle (incidence_beam_conditions == True)
            5. PMOM for each layer (but only if 1. == True and only for layers with scattering)

            Default is None, which makes [False, False, False, False, False].
        radius: float, optional
            The planetary radius. This is presumably only used if
            do_pseudo_sphere == True, although there is no documentation on
            this. Default is 6371.0.


        Raises
        ------
        TypeError
            Raised if any inputs cannot be cast to the correct type.
        ValueError
            Raised if any inputs cannot be cast to the correct type or if
            print_variables does not have 5 elements.

        Warnings
        --------
        UserWarning
            Raised if accuracy is not between 0 and 0.01.

        """
        self.__accuracy = self.__make_accuracy(accuracy)
        self.__delta_m_plus = self.__make_delta_m_plus(delta_m_plus)
        self.__do_pseudo_sphere = self.__make_do_pseudo_sphere(do_pseudo_sphere)
        self.__header = self.__make_header(header)
        self.__print_variables = self.__make_print_variables(print_variables)
        self.__radius = self.__make_radius(radius)

        self.__warn_if_accuracy_is_outside_valid_range()

    def __make_accuracy(self, accuracy: float) -> float:
        return self.__cast_variable_to_float(accuracy, 'accuracy')

    def __make_delta_m_plus(self, delta_m_plus: bool) -> bool:
        return self.__cast_variable_to_bool(delta_m_plus, 'delta_m_plus')

    def __make_do_pseudo_sphere(self, do_pseudo_sphere: bool) -> bool:
        return self.__cast_variable_to_bool(
            do_pseudo_sphere, 'do_pseudo_sphere')

    # TODO: this does more than one thing
    @staticmethod
    def __make_header(header) -> str:
        try:
            header = str(header)
            if len(header) >= 127:
                header = header[:126]
            return header
        except TypeError as te:
            raise TypeError('header cannot be cast into a string.') from te

    # TODO: this does more than one thing
    @staticmethod
    def __make_print_variables(pvar) -> list[bool]:
        if pvar is None:
            return [False, False, False, False, False]
        else:
            try:
                prnt = [bool(f) for f in pvar]
                if len(prnt) != 5:
                    raise ValueError('print_variables should have 5 elements.')
                return prnt
            except TypeError as te:
                raise TypeError('print_variables should be a list of bools') \
                    from te

    def __make_radius(self, radius: float) -> float:
        radius = self.__cast_variable_to_float(radius, 'radius')
        self.__raise_value_error_if_radius_is_not_positive(radius)
        return radius

    @staticmethod
    def __cast_variable_to_float(variable: float, name: str) -> float:
        try:
            return float(variable)
        except TypeError as te:
            raise TypeError(f'{name} cannot be cast into a float.') from te
        except ValueError as ve:
            raise ValueError(f'{name} cannot be cast into a float.') from ve

    @staticmethod
    def __cast_variable_to_bool(variable: bool, name: str) -> bool:
        try:
            return bool(variable)
        except TypeError as te:
            raise TypeError(f'{name} cannot be cast into a boolean.') from te

    @staticmethod
    def __raise_value_error_if_radius_is_not_positive(radius) -> None:
        if radius <= 0:
            raise ValueError('radius must be positive.')

    def __warn_if_accuracy_is_outside_valid_range(self) -> None:
        if not 0 <= self.__accuracy <= 0.01:
            warn('accuracy is expected to be between 0 and 0.01.')

    @property
    def accuracy(self) -> float:
        """Get the input accuracy.

        Returns
        -------
        float
            The accuracy.

        Notes
        -----
        In DISORT, this variable is named "ACCUR".

        """
        return self.__accuracy

    @property
    def delta_m_plus(self) -> bool:
        """Get whether to use delta-M+ scaling.

        Returns
        -------
        bool
            True if delta-M+ scaling is requested; False otherwise.

        Notes
        -----
        In DISORT, this variable is named "DELTAMPLUS". There is no
        documentation on this variable.

        """
        return self.__delta_m_plus

    @property
    def do_pseudo_sphere(self) -> bool:
        """Get whether to perform a pseudo-spherical correction.

        Returns
        -------
        bool
            True if a pseudo-spherical correction is requested; False otherwise.

        Notes
        -----
        In DISORT, this variable is named "DO_PSEUDO_SPHERE". There is no
        documentation on this variable.

        """
        return self.__do_pseudo_sphere

    @property
    def header(self) -> str:
        """Get the characters that will appear in the DISORT banner.

        Returns
        -------
        str
            The characters in the DISORT banner.

        Notes
        -----
        In DISORT, this variable is named "HEADER".

        """
        return self.__header

    @property
    def print_variables(self) -> list[bool]:
        """Get the variables to print.

        Returns
        -------
        list[bool]
            The variables to print

        Notes
        -----
        In DISORT, this variable is named "PRNT".

        """
        return self.__print_variables

    @property
    def radius(self) -> float:
        """Get the planetary radius.

        Returns
        -------
        float
            The planetary radius.

        Notes
        -----
        In DISORT, this variable is named "EARTH_RADIUS". There is no
        documentation on this variable, though I'm presuming it could be any
        planetary radius.

        """
        return self.__radius







class UserLevel:
    """Create the user levels at which radiant quantities are returned.

    UserLevel checks that radiant quantities can be returned at user levels.

    """
    def __init__(self, cp: ComputationalParameters, mb: ModelBehavior,
                 optical_depth_output=None):
        """
        Parameters
        ----------
        cp: ComputationalParameters
            The computational parameters
        mb: ModelBehavior
            The model behavior
        optical_depth_output: np.ndarray or None
            If radiant quantities are to be returned at user specified optical
            depths (as specified in mb) this is the monotonically increasing 1D
            array of optical depths where they should be output. It must not
            exceed the column integrated optical depth of DTAUC. If radiant
            quantities can simply be returned at the boundary of every
            computational layer (user_optical_depths = False) this can be None,
            and this object will create an array of 0s for the output. Default
            is None.

        Notes
        -----
        The documentation suggests if user_optical_depths = False, this array
        is populated with optical depths at the boundaries of the computational
        layers. However, this seems equal to DTAUC to me but with a 0 appended.
        Additionally, making this case work would require changing f2py to
        make this an output array... which I'm not inclined to do if you won't
        use it.

        """
        self.__optical_depth_output = self.__make_optical_depth_output(
            cp, mb, optical_depth_output)

    @staticmethod
    def __make_optical_depth_output(cp: ComputationalParameters,
                                    mb: ModelBehavior, optical_depth_output):
        if mb.user_optical_depths and optical_depth_output is not None:
            # TODO: ensure this is monotonically increasing
            # TODO: ensure the last element does not exceed the total optical
            #  depth of the medium from dtauc
            return optical_depth_output
        else:
            return np.zeros(cp.n_user_levels)

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
