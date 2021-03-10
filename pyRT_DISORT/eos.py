"""The eos module contains data structures to compute and hold equation of state
variables used throughout pyRT_DISORT.
"""
import numpy as np
from scipy.constants import Boltzmann
from scipy.integrate import quadrature as quad


class Hydrostatic:
    """A data structure that computes a hydrostatic equation of state.

    Hydrostatic accepts equation of state variables and computes atmospheric
    properties from them, assuming the atmosphere follows the equation

    .. math::
       :label: hydrostatic_equation

       P = n k_B T

    where :math:`P` is the pressure, :math:`n` is the number density,
    :math:`k_B` is Boltzmann's constant, and :math:`T` is the temperature.

    """

    def __init__(self, altitude_grid: np.ndarray, pressure_grid: np.ndarray,
                 temperature_grid: np.ndarray, altitude_boundaries: np.ndarray,
                 particle_mass: float, gravity: float) -> None:
        r"""
        Parameters
        ----------
        altitude_grid
            The altitude grid [km] over which the equation of state variables
            are defined. To keep with DISORT's conventions, *these must be
            decreasing* over the 0 :sup:`th` dimension.
        pressure_grid
            The pressure [Pa] at all values in :code:`altitude_grid`.
        temperature_grid
            The temperature [K] at all values in :code:`altitude_grid`.
        altitude_boundaries
            The altitude to interpolate the pressure and temperature.
        particle_mass
            The average mass [kg] of atmospheric particles.
        gravity
            The gravitational acceleration
            [:math:`\frac{\text{kg m}}{\text{s}^2}`] of the atmosphere.

        Notes
        -----
        The input can be ND arrays, as long as they have compatible shapes. In
        this scenario, :code:`altitude_grid`, :code:`pressure_grid`, and
        :code:`temperature_grid` must be of shape Mx(pixels) whereas
        :code:`altitude_boundaries` must be of shape Nx(pixels).

        Also, scipy's Gaussian quadrature routine becomes less accurate the
        larger the atmosphere's scale height is. I'm working to reduce the
        errors. In the meantime the column density is fairly close to analytical
        results but should be improved.

        """

        self.__altitude_grid = altitude_grid
        self.__altitude = altitude_boundaries

        self.__n_layers = self.__extract_n_layers()

        self.__pressure = \
            self.__interpolate_to_boundary_alts(pressure_grid)
        self.__temperature = \
            self.__interpolate_to_boundary_alts(temperature_grid)
        self.__number_density = \
            self.__compute_number_density(self.__pressure, self.__temperature)
        self.__column_density = \
            self.__compute_column_density()
        self.__scale_height = \
            self.__compute_scale_height(particle_mass, gravity)

    def __extract_n_layers(self) -> int:
        return self.__altitude.shape[0] - 1

    # TODO: Ideally I'd like to vectorize this
    def __interpolate_to_boundary_alts(self, grid: np.ndarray) -> np.ndarray:
        flattened_altitude_grid = \
            self.__flatten_along_pixel_dimension(self.__altitude_grid)
        flattened_boundaries = \
            self.__flatten_along_pixel_dimension(self.__altitude)
        flattened_quantity_grid = self.__flatten_along_pixel_dimension(grid)
        interpolated_quantity = np.zeros(flattened_boundaries.shape)
        for pixel in range(flattened_boundaries.shape[1]):
            interpolated_quantity[:, pixel] = \
                np.interp(flattened_boundaries[:, pixel],
                          np.flip(flattened_altitude_grid[:, pixel]),
                          np.flip(flattened_quantity_grid[:, pixel]))
        return interpolated_quantity.reshape(self.__altitude.shape)

    @staticmethod
    def __flatten_along_pixel_dimension(grid: np.ndarray) -> np.ndarray:
        return grid.reshape(grid.shape[0], int(grid.size / grid.shape[0]))

    @staticmethod
    def __compute_number_density(pressure: np.ndarray,
                                 temperature: np.ndarray) -> np.ndarray:
        return pressure / temperature / Boltzmann

    # TODO: Ideally I'd like to vectorize this
    # TODO: Mike said to do this in log(z) space. Is this still necessary?
    def __compute_column_density(self) -> np.ndarray:
        flattened_boundaries = \
            np.flipud(self.__flatten_along_pixel_dimension(self.__altitude)
                      * 1000)
        flattened_pressure = \
            np.flipud(self.__flatten_along_pixel_dimension(self.__pressure))
        flattened_temperature = \
            np.flipud(self.__flatten_along_pixel_dimension(self.__temperature))
        column_density = np.zeros((self.__n_layers,
                                   flattened_boundaries.shape[1]))
        for pixel in range(flattened_boundaries.shape[1]):
            colden = [quad(self.__make_number_density_at_altitude,
                           flattened_boundaries[i, pixel],
                           flattened_boundaries[i+1, pixel],
                           args=(flattened_boundaries[:, pixel],
                                 flattened_pressure[:, pixel],
                                 flattened_temperature[:, pixel]))[0]
                      for i in range(self.__n_layers)]
            column_density[:, pixel] = np.flip(np.array(colden))
        if np.ndim(self.__altitude) == 1:
            return np.squeeze(column_density)
        else:
            return column_density.reshape((column_density.shape[0],) +
                                          self.__altitude.shape[1:])

    def __make_number_density_at_altitude(
            self, z: float, alt_grid: np.ndarray, pressure: np.ndarray,
            temperature: np.ndarray) -> np.ndarray:
        p = np.interp(z, alt_grid, pressure)
        t = np.interp(z, alt_grid, temperature)
        return self.__compute_number_density(p, t)

    def __compute_scale_height(self, particle_mass: float,
                               gravity: float) -> np.ndarray:
        return Boltzmann * self.__temperature / (particle_mass * gravity)

    @property
    def n_layers(self) -> int:
        """Get the number of layers in the model. This value is inferred from
        the 0 :sup:`th` dimension of :code:`altitude_boundaries`.

        """
        return self.__n_layers

    @property
    def altitude(self) -> np.ndarray:
        """Get the input boundary altitude [km].

        """
        return self.__altitude

    @property
    def pressure(self) -> np.ndarray:
        """Get the pressure [Pa] at the boundary altitude. This variable is
        obtained by linearly interpolating the input pressure onto
        :code:`altitude_boundaries` for each pixel.

        """
        return self.__pressure

    @property
    def temperature(self) -> np.ndarray:
        """Get the temperature [K] at the boundary altitude. This variable is
        obtained by linearly interpolating the input temperature onto
        :code:`altitude_boundaries` for each pixel.

        Notes
        -----
        In DISORT, this variable is named :code:`TEMPER`.

        """
        return self.__temperature

    @property
    def number_density(self) -> np.ndarray:
        r"""Get the number density [:math:`\frac{\text{particles}}{\text{m}^3}`]
        at the boundary altitude. This variable is obtained by getting the
        pressure and temperature at the boundary altitude, then solving
        :eq:`hydrostatic_equation`.

        """
        return self.__number_density

    # TODO: fix the line break
    @property
    def column_density(self) -> np.ndarray:
        r"""Get the column density [:math:`\frac{\text{particles}}{\text{m}^2}`]
        at the boundary *layers*. This array is obtained by linearly
        interpolating the input pressure and temperature of each pixel, then
        integrating (using Gaussian quadrature) between the pixel's boundary
        altitude such that

        .. math::
           N = \int n(z) dz

        is satisfied, where :math:`N` is the column density and :math:`n(z)` is
        the number density.

        """
        return self.__column_density

    @property
    def scale_height(self) -> np.ndarray:
        r"""Get the scale height [km] at the boundary altitude. For a
        hydrostatic atmosphere, the scale height is defined as

        .. math::
           H = \frac{k_B T}{mg}

        where :math:`H` is the scale height, :math:`k_B` is Boltzmann's
        constant, :math:`T` is the temperature, :math:`m` is the average mass
        of an atmospheric particle, and :math:`g` is the planetary gravity.

        Notes
        -----
        In DISORT, this variable is named :code:`H_LYR`. Despite the name, this
        variable should have length of :code:`n_layers + 1`. It is only used if
        :code:`do_pseudo_sphere==True` (defined in
        :class:`controller.ModelBehavior`).

        """
        return self.__scale_height
