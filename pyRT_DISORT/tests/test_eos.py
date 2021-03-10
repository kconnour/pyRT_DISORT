import numpy as np
import pytest
from scipy.constants import Boltzmann
from pyRT_DISORT.eos import Hydrostatic


# TODO: these tests look awful to read
class TestHydrostatic:
    pass


class TestAltitude(TestHydrostatic):
    def setup(self) -> None:
        altitude_grid = np.broadcast_to(np.linspace(100, 0, num=20),
                                        (5, 20)).T
        exp_pressure = 1000 * np.exp(-altitude_grid / 10)
        const_temper = np.broadcast_to(np.linspace(100, 100, num=20),
                                       (5, 20)).T
        self.same_boundaries = altitude_grid
        mass = 7.3 * 10 ** -26
        g = 3.71
        self.same_hydro = Hydrostatic(altitude_grid, exp_pressure, const_temper,
                                      self.same_boundaries, mass, g)

    def test_altitude_is_unchanged(self) -> None:
        assert np.array_equal(self.same_boundaries, self.same_hydro.altitude)

    def test_altitude_is_read_only(self) -> None:
        with pytest.raises(AttributeError):
            self.same_hydro.altitude = 0


class TestNLayers(TestHydrostatic):
    def setup(self) -> None:
        altitude_grid = np.broadcast_to(np.linspace(100, 0, num=20),
                                        (5, 20)).T
        exp_pressure = 1000 * np.exp(-altitude_grid / 10)
        const_temper = np.broadcast_to(np.linspace(100, 100, num=20),
                                       (5, 20)).T
        same_boundaries = altitude_grid
        mass = 7.3 * 10 ** -26
        g = 3.71
        self.same_hydro = Hydrostatic(altitude_grid, exp_pressure,
                                      const_temper, same_boundaries, mass, g)

    def test_n_layers_is_determined_by_altitude_boundaries(self) -> None:
        assert self.same_hydro.n_layers == 19

    def test_n_layers_is_read_only(self) -> None:
        with pytest.raises(AttributeError):
            self.same_hydro.n_layers = 19


class TestPressure(TestHydrostatic):
    def setup(self) -> None:
        altitude_grid = np.broadcast_to(np.linspace(100, 0, num=20), (5, 20)).T
        self.exp_pressure = 1000 * np.exp(-altitude_grid / 10)
        const_temper = np.broadcast_to(np.linspace(100, 100, num=20), (5, 20)).T
        self.same_boundaries = altitude_grid
        mass = 7.3 * 10**-26
        g = 3.71
        self.same_hydro = Hydrostatic(altitude_grid, self.exp_pressure,
                                      const_temper, self.same_boundaries, mass,
                                      g)

        self.different_boundaries = np.broadcast_to(
            np.linspace(100, 0, num=15), (5, 15)).T
        self.different_hydro = Hydrostatic(altitude_grid, self.exp_pressure,
                                           const_temper,
                                           self.different_boundaries, mass, g)

    def test_pressure_is_unchanged_if_altitude_grids_are_equal(self) -> None:
        assert np.array_equal(self.exp_pressure, self.same_hydro.pressure)

    def test_pressure_is_linearly_interpolated(self) -> None:
        assert np.array_equal(self.exp_pressure[[0, -1], :],
                              self.different_hydro.pressure[[0, -1], :])
        assert not np.array_equal(self.exp_pressure,
                                  self.different_hydro.pressure)

    def test_pressure_is_read_only(self):
        with pytest.raises(AttributeError):
            self.same_hydro.pressure = 0

    def test_pressure_has_same_shape_as_altitude_boundaries(self) -> None:
        assert self.different_boundaries.shape == \
               self.different_hydro.pressure.shape


class TestTemperature:
    def setup(self) -> None:
        altitude_grid = np.broadcast_to(np.linspace(100, 0, num=20), (5, 20)).T
        exp_pressure = 1000 * np.exp(-altitude_grid / 10)
        self.lin_temper = np.broadcast_to(np.linspace(100, 250, num=20), (5, 20)).T
        self.same_boundaries = altitude_grid
        mass = 7.3 * 10**-26
        g = 3.71
        self.same_hydro = Hydrostatic(altitude_grid, exp_pressure,
                                      self.lin_temper,
                                      self.same_boundaries, mass, g)

        self.different_boundaries = np.broadcast_to(
            np.linspace(100, 0, num=15), (5, 15)).T
        self.different_hydro = Hydrostatic(altitude_grid, exp_pressure,
                                           self.lin_temper,
                                           self.different_boundaries, mass, g)

    def test_temperature_is_unchanged_if_altitude_grids_are_equal(self) -> None:
        assert np.array_equal(self.lin_temper, self.same_hydro.temperature)

    def test_temperature_is_linearly_interpolated(self) -> None:
        assert np.array_equal(self.lin_temper[[0, -1], :],
                              self.different_hydro.temperature[[0, -1], :])
        assert not np.array_equal(self.lin_temper,
                                  self.different_hydro.temperature)

    def test_temperature_is_read_only(self):
        with pytest.raises(AttributeError):
            self.same_hydro.temperature = 0

    def test_temperature_has_same_shape_as_altitude_boundaries(self) -> None:
        assert self.different_boundaries.shape == \
               self.different_hydro.temperature.shape


class TestNumberDensity(TestHydrostatic):
    def setup(self) -> None:
        altitude_grid = np.broadcast_to(np.linspace(100, 0, num=20),
                                        (5, 20)).T
        self.exp_pressure = 1000 * np.exp(-altitude_grid / 10)
        const_temper = np.broadcast_to(np.linspace(100, 100, num=20),
                                       (5, 20)).T
        self.same_boundaries = altitude_grid
        mass = 7.3 * 10 ** -26
        g = 3.71
        self.same_hydro = Hydrostatic(altitude_grid, self.exp_pressure,
                                      const_temper,
                                      self.same_boundaries, mass, g)

        self.different_boundaries = np.broadcast_to(
            np.linspace(100, 0, num=15), (5, 15)).T
        self.different_hydro = Hydrostatic(altitude_grid, self.exp_pressure,
                                           const_temper,
                                           self.different_boundaries, mass, g)

    def test_number_density_matches_analytic_cases(self) -> None:
        assert np.array_equal(self.same_hydro.number_density,
                              self.exp_pressure / 100 / Boltzmann)

    def test_number_density_is_read_only(self):
        with pytest.raises(AttributeError):
            self.same_hydro.number_density = 0

    def test_number_density_has_same_shape_as_altitude_boundaries(self) -> None:
        assert self.different_boundaries.shape == \
               self.different_hydro.number_density.shape


class TestColumnDensity(TestHydrostatic):
    def setup(self) -> None:
        self.altitude_grid = np.linspace(100, 0, num=21)
        self.exp_pressure = 1000 * np.exp(-self.altitude_grid / 10)
        const_pressure = np.linspace(1000, 1000, num=21)
        const_temper = np.linspace(200, 200, num=21)
        self.same_boundaries = self.altitude_grid
        self.mass = 7.3 * 10 ** -26
        self.g = 3.71
        self.constant_hydro = Hydrostatic(self.altitude_grid, const_pressure, const_temper, self.same_boundaries, self.mass, self.g)
        self.same_hydro = Hydrostatic(self.altitude_grid, self.exp_pressure,
                                      const_temper,
                                      self.same_boundaries, self.mass, self.g)

    def test_column_density_matches_analytic_solution(self) -> None:
        # For constant P and T, N = P/kT * z
        analytic_constant = 5 / Boltzmann * (
                    self.altitude_grid[:-1] - self.altitude_grid[1:]) * 1000
        assert np.allclose(analytic_constant, self.constant_hydro.column_density)

        # For a constant T, N = -P0 / mg * e**(-z / H)
        H = Boltzmann * 200 / self.mass / self.g / 1000
        analytic_solution = -1000 / self.mass / self.g * \
                            (np.exp(-self.altitude_grid[:-1] / H) -
                             np.exp(-self.altitude_grid[1:] / H))
        assert np.allclose(analytic_solution, self.same_hydro.column_density, rtol=0.2)

    def test_number_density_is_read_only(self):
        with pytest.raises(AttributeError):
            self.same_hydro.column_density = 0

    def test_number_density_has_same_shape_as_altitude_boundaries(self) -> None:
        assert (self.same_boundaries.shape[0] - 1,) == \
                self.same_hydro.column_density.shape


class TestScaleHeight(TestHydrostatic):
    def setup(self) -> None:
        self.altitude_grid = np.linspace(100, 0, num=21)
        self.exp_pressure = 1000 * np.exp(-self.altitude_grid / 10)
        const_pressure = np.linspace(1000, 1000, num=21)
        const_temper = np.linspace(200, 200, num=21)
        self.same_boundaries = self.altitude_grid
        self.mass = 7.3 * 10 ** -26
        self.g = 3.71
        self.constant_hydro = Hydrostatic(self.altitude_grid, const_pressure, const_temper, self.same_boundaries, self.mass, self.g)
        self.same_hydro = Hydrostatic(self.altitude_grid, self.exp_pressure,
                                      const_temper,
                                      self.same_boundaries, self.mass, self.g)

    def test_scale_height_is_read_only(self):
        with pytest.raises(AttributeError):
            self.same_hydro.scale_height = 0

    def test_scale_height_has_same_shape_as_altitude_boundaries(self) -> None:
        assert self.constant_hydro.altitude.shape == \
               self.constant_hydro.scale_height.shape

