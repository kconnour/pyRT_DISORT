import numpy as np
import pytest
from scipy.constants import Boltzmann
from pyRT_DISORT.eos import ModelEquationOfState, Hydrostatic


class TestHydrostatic:
    def setup(self) -> None:
        self.pressure = np.broadcast_to(np.linspace(1000, 100, num=20), (5, 20))
        self.temper = np.broadcast_to(np.linspace(250, 100, num=20), (5, 20))
        self.hydro = Hydrostatic(self.pressure, self.temper)


class TestHydrostaticInit(TestHydrostatic):
    def test_hydrostatic_contains_3_attributes(self) -> None:
        assert len(self.hydro.__dict__.items()) == 3


class TestPressure(TestHydrostatic):
    def test_pressure_is_unchanged(self) -> None:
        assert np.array_equal(self.hydro.pressure, self.pressure)

    def test_pressure_is_read_only(self) -> None:
        with pytest.raises(AttributeError):
            self.hydro.pressure = self.pressure

    def test_non_positive_pressure_raises_value_error(self) -> None:
        pressure = np.copy(self.pressure)
        pressure[0] = np.nextafter(0, 1)
        Hydrostatic(pressure, self.temper)

        pressure[0] = 0
        with pytest.raises(ValueError):
            Hydrostatic(pressure, self.temper)

        pressure[0] = np.nextafter(0, -1)
        with pytest.raises(ValueError):
            Hydrostatic(pressure, self.temper)

    def test_pressure_array_of_strs_raises_type_error(self) -> None:
        pressure = np.copy(self.pressure).astype('str')
        with pytest.raises(TypeError):
            Hydrostatic(pressure, self.temper)


class TestTemperature(TestHydrostatic):
    def test_temperature_is_unchanged(self) -> None:
        assert np.array_equal(self.hydro.temperature, self.temper)

    def test_temperature_is_read_only(self) -> None:
        with pytest.raises(AttributeError):
            self.hydro.temperature = self.temper

    def test_non_positive_temperature_raises_value_error(self) -> None:
        temperature = np.copy(self.temper)
        temperature[0] = np.nextafter(0, 1)
        Hydrostatic(temperature, self.temper)

        temperature[0] = 0
        with pytest.raises(ValueError):
            Hydrostatic(temperature, self.temper)

        temperature[0] = np.nextafter(0, -1)
        with pytest.raises(ValueError):
            Hydrostatic(temperature, self.temper)

    def test_temperature_array_of_strs_raises_type_error(self) -> None:
        temperature = np.copy(self.temper).astype('str')
        with pytest.raises(TypeError):
            Hydrostatic(self.pressure, temperature)


class TestNumberDensity(TestHydrostatic):
    def test_number_density_computations_match_known_values(self) -> None:
        temperature = np.ones(50)
        pressure = np.linspace(1, 50, num=50)
        hydro = Hydrostatic(pressure, temperature)
        assert np.array_equal(hydro.number_density, pressure / Boltzmann)

    def test_number_density_is_read_only(self) -> None:
        with pytest.raises(AttributeError):
            self.hydro.number_density = 0




'''class TestModelEquationOfState(TestCase):
    def setUp(self) -> None:
        self.eos = ModelEquationOfState


class TestInit(TestModelEquationOfState):
    def test_2d_altitude_grid_raises_value_error(self):
        with self.assertRaises(ValueError):
            a = np.ones((10, 5))
            junk = np.linspace(1, 10)
            ModelEquationOfState(a, junk, junk, junk, junk)

    def test_mono_increasing_altitude_model_raises_value_error(self):
        a = np.linspace(1, 50)
        ModelEquationOfState(a, a, a, a, np.flip(a))
        with self.assertRaises(ValueError):
            ModelEquationOfState(a, a, a, a, a)

    def test_list_altitude_grid_raises_type_error(self):
        altitude_grid = [0, 10, 20]
        junk = np.linspace(20, 10)
        with self.assertRaises(TypeError):
            ModelEquationOfState(altitude_grid, junk, junk, junk, junk)

    def test_index_error_raised_if_grids_do_not_have_same_shape(self):
        a = np.linspace(10, 20)
        with self.assertRaises(IndexError):
            ModelEquationOfState(a, a, a[:-1], a, np.flip(a))


class TestAltitudeBoundaries(TestModelEquationOfState):
    def test_altitude_boundaries_is_unchanged(self):
        junk = np.linspace(50, 10)
        ab = junk - 5
        eos = ModelEquationOfState(junk, junk + 1, junk + 2, junk + 3, ab)
        self.assertTrue(np.array_equal(ab, eos.altitude_boundaries))


class TestNLayers(TestModelEquationOfState):
    def test_n_layers_is_defined_by_altitude_model(self):
        junk = np.linspace(50, 10, num=10)
        ab = np.linspace(60, 20, num=30)
        eos = ModelEquationOfState(junk, junk, junk, junk, ab)
        self.assertEqual(29, eos.n_layers)


class TestNumberDensityBoundaries(TestModelEquationOfState):
    pass


class TestPressureBoundaries(TestModelEquationOfState):
    def test_pressure_is_linearly_interpolated(self):
        alts = np.array([10, 20, 30])
        pressure = np.array([50, 30, 10])
        temperatures = np.array([200, 170, 140])
        # for some reason it gives me an error if I use realistic values
        num_den = np.array([100, 50, 25])
        malts = np.array([25, 15])
        eos = ModelEquationOfState(alts, pressure, temperatures, num_den, malts)
        expected_pressure = np.array([20, 40])
        self.assertTrue(np.array_equal(expected_pressure,
                                       eos.pressure_boundaries))


class TestTemperatureBoundaries(TestModelEquationOfState):
    def test_temperature_is_linearly_interpolated(self):
        alts = np.array([10, 20, 30])
        pressure = np.array([50, 30, 10])
        temperatures = np.array([200, 170, 140])
        # for some reason it gives me an error if I use realistic values
        num_den = np.array([100, 50, 25])
        malts = np.array([25, 15])
        eos = ModelEquationOfState(alts, pressure, temperatures, num_den, malts)
        expected_temperatures = np.array([155, 185])
        self.assertTrue(np.array_equal(expected_temperatures,
                                       eos.temperature_boundaries))'''

