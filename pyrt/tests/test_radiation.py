from unittest import TestCase
import numpy as np
from pyRT_DISORT.radiation import IncidentFlux, ThermalEmission


class TestIncidentFlux(TestCase):
    def setUp(self) -> None:
        self.array_fluxes = np.array([1, 2])


class TestIncidentFluxInit(TestIncidentFlux):
    def test_int_beam_flux_raises_no_error(self) -> None:
        IncidentFlux(beam_flux=3)

    def test_str_beam_flux_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            IncidentFlux(beam_flux='foo')

    def test_ndarray_beam_flux_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            IncidentFlux(beam_flux=self.array_fluxes)

    def test_int_isotropic_flux_raises_no_error(self) -> None:
        IncidentFlux(isotropic_flux=3)

    def test_str_isotropic_flux_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            IncidentFlux(isotropic_flux='foo')

    def test_ndarray_isotropic_flux_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            IncidentFlux(isotropic_flux=self.array_fluxes)


class TestBeamFlux(TestIncidentFlux):
    def test_beam_flux_is_unchanged(self) -> None:
        incidence = IncidentFlux(beam_flux=2.7564)
        self.assertEqual(2.7564, incidence.beam_flux)

    def test_beam_flux_defaults_to_pi(self) -> None:
        self.assertEqual(np.pi, IncidentFlux().beam_flux)

    def test_beam_flux_is_read_only(self) -> None:
        incidence = IncidentFlux()
        with self.assertRaises(AttributeError):
            incidence.beam_flux = 0


class TestIsotropicFlux(TestIncidentFlux):
    def test_isotropic_flux_is_unchanged(self) -> None:
        incidence = IncidentFlux(isotropic_flux=10.0)
        self.assertEqual(10.0, incidence.isotropic_flux)

    def test_isotropic_flux_defaults_to_0(self) -> None:
        self.assertEqual(0, IncidentFlux().isotropic_flux)

    def test_isotropic_flux_is_read_only(self) -> None:
        incidence = IncidentFlux()
        with self.assertRaises(AttributeError):
            incidence.isotropic_flux = 0


class TestThermalEmission(TestCase):
    def setUp(self) -> None:
        pass


class TestThermalEmissionInit(TestThermalEmission):
    def test_list_thermal_emission_returns_true(self) -> None:
        te = ThermalEmission(thermal_emission=[100])
        self.assertTrue(te.thermal_emission)

    def test_str_bottom_temperature_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            ThermalEmission(bottom_temperature='foo')

    def test_list_bottom_temperature_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            ThermalEmission(bottom_temperature=[100])

    def test_infinite_bottom_temperature_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            ThermalEmission(bottom_temperature=np.inf)

    def test_nan_bottom_temperature_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            ThermalEmission(bottom_temperature=np.nan)

    def test_str_top_temperature_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            ThermalEmission(top_temperature='foo')

    def test_list_top_temperature_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            ThermalEmission(top_temperature=[100])

    def test_infinite_top_temperature_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            ThermalEmission(top_temperature=np.inf)

    def test_nan_top_temperature_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            ThermalEmission(bottom_temperature=np.nan)

    def test_top_emissivity_outside_0_to_1_raises_value_error(self) -> None:
        ThermalEmission(top_emissivity=0.0)
        with self.assertRaises(ValueError):
            ThermalEmission(top_emissivity=np.nextafter(0, -1))

        ThermalEmission(top_emissivity=1.0)
        with self.assertRaises(ValueError):
            ThermalEmission(top_emissivity=np.nextafter(1, 2))

        with self.assertRaises(ValueError):
            ThermalEmission(top_emissivity=np.inf)

        with self.assertRaises(ValueError):
            ThermalEmission(top_emissivity=np.nan)


class TestThermalEmissionProperty(TestThermalEmission):
    def test_thermal_emission_is_unchanged(self) -> None:
        te = ThermalEmission(thermal_emission=True)
        self.assertEqual(True, te.thermal_emission)

    def test_top_emissivity_defaults_to_false(self) -> None:
        self.assertEqual(False, ThermalEmission().thermal_emission)

    def test_thermal_emission_is_read_only(self) -> None:
        te = ThermalEmission()
        with self.assertRaises(AttributeError):
            te.thermal_emission = 0


class TestBottomTemperature(TestThermalEmission):
    def test_bottom_temperature_is_unchanged(self) -> None:
        te = ThermalEmission(bottom_temperature=200.0)
        self.assertEqual(200.0, te.bottom_temperature)

    def test_bottom_temperature_defaults_to_0(self) -> None:
        self.assertEqual(0, ThermalEmission().bottom_temperature)

    def test_bottom_temperature_is_read_only(self) -> None:
        te = ThermalEmission()
        with self.assertRaises(AttributeError):
            te.bottom_temperature = 0


class TestTopTemperature(TestThermalEmission):
    def test_top_temperature_is_unchanged(self) -> None:
        te = ThermalEmission(top_temperature=200.0)
        self.assertEqual(200.0, te.top_temperature)

    def test_top_temperature_defaults_to_0(self) -> None:
        self.assertEqual(0, ThermalEmission().top_temperature)

    def test_top_temperature_is_read_only(self) -> None:
        te = ThermalEmission()
        with self.assertRaises(AttributeError):
            te.top_temperature = 0


class TestTopEmissivity(TestThermalEmission):
    def test_top_emissivity_is_unchanged(self) -> None:
        te = ThermalEmission(top_emissivity=0.5)
        self.assertEqual(0.5, te.top_emissivity)

    def test_top_emissivity_defaults_to_1(self) -> None:
        self.assertEqual(1, ThermalEmission().top_emissivity)

    def test_top_emissivity_is_read_only(self) -> None:
        te = ThermalEmission()
        with self.assertRaises(AttributeError):
            te.top_emissivity = 0
