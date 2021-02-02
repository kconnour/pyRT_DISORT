from unittest import TestCase
import numpy as np
from pyRT_DISORT.boundary_conditions import BoundaryConditions


class TestInit(TestCase):
    def test_int_thermal_emission_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            BoundaryConditions(thermal_emission=0)

    def test_float_thermal_emission_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            BoundaryConditions(thermal_emission=1.0)

    def test_int_bottom_temperature_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            BoundaryConditions(bottom_temperature=200)

    def test_negative_bottom_temperature_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BoundaryConditions(bottom_temperature=np.nextafter(0, -1))

    def test_infinite_bottom_temperature_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BoundaryConditions(bottom_temperature=np.inf)

    def test_int_top_temperature_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            BoundaryConditions(top_temperature=200)

    def test_negative_top_temperature_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BoundaryConditions(top_temperature=np.nextafter(0, -1))

    def test_infinite_top_temperature_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BoundaryConditions(top_temperature=np.inf)

    def test_int_top_emissivity_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            BoundaryConditions(top_emissivity=0)

        with self.assertRaises(TypeError):
            BoundaryConditions(top_emissivity=1)

    def test_top_emissivity_outside_0_to_1_raises_value_error(self) -> None:
        BoundaryConditions(top_emissivity=0.0)
        with self.assertRaises(ValueError):
            BoundaryConditions(top_emissivity=np.nextafter(0, -1))

        BoundaryConditions(top_emissivity=1.0)
        with self.assertRaises(ValueError):
            BoundaryConditions(top_emissivity=np.nextafter(1, 2))

    def test_int_beam_flux_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            BoundaryConditions(beam_flux=3)

    def test_negative_beam_flux_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BoundaryConditions(beam_flux=np.nextafter(0, -1))

    def test_infinite_beam_flux_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BoundaryConditions(beam_flux=np.inf)

    def test_int_isotropic_flux_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            BoundaryConditions(isotropic_flux=10)

    def test_negative_isotropic_flux_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BoundaryConditions(isotropic_flux=np.nextafter(0, -1))

    def test_infinite_isotropic_flux_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BoundaryConditions(isotropic_flux=np.inf)

    def test_int_lambertian_bottom_boundary_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            BoundaryConditions(lambertian_bottom_boundary=0)

    def test_float_lambertian_bottom_boundary_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            BoundaryConditions(lambertian_bottom_boundary=1.0)

    def test_int_albedo_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            BoundaryConditions(albedo=0)

        with self.assertRaises(TypeError):
            BoundaryConditions(albedo=1)

    def test_albedo_outside_0_to_1_raises_value_error(self) -> None:
        BoundaryConditions(albedo=0.0)
        with self.assertRaises(ValueError):
            BoundaryConditions(albedo=np.nextafter(0, -1))

        BoundaryConditions(albedo=1.0)
        with self.assertRaises(ValueError):
            BoundaryConditions(albedo=np.nextafter(1, 2))


class TestAlbedo(TestCase):
    def test_albedo_is_unchanged(self) -> None:
        test_albedo = 0.5
        bc = BoundaryConditions(albedo=test_albedo)
        self.assertEqual(test_albedo, bc.albedo)

    def test_albedo_defaults_to_0(self) -> None:
        bc = BoundaryConditions()
        self.assertEqual(0, bc.albedo)


class TestBeamFlux(TestCase):
    def test_beam_flux_is_unchanged(self) -> None:
        test_beam_flux = 2.7564
        bc = BoundaryConditions(beam_flux=test_beam_flux)
        self.assertEqual(test_beam_flux, bc.beam_flux)

    def test_beam_flux_defaults_to_pi(self) -> None:
        bc = BoundaryConditions()
        self.assertEqual(np.pi, bc.beam_flux)


class TestBottomTemperature(TestCase):
    def test_bottom_temperature_is_unchanged(self) -> None:
        test_bottom_temperature = 200.0
        bc = BoundaryConditions(bottom_temperature=test_bottom_temperature)
        self.assertEqual(test_bottom_temperature, bc.bottom_temperature)

    def test_bottom_temperature_defaults_to_0(self) -> None:
        bc = BoundaryConditions()
        self.assertEqual(0, bc.bottom_temperature)


class TestIsotropicFlux(TestCase):
    def test_isotropic_flux_is_unchanged(self) -> None:
        test_isotropic_flux = 10.0
        bc = BoundaryConditions(isotropic_flux=test_isotropic_flux)
        self.assertEqual(test_isotropic_flux, bc.isotropic_flux)

    def test_isotropic_flux_defaults_to_0(self) -> None:
        bc = BoundaryConditions()
        self.assertEqual(0, bc.isotropic_flux)


class TestLambertianBottomBoundary(TestCase):
    def test_lambertian_bottom_boundary_is_unchanged(self) -> None:
        lambertian = False
        bc = BoundaryConditions(lambertian_bottom_boundary=lambertian)
        self.assertEqual(lambertian, bc.lambertian_bottom_boundary)

    def test_lambertian_bottom_boundary_defaults_to_true(self) -> None:
        bc = BoundaryConditions()
        self.assertEqual(True, bc.lambertian_bottom_boundary)


class TestThermalEmission(TestCase):
    def test_thermal_emission_is_unchanged(self) -> None:
        thermal_emission = True
        bc = BoundaryConditions(thermal_emission=thermal_emission)
        self.assertEqual(thermal_emission, bc.thermal_emission)

    def test_top_emissivity_defaults_to_false(self) -> None:
        bc = BoundaryConditions()
        self.assertEqual(False, bc.thermal_emission)


class TestTopEmissivity(TestCase):
    def test_top_emissivity_is_unchanged(self) -> None:
        test_top_emissivity = 0.5
        bc = BoundaryConditions(top_emissivity=test_top_emissivity)
        self.assertEqual(test_top_emissivity, bc.top_emissivity)

    def test_top_emissivity_defaults_to_1(self) -> None:
        bc = BoundaryConditions()
        self.assertEqual(1, bc.top_emissivity)


class TestTopTemperature(TestCase):
    def test_top_temperature_is_unchanged(self) -> None:
        test_top_temperature = 200.0
        bc = BoundaryConditions(top_temperature=test_top_temperature)
        self.assertEqual(test_top_temperature, bc.top_temperature)

    def test_top_temperature_defaults_to_0(self) -> None:
        bc = BoundaryConditions()
        self.assertEqual(0, bc.top_temperature)
