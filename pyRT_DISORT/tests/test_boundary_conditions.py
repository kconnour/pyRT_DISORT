from unittest import TestCase
import numpy as np
from pyRT_DISORT.boundary_conditions import BoundaryConditions


class TestInit(TestCase):
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

    def test_int_beam_flux_raises_type_error(self):
        with self.assertRaises(TypeError):
            BoundaryConditions(beam_flux=3)

    def test_negative_beam_flux_raises_value_error(self):
        with self.assertRaises(ValueError):
            BoundaryConditions(beam_flux=np.nextafter(0, -1))

    def test_infinite_beam_flux_raises_value_error(self):
        with self.assertRaises(ValueError):
            BoundaryConditions(beam_flux=np.inf)


class TestAlbedo(TestCase):
    def test_albedo_is_unchanged(self) -> None:
        test_albedo = 0.5
        bc = BoundaryConditions(albedo=test_albedo)
        self.assertEqual(test_albedo, bc.albedo)

    def test_albedo_defaults_to_0(self):
        bc = BoundaryConditions()
        self.assertEqual(0, bc.albedo)


class TestBeamFlux(TestCase):
    def test_beam_flux_is_unchanged(self) -> None:
        test_beam_flux = 2.7564
        bc = BoundaryConditions(beam_flux=test_beam_flux)
        self.assertEqual(test_beam_flux, bc.beam_flux)

    def test_albedo_defaults_to_0(self):
        bc = BoundaryConditions()
        self.assertEqual(np.pi, bc.beam_flux)
