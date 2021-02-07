from unittest import TestCase
import numpy as np
from pyRT_DISORT.surface import Surface, Lambertian


class TestSurface(TestCase):
    def setUp(self) -> None:
        self.surface = Surface(0.5)


class TestSurfaceInit(TestSurface):
    def test_int_input_is_ok(self) -> None:
        Surface(1)

    def test_ndarray_raises_value_error(self) -> None:
        a = np.zeros(10)
        with self.assertRaises(ValueError):
            Surface(a)


class TestAlbedo(TestSurface):
    def test_albedo_is_unchanged(self) -> None:
        self.assertEqual(0.5, self.surface.albedo)

    def test_albedo_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.surface.albedo = 0


class TestLambertian(TestCase):
    def setUp(self) -> None:
        self.lamber = Lambertian(0.5)


class TestLambertianInit(TestLambertian):
    pass


class TestLambertianProperty(TestLambertian):
    def test_lambertian_is_always_true(self) -> None:
        self.assertTrue(self.lamber.lambertian)

    def test_lambertian_is_immutable(self) -> None:
        with self.assertRaises(AttributeError):
            self.lamber.lambertian = False
