from unittest import TestCase
import numpy as np
from pyRT_DISORT.controller import ComputationalParameters
from pyRT_DISORT.surface import Surface, Lambertian


class TestSurface(TestCase):
    def setUp(self) -> None:
        self.cp = ComputationalParameters(10, 30, 20, 40, 50, 60)
        self.surface = Surface(0.5, self.cp)


class TestSurfaceInit(TestSurface):
    def test_int_input_is_ok(self) -> None:
        Surface(1, self.cp)

    def test_ndarray_raises_type_error(self) -> None:
        a = np.zeros(10)
        with self.assertRaises(TypeError):
            Surface(a, self.cp)

    def test_unphysical_albedo_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            Surface(np.nextafter(0, -1), self.cp)

        with self.assertRaises(ValueError):
            Surface(np.nextafter(1, 2), self.cp)

    def test_int_cp_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            Surface(1, 1)


class TestAlbedo(TestSurface):
    def test_albedo_is_unchanged(self) -> None:
        self.assertEqual(0.5, self.surface.albedo)

    def test_albedo_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.surface.albedo = 0


class TestLambertian(TestCase):
    def setUp(self) -> None:
        self.cp = ComputationalParameters(10, 30, 20, 40, 50, 60)
        self.lamber = Lambertian(0.5, self.cp)


class TestLambertianInit(TestLambertian):
    pass


class TestLambertianProperty(TestLambertian):
    def test_lambertian_is_always_true(self) -> None:
        self.assertTrue(self.lamber.lambertian)

    def test_lambertian_is_immutable(self) -> None:
        with self.assertRaises(AttributeError):
            self.lamber.lambertian = False


class TestBemst(TestLambertian):
    def test_bemst_matches_known_output(self) -> None:
        answer = np.zeros(10)
        self.assertTrue(np.array_equal(answer, self.lamber.bemst))

    def test_bemst_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.lamber.bemst = 0


class TestEmust(TestLambertian):
    def test_emust_matches_known_output(self) -> None:
        answer = np.zeros(50)
        self.assertTrue(np.array_equal(answer, self.lamber.emust))

    def test_emust_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.lamber.emust = 0


class TestRhoAccurate(TestLambertian):
    def test_rho_accurate_matches_known_output(self) -> None:
        answer = np.zeros((50, 40))
        self.assertTrue(np.array_equal(answer, self.lamber.rho_accurate))

    def test_rho_accurate_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.lamber.rho_accurate = 0


class TestRhoq(TestLambertian):
    def test_rhoq_matches_known_output(self) -> None:
        answer = np.zeros((20, 11, 20))
        self.assertTrue(np.array_equal(answer, self.lamber.rhoq))

    def test_rhoq_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.lamber.rhoq = 0


class TestRhou(TestLambertian):
    def test_rhou_matches_known_output(self) -> None:
        answer = np.zeros((10, 11, 20))
        self.assertTrue(np.array_equal(answer, self.lamber.rhou))

    def test_rhou_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.lamber.rhou = 0


# TODO: I don't know how to test the Hapke classes since I don't know what
#  values the arrays are supposed to contain.
