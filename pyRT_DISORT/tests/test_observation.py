from unittest import TestCase
import numpy as np
import numpy.testing as npt
from pyRT_DISORT.observation import Angles, Wavelengths


class TestAngles(TestCase):
    def setUp(self) -> None:
        self.dummy_angles = np.linspace(1, 50, num=2)
        self.angles = Angles(self.dummy_angles, self.dummy_angles,
                             self.dummy_angles)
        self.str_angles = np.linspace(1, 50, num=2, dtype=str)
        self.zero = np.array([0])
        self.one = np.array([1])
        self.forty_five = np.array([45])
        self.sixty = np.array([60])
        self.ninety = np.array([90])
        self.one_eighty = np.array([180])
        self.small_neg = np.array([np.nextafter(0, -1)])
        self.test_2d_array = np.ones((10, 5))


class TestAnglesInit(TestAngles):
    def test_float_incidence_angles_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            Angles(1, self.one, self.one)

    def test_2d_incidence_angles_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            Angles(self.test_2d_array, self.one, self.one)

    def test_incidence_angles_outside_0_to_180_raises_value_error(self) -> None:
        Angles(self.zero, self.one, self.one)
        Angles(self.one_eighty, self.one, self.one)
        with self.assertRaises(ValueError):
            Angles(self.small_neg, self.one, self.one)
        with self.assertRaises(ValueError):
            test_angle = np.array([np.nextafter(181, 181)])
            Angles(test_angle, self.one, self.one)

    def test_oddly_shaped_incidence_angles_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            Angles(self.one, self.dummy_angles, self.dummy_angles)

    def test_array_of_strs_incidence_angles_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            Angles(self.str_angles, self.dummy_angles, self.dummy_angles)

    def test_float_emission_angles_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            Angles(self.one, 1, self.one)

    def test_2d_emission_angles_raises_type_error(self) -> None:
        with self.assertRaises(ValueError):
            Angles(self.one, self.test_2d_array, self.one)

    def test_emission_angles_outside_0_to_90_raises_value_error(self) -> None:
        Angles(self.one, self.zero, self.one)
        Angles(self.one, self.ninety, self.one)
        with self.assertRaises(ValueError):
            Angles(self.one, self.small_neg, self.one)
        with self.assertRaises(ValueError):
            test_angle = np.array([np.nextafter(90, 91)])
            Angles(self.one, test_angle, self.one)

    def test_oddly_shaped_emission_angles_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            Angles(self.dummy_angles, self.one, self.dummy_angles)

    def test_array_of_strs_emission_angles_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            Angles(self.dummy_angles, self.str_angles, self.dummy_angles)

    def test_float_phase_angles_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            Angles(self.one, self.one, 1)

    def test_2d_phase_angles_raises_type_error(self) -> None:
        with self.assertRaises(ValueError):
            Angles(self.one, self.one, self.test_2d_array)

    def test_phase_angles_outside_0_to_180_raises_value_error(self) -> None:
        Angles(self.one, self.one, self.zero)
        Angles(self.one, self.one, self.one_eighty)
        with self.assertRaises(ValueError):
            Angles(self.one, self.one, self.small_neg)
        with self.assertRaises(ValueError):
            test_angle = np.array([np.nextafter(180, 181)])
            Angles(self.one, self.one, test_angle)

    def test_oddly_shaped_phase_angles_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            Angles(self.dummy_angles, self.dummy_angles, self.one)

    def test_array_of_strs_phase_angles_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            Angles(self.dummy_angles, self.dummy_angles, self.str_angles)


class TestIncidence(TestAngles):
    def test_incidence_angles_is_unchanged(self) -> None:
        incidence_angle = self.dummy_angles + 10
        a = Angles(incidence_angle, self.dummy_angles, self.dummy_angles)
        self.assertTrue(np.array_equal(incidence_angle, a.incidence))

    def test_incidence_angles_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.angles.incidence = self.dummy_angles


class TestEmission(TestAngles):
    def test_emission_angles_is_unchanged(self) -> None:
        emission_angle = self.dummy_angles + 10
        a = Angles(self.dummy_angles, emission_angle, self.dummy_angles)
        self.assertTrue(np.array_equal(emission_angle, a.emission))

    def test_emission_angles_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.angles.emission = self.dummy_angles


class TestPhase(TestAngles):
    def test_phase_angles_is_unchanged(self) -> None:
        phase_angle = self.dummy_angles + 10
        a = Angles(self.dummy_angles, self.dummy_angles, phase_angle)
        self.assertTrue(np.array_equal(phase_angle, a.phase))

    def test_phase_angles_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.angles.phase = self.dummy_angles


class TestMu0(TestAngles):
    def test_mu0_passes_test_cases(self):
        a = Angles(self.zero, self.one, self.one)
        self.assertEqual(1, a.mu0[0])

        a = Angles(self.forty_five, self.one, self.one)
        self.assertEqual(np.sqrt(2) / 2, a.mu0[0])

        a = Angles(self.sixty, self.one, self.one)
        self.assertAlmostEqual(0.5, a.mu0[0])

        a = Angles(self.ninety, self.one, self.one)
        self.assertAlmostEqual(0, a.mu0[0])

        a = Angles(self.one_eighty, self.one, self.one)
        self.assertEqual(-1, a.mu0[0])

    def test_mu0_is_read_only(self):
        with npt.assert_raises(AttributeError):
            self.angles.mu0 = self.dummy_angles

    def test_mu0_is_same_shape_as_incidence_angles(self):
        self.assertEqual(self.angles.incidence.shape, self.angles.mu0.shape)


class TestMu(TestAngles):
    def test_mu_passes_test_cases(self):
        a = Angles(self.one, self.zero, self.one)
        self.assertEqual(1, a.mu[0])

        a = Angles(self.one, self.forty_five, self.one)
        self.assertEqual(np.sqrt(2) / 2, a.mu[0])

        a = Angles(self.one, self.sixty, self.one)
        self.assertAlmostEqual(0.5, a.mu[0])

        a = Angles(self.one, self.ninety, self.one)
        self.assertAlmostEqual(0, a.mu[0])

    def test_mu_is_read_only(self):
        with npt.assert_raises(AttributeError):
            self.angles.mu = self.dummy_angles

    def test_mu_is_same_shape_as_emission_angles(self):
        self.assertEqual(self.angles.emission.shape, self.angles.mu.shape)


class TestPhi0(TestAngles):
    def test_phi0_is_always_0(self):
        self.assertTrue(np.all(self.angles.phi0 == 0))

    def test_phi0_is_read_only(self):
        with npt.assert_raises(AttributeError):
            self.angles.phi0 = self.dummy_angles

    def test_phi0_is_same_shape_as_phase_angles(self):
        self.assertEqual(self.angles.phase.shape, self.angles.phi0.shape)


class TestPhi(TestAngles):
    def test_phi_matches_disort_multi_computations(self):
        a = Angles(self.zero, self.zero, self.zero)
        self.assertEqual(0, a.phi[0])

        a = Angles(np.array([10]), np.array([10]), np.array([10]))
        self.assertAlmostEqual(119.747139, a.phi[0], places=4)

        a = Angles(np.array([70]), np.array([70]), np.array([70]))
        self.assertAlmostEqual(104.764977, a.phi[0], places=4)

    def test_phi_is_read_only(self):
        with npt.assert_raises(AttributeError):
            self.angles.phi = self.dummy_angles

    def test_phi_is_same_shape_as_phase_angles(self):
        self.assertEqual(self.angles.phase.shape, self.angles.phi.shape)


class TestWavelengths(TestCase):
    def setUp(self) -> None:
        self.wavelengths = Wavelengths


class TestInit(TestWavelengths):
    def test_index_error_raised_if_different_input_sizes(self):
        short = np.linspace(10, 20)
        long = short + 1
        short = short[:-1]
        with self.assertRaises(IndexError):
            Wavelengths(short, long)

    def test_float_input_raises_type_error(self):
        with self.assertRaises(TypeError):
            Wavelengths(1, np.array([3, 4]))

    def test_negative_input_raises_value_error(self):
        short = np.linspace(np.nextafter(0, -1), 20)
        with self.assertRaises(ValueError):
            Wavelengths(short, short + 1)

    def test_infinite_input_raises_value_error(self):
        long = np.linspace(10, 20)
        short = long - 1
        long[-1] = np.inf
        with self.assertRaises(ValueError):
            Wavelengths(short, long)

    def test_equal_input_raises_value_error(self):
        wavelengths = np.linspace(10, 20)
        with self.assertRaises(ValueError):
            Wavelengths(wavelengths, wavelengths)


class TestShortWavelengths(TestWavelengths):
    def test_short_wavelength_is_read_only(self):
        short = np.linspace(10, 20)
        w = Wavelengths(short, short + 1)
        with self.assertRaises(AttributeError):
            w.short_wavelengths = short

    def test_short_wavelength_is_unmodified(self):
        short = np.linspace(10, 20)
        w = Wavelengths(short, short + 1)
        self.assertTrue(np.array_equal(short, w.short_wavelengths))


class TestLongWavelengths(TestWavelengths):
    def test_long_wavelength_is_read_only(self):
        long = np.linspace(10, 20)
        w = Wavelengths(long - 1, long)
        with self.assertRaises(AttributeError):
            w.long_wavelengths = long

    def test_long_wavelength_is_unmodified(self):
        long = np.linspace(10, 20)
        w = Wavelengths(long - 1, long)
        self.assertTrue(np.array_equal(long, w.long_wavelengths))


class TestHighWavenumbers(TestWavelengths):
    def test_high_wavenumbers_match_known_values(self):
        short = np.array([10, 11])
        w = Wavelengths(short, short + 1)
        expected = np.array([1000, 909.090909])
        npt.assert_almost_equal(expected, w.high_wavenumber)


class TestLowWavenumbers(TestWavelengths):
    def test_low_wavenumbers_match_known_values(self):
        long = np.array([10, 11])
        w = Wavelengths(long - 1, long)
        expected = np.array([1000, 909.090909])
        npt.assert_almost_equal(expected, w.low_wavenumber)
