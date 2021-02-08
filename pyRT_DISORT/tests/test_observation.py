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
        self.odd = np.array([1, 2, 3])
        self.zero = np.array([0])
        self.one = np.array([1])
        self.forty_five = np.array([45])
        self.sixty = np.array([60])
        self.ninety = np.array([90])
        self.one_eighty = np.array([180])
        self.small_neg = np.array([np.nextafter(0, -1)])
        self.test_2d_array = np.ones((10, 5))


class TestAnglesInit(TestAngles):
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
            Angles(self.odd, self.dummy_angles, self.dummy_angles)

    def test_array_of_strs_incidence_angles_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            Angles(self.str_angles, self.dummy_angles, self.dummy_angles)

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
            Angles(self.dummy_angles, self.odd, self.dummy_angles)

    def test_array_of_strs_emission_angles_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            Angles(self.dummy_angles, self.str_angles, self.dummy_angles)

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
            Angles(self.dummy_angles, self.dummy_angles, self.odd)

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
        self.dummy_wavelengths = np.linspace(10, 20, num=50)
        self.single_wavelength = np.array([10])
        self.wavelengths_2d = np.broadcast_to(self.dummy_wavelengths, (10, 50))
        self.test_wavelengths = Wavelengths(
            self.dummy_wavelengths, self.dummy_wavelengths + 1)
        self.known_wavelengths = np.array([10, 11])
        self.known_wavenumbers = np.array([1000, 909.090909])


class TestWavelengthsInit(TestWavelengths):
    def test_float_short_wavelengths_raises_type_error(self):
        with self.assertRaises(TypeError):
            Wavelengths(10, self.single_wavelength)

    def test_float_long_wavelengths_raises_type_error(self):
        with self.assertRaises(TypeError):
            Wavelengths(self.single_wavelength, 10)

    def test_negative_short_wavelength_raises_value_error(self):
        short = np.copy(self.dummy_wavelengths)
        short[0] = np.nextafter(0, -1)
        with self.assertRaises(ValueError):
            Wavelengths(short, short + 1)

    def test_infinite_long_wavelength_raises_value_error(self):
        short = np.copy(self.dummy_wavelengths)
        long = short + 1
        long[-1] = np.inf
        with self.assertRaises(ValueError):
            Wavelengths(short, long)

    def test_shorter_long_wavelength_raises_value_error(self):
        short = np.copy(self.dummy_wavelengths)
        long = short - 1
        with self.assertRaises(ValueError):
            Wavelengths(short, long)

    def test_equal_input_raises_value_error(self):
        with self.assertRaises(ValueError):
            Wavelengths(self.dummy_wavelengths, self.dummy_wavelengths)

    def test_2d_short_wavelengths_raises_value_error(self):
        with self.assertRaises(ValueError):
            Wavelengths(self.wavelengths_2d, self.dummy_wavelengths)

    def test_2d_long_wavelengths_raises_value_error(self):
        with self.assertRaises(ValueError):
            Wavelengths(self.dummy_wavelengths, self.wavelengths_2d)

    def test_index_error_raised_if_different_input_sizes(self):
        short = np.copy(self.dummy_wavelengths)
        long = short[:-1]
        with self.assertRaises(IndexError):
            Wavelengths(short, long)


class TestShortWavelengths(TestWavelengths):
    def test_short_wavelength_is_unmodified(self):
        self.assertTrue(np.array_equal(
            self.dummy_wavelengths, self.test_wavelengths.short_wavelengths))

    def test_short_wavelength_is_read_only(self):
        with self.assertRaises(AttributeError):
            self.test_wavelengths.short_wavelengths = self.dummy_wavelengths


class TestLongWavelengths(TestWavelengths):
    def test_short_wavelength_is_unmodified(self):
        self.assertTrue(np.array_equal(
            self.dummy_wavelengths + 1, self.test_wavelengths.long_wavelengths))

    def test_short_wavelength_is_read_only(self):
        with self.assertRaises(AttributeError):
            self.test_wavelengths.long_wavelengths = self.dummy_wavelengths


class TestHighWavenumbers(TestWavelengths):
    def test_high_wavenumbers_match_known_values(self):
        w = Wavelengths(self.known_wavelengths, self.known_wavelengths + 1)
        npt.assert_almost_equal(self.known_wavenumbers, w.high_wavenumber)

    def test_high_wavenumbers_is_read_only(self):
        with self.assertRaises(AttributeError):
            self.test_wavelengths.high_wavenumber = self.dummy_wavelengths


class TestLowWavenumbers(TestWavelengths):
    def test_low_wavenumbers_match_known_values(self):
        w = Wavelengths(self.known_wavelengths - 1, self.known_wavelengths)
        npt.assert_almost_equal(self.known_wavenumbers, w.low_wavenumber)

    def test_low_wavenumbers_is_read_only(self):
        with self.assertRaises(AttributeError):
            self.test_wavelengths.low_wavenumber = self.dummy_wavelengths
