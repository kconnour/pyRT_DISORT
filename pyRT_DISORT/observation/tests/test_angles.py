# Built-in imports
from unittest import TestCase

# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.observation.observation import Angles


class TestAngles(TestCase):
    def setUp(self):
        self.angles = Angles
        self.random_int = np.random.randint(100, 200 + 1)


class TestAnglesInit(TestAngles):
    def test_angles_has_7_known_attributes(self):
        example_angles = np.linspace(10, 50)
        test_angles = self.angles(example_angles, example_angles, example_angles)
        self.assertTrue(hasattr(test_angles, 'solar_zenith_angles'))
        self.assertTrue(hasattr(test_angles, 'emission_angles'))
        self.assertTrue(hasattr(test_angles, 'phase_angles'))
        self.assertTrue(hasattr(test_angles, 'mu'))
        self.assertTrue(hasattr(test_angles, 'mu0'))
        self.assertTrue(hasattr(test_angles, 'phi'))
        self.assertTrue(hasattr(test_angles, 'phi0'))
        self.assertEqual(7, len(test_angles.__dict__.keys()))

    def test_input_angles_cannot_be_ints(self):
        test_angles = np.array([1])
        self.assertRaises(TypeError, lambda: self.angles(1, test_angles, test_angles))
        self.assertRaises(TypeError, lambda: self.angles(test_angles, 1, test_angles))
        self.assertRaises(TypeError, lambda: self.angles(test_angles, test_angles, 1))

    def test_input_angles_cannot_be_floats(self):
        test_angles = np.array([1.5])
        self.assertRaises(TypeError, lambda: self.angles(1.5, test_angles, test_angles))
        self.assertRaises(TypeError, lambda: self.angles(test_angles, 1.5, test_angles))
        self.assertRaises(TypeError, lambda: self.angles(test_angles, test_angles, 1.5))

    def test_input_angles_must_have_same_shapes(self):
        more_angles = np.linspace(10, 20, num=50)
        fewer_angles = np.linspace(10, 20, num=25)
        self.assertRaises(ValueError, lambda: self.angles(more_angles, fewer_angles, fewer_angles))
        self.assertRaises(ValueError, lambda: self.angles(fewer_angles, more_angles, fewer_angles))
        self.assertRaises(ValueError, lambda: self.angles(fewer_angles, fewer_angles, more_angles))
        self.angles(more_angles, more_angles, more_angles)

    def test_input_angles_must_be_1d(self):
        multi_dim_angles = np.ones((10, 10))
        single_dim_angles = np.ones(10)
        self.assertRaises(IndexError, lambda: self.angles(multi_dim_angles, single_dim_angles, single_dim_angles))
        self.assertRaises(IndexError, lambda: self.angles(single_dim_angles, multi_dim_angles, single_dim_angles))
        self.assertRaises(IndexError, lambda: self.angles(single_dim_angles, single_dim_angles, multi_dim_angles))
        self.angles(single_dim_angles, single_dim_angles, single_dim_angles)

    def test_solar_zenith_angles_must_be_between_0_and_180(self):
        test_angle = np.array([50])
        self.angles(np.array([0]), test_angle, test_angle)
        self.angles(np.array([180]), test_angle, test_angle)
        self.assertRaises(ValueError, lambda: self.angles(np.nextafter(np.array([0]), -1), test_angle, test_angle))
        self.assertRaises(ValueError, lambda: self.angles(np.nextafter(np.array([180]), 181), test_angle, test_angle))

    def test_emission_angles_must_be_between_0_and_90(self):
        test_angle = np.array([50])
        self.angles(test_angle, np.array([0]), test_angle)
        self.angles(test_angle, np.array([90]), test_angle)
        self.assertRaises(ValueError, lambda: self.angles(test_angle, np.nextafter(np.array([0]), -1), test_angle))
        self.assertRaises(ValueError, lambda: self.angles(test_angle, np.nextafter(np.array([90]), 91), test_angle))

    def test_phase_angles_must_be_between_0_and_180(self):
        test_angle = np.array([50])
        self.angles(test_angle, test_angle, np.array([0]))
        self.angles(test_angle, test_angle, np.array([180]))
        self.assertRaises(ValueError, lambda: self.angles(test_angle, test_angle, np.nextafter(np.array([0]), -1)))
        self.assertRaises(ValueError, lambda: self.angles(test_angle, test_angle, np.nextafter(np.array([180]), 181)))

    def test_mu_is_same_shape_as_emission_angles(self):
        test_emission_angles = np.linspace(10, 80, num=self.random_int)
        test_other_angles = np.linspace(20, 70, num=self.random_int)
        angles = self.angles(test_other_angles, test_emission_angles, test_other_angles)
        self.assertEqual(test_emission_angles.shape, angles.mu.shape)

    def test_mu_is_cosine_of_emission_angles(self):
        test_angles = np.array([50])
        self.assertEqual(1, self.angles(test_angles, np.array([0]), test_angles).mu[0])
        self.assertEqual(np.sqrt(2) / 2, self.angles(test_angles, np.array([45]), test_angles).mu[0])
        # I guess this is a numpy issue... cos(90) = 10**-17 and not 0
        self.assertAlmostEqual(0, self.angles(test_angles, np.array([90]), test_angles).mu[0])

    def test_mu0_is_same_shape_as_solar_zenith_angles(self):
        test_solar_zenith_angles = np.linspace(10, 80, num=self.random_int)
        test_other_angles = np.linspace(20, 70, num=self.random_int)
        angles = self.angles(test_solar_zenith_angles, test_other_angles, test_other_angles)
        self.assertEqual(test_solar_zenith_angles.shape, angles.mu0.shape)

    def test_mu0_is_cosine_of_solar_zenith_angles(self):
        test_angles = np.array([50])
        self.assertEqual(1, self.angles(np.array([0]), test_angles, test_angles).mu0[0])
        self.assertEqual(np.sqrt(2) / 2, self.angles(np.array([45]), test_angles, test_angles).mu0[0])
        self.assertAlmostEqual(0, self.angles(np.array([90]), test_angles, test_angles).mu0[0])
        self.assertAlmostEqual(-np.sqrt(2) / 2, self.angles(np.array([135]), test_angles, test_angles).mu0[0])
        self.assertEqual(-1, self.angles(np.array([180]), test_angles, test_angles).mu0[0])

    def test_phi_is_same_shape_as_phase_angles(self):
        test_phase_angles = np.linspace(10, 80, num=self.random_int)
        test_other_angles = np.linspace(20, 70, num=self.random_int)
        angles = self.angles(test_other_angles, test_other_angles, test_phase_angles)
        self.assertEqual(test_phase_angles.shape, angles.phi.shape)

    def test_phi_meets_test_cases(self):
        self.assertEqual(0, self.angles(np.array([0]), np.array([0]), np.array([0])).phi[0])
        self.assertEqual(119.74712028120182, self.angles(np.array([10]), np.array([10]), np.array([10])).phi[0])
        self.assertEqual(2.091309795559937e-06, self.angles(np.array([10]), np.array([20]), np.array([30])).phi[0])

    def test_phi0_is_same_shape_as_phase_angles(self):
        test_phase_angles = np.linspace(10, 80, num=self.random_int)
        test_other_angles = np.linspace(20, 70, num=self.random_int)
        angles = self.angles(test_other_angles, test_other_angles, test_phase_angles)
        self.assertEqual(test_phase_angles.shape, angles.phi0.shape)

    def test_phi0_is_always_0(self):
        test_phase_angles = np.linspace(10, 80, num=self.random_int)
        test_other_angles = np.linspace(20, 70, num=self.random_int)
        angles = self.angles(test_other_angles, test_other_angles, test_phase_angles)
        self.assertTrue(np.all(angles.phi0 == 0))
