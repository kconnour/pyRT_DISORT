# Built-in imports
from unittest import TestCase

# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.observation.observation import Observation


class TestObservation(TestCase):
    def setUp(self):
        self.observation = Observation
        self.example_short_wavelength = np.array([1])
        self.example_long_wavelength = np.array([10])
        self.example_solar_zenith_angle = np.array([20])
        self.example_emission_angle = np.array([30])
        self.example_phase_angle = np.array([40])
        self.smallest_negative_float = np.array([np.nextafter(0, -1)])
        self.smallest_float_above_180 = np.array([np.nextafter(180, 181)])
        self.smallest_float_above_90 = np.array([np.nextafter(90, 91)])


class TestObservationInit(TestObservation):
    def test_observation_has_11_known_attributes(self):
        test_observation = self.observation(self.example_short_wavelength, self.example_long_wavelength,
                                            self.example_solar_zenith_angle, self.example_emission_angle,
                                            self.example_phase_angle)
        self.assertTrue(hasattr(test_observation, 'short_wavelength'))
        self.assertTrue(hasattr(test_observation, 'long_wavelength'))
        self.assertTrue(hasattr(test_observation, 'solar_zenith_angle'))
        self.assertTrue(hasattr(test_observation, 'emission_angle'))
        self.assertTrue(hasattr(test_observation, 'phase_angle'))
        self.assertTrue(hasattr(test_observation, 'low_wavenumber'))
        self.assertTrue(hasattr(test_observation, 'high_wavenumber'))
        self.assertTrue(hasattr(test_observation, 'mu0'))
        self.assertTrue(hasattr(test_observation, 'mu'))
        self.assertTrue(hasattr(test_observation, 'phi0'))
        self.assertTrue(hasattr(test_observation, 'phi'))
        self.assertTrue(len(test_observation.__dict__.keys()), 11)

    def test_observation_inputs_cannot_be_ints(self):
        self.assertRaises(TypeError, lambda: self.observation(1, self.example_long_wavelength,
                                                              self.example_solar_zenith_angle,
                                                              self.example_emission_angle, self.example_phase_angle))
        self.assertRaises(TypeError, lambda: self.observation(self.example_short_wavelength, 1,
                                                              self.example_solar_zenith_angle,
                                                              self.example_emission_angle, self.example_phase_angle))
        self.assertRaises(TypeError, lambda: self.observation(self.example_short_wavelength,
                                                              self.example_long_wavelength, 1,
                                                              self.example_emission_angle, self.example_phase_angle))
        self.assertRaises(TypeError, lambda: self.observation(self.example_short_wavelength,
                                                              self.example_long_wavelength,
                                                              self.example_solar_zenith_angle, 1,
                                                              self.example_phase_angle))
        self.assertRaises(TypeError, lambda: self.observation(self.example_short_wavelength,
                                                              self.example_long_wavelength,
                                                              self.example_solar_zenith_angle,
                                                              self.example_emission_angle, 1))

    def test_long_wavelength_cannot_be_shorter_than_short_wavelength(self):
        next_shortest_wavelength = np.nextafter(self.example_short_wavelength, 0)
        self.assertRaises(ValueError, lambda: self.observation(self.example_short_wavelength, next_shortest_wavelength,
                                                               self.example_solar_zenith_angle,
                                                               self.example_emission_angle, self.example_phase_angle))

    def test_long_wavelength_cannot_equal_short_wavelength(self):
        self.assertRaises(ValueError, lambda: self.observation(self.example_short_wavelength,
                                                               self.example_short_wavelength,
                                                               self.example_solar_zenith_angle,
                                                               self.example_emission_angle,
                                                               self.example_phase_angle))

    def test_short_wavelength_must_be_positive(self):
        self.assertRaises(ValueError, lambda: self.observation(self.smallest_negative_float,
                                                               self.example_long_wavelength,
                                                               self.example_solar_zenith_angle,
                                                               self.example_emission_angle, self.example_phase_angle))

    def test_wavelengths_must_be_finite(self):
        inf = np.array([np.inf])
        self.assertRaises(ValueError, lambda: self.observation(self.example_short_wavelength, inf,
                                                               self.example_solar_zenith_angle,
                                                               self.example_emission_angle, self.example_phase_angle))

    def test_solar_zenith_angle_must_be_between_0_and_180(self):
        # At 0 and 180, I should be able to make a class without error
        self.observation(self.example_short_wavelength, self.example_long_wavelength, np.array([0]),
                         self.example_emission_angle, self.example_phase_angle)
        self.observation(self.example_short_wavelength, self.example_long_wavelength, np.array([180]),
                         self.example_emission_angle, self.example_phase_angle)
        # At the float just outside [0, 180], I shouldn't be able to make an object
        self.assertRaises(ValueError, lambda: self.observation(self.example_short_wavelength,
                                                               self.example_long_wavelength,
                                                               self.smallest_negative_float,
                                                               self.example_emission_angle, self.example_phase_angle))
        self.assertRaises(ValueError, lambda: self.observation(self.example_short_wavelength,
                                                               self.example_long_wavelength,
                                                               self.smallest_float_above_180,
                                                               self.example_emission_angle, self.example_phase_angle))

    def test_emission_angle_must_be_between_0_and_90(self):
        self.observation(self.example_short_wavelength, self.example_long_wavelength, self.example_solar_zenith_angle,
                         np.array([0]), self.example_phase_angle)
        self.observation(self.example_short_wavelength, self.example_long_wavelength, self.example_solar_zenith_angle,
                         np.array([90]), self.example_phase_angle)

        self.assertRaises(ValueError,
                          lambda: self.observation(self.example_short_wavelength, self.example_long_wavelength,
                                                   self.example_solar_zenith_angle, self.smallest_negative_float,
                                                   self.example_phase_angle))
        self.assertRaises(ValueError,
                          lambda: self.observation(self.example_short_wavelength, self.example_long_wavelength,
                                                   self.example_solar_zenith_angle, self.smallest_float_above_90,
                                                   self.example_phase_angle))

    def test_phase_angle_must_be_between_0_and_180(self):
        self.observation(self.example_short_wavelength, self.example_long_wavelength, self.example_solar_zenith_angle,
                         self.example_emission_angle, np.array([0]))
        self.observation(self.example_short_wavelength, self.example_long_wavelength, self.example_solar_zenith_angle,
                         self.example_emission_angle, np.array([180]))
        self.assertRaises(ValueError,
                          lambda: self.observation(self.example_short_wavelength, self.example_long_wavelength,
                                                   self.example_solar_zenith_angle, self.example_emission_angle,
                                                   self.smallest_negative_float))
        self.assertRaises(ValueError,
                          lambda: self.observation(self.example_short_wavelength, self.example_long_wavelength,
                                                   self.example_solar_zenith_angle, self.example_emission_angle,
                                                   self.smallest_float_above_180))

    def test_input_wavelengths_must_have_same_shape(self):
        self.assertRaises(ValueError,
                          lambda: self.observation(self.example_short_wavelength, np.array([10, 11]),
                                                   self.example_solar_zenith_angle, self.example_emission_angle,
                                                   self.example_phase_angle))
        self.assertRaises(ValueError,
                          lambda: self.observation(np.array([1, 2]), self.example_long_wavelength,
                                                   self.example_solar_zenith_angle, self.example_emission_angle,
                                                   self.example_phase_angle))

    def test_input_angles_have_same_shape(self):
        self.assertRaises(ValueError,
                          lambda: self.observation(self.example_short_wavelength, self.example_long_wavelength,
                                                   self.example_solar_zenith_angle, self.example_emission_angle,
                                                   np.array([1, 2])))
        self.assertRaises(ValueError,
                          lambda: self.observation(self.example_short_wavelength, self.example_long_wavelength,
                                                   self.example_solar_zenith_angle, np.array([1, 2]),
                                                   self.example_phase_angle))
        self.assertRaises(ValueError,
                          lambda: self.observation(self.example_short_wavelength, self.example_long_wavelength,
                                                   np.array([1, 2]), self.example_emission_angle,
                                                   self.example_phase_angle))

    def test_short_wavelength_is_converted_to_high_wavenumber(self):
        example_long_wavelength = np.array([1000])
        self.assertEqual(10 ** 4, self.observation(np.array([1]), example_long_wavelength,
                                                   self.example_solar_zenith_angle, self.example_emission_angle,
                                                   self.example_phase_angle).high_wavenumber)
        self.assertEqual(10 ** 2, self.observation(np.array([100]), example_long_wavelength,
                                                   self.example_solar_zenith_angle, self.example_emission_angle,
                                                   self.example_phase_angle).high_wavenumber)

    def test_long_wavelength_is_converted_to_low_wavenumber(self):
        example_short_wavelength = np.array([0.1])
        self.assertEqual(10 ** 4, self.observation(example_short_wavelength, np.array([1]),
                                                   self.example_solar_zenith_angle, self.example_emission_angle,
                                                   self.example_phase_angle).low_wavenumber)
        self.assertEqual(10 ** 2, self.observation(example_short_wavelength, np.array([100]),
                                                   self.example_solar_zenith_angle, self.example_emission_angle,
                                                   self.example_phase_angle).low_wavenumber)

    def test_mu0_is_same_shape_as_solar_zenith_angles(self):
        random_int = np.random.randint(100, 200 + 1)
        example_solar_zenith_angles = np.linspace(10, 25, num=random_int)
        example_emission_angles = np.linspace(10, 20, num=random_int)
        example_phase_angles = np.linspace(10, 30, num=random_int)
        self.assertEqual(example_solar_zenith_angles.shape, self.observation(self.example_short_wavelength,
                                                                             self.example_long_wavelength,
                                                                             example_solar_zenith_angles,
                                                                             example_emission_angles,
                                                                             example_phase_angles).mu0.shape)

    def test_mu0_is_cosine_of_solar_zenith_angles(self):
        random_int = np.random.randint(100, 200 + 1)
        example_solar_zenith_angles = np.linspace(10, 25, num=random_int)
        example_emission_angles = np.linspace(10, 20, num=random_int)
        example_phase_angles = np.linspace(10, 30, num=random_int)
        truth = np.array_equal(np.cos(np.radians(example_solar_zenith_angles)),
                               self.observation(self.example_short_wavelength, self.example_long_wavelength,
                                                example_solar_zenith_angles, example_emission_angles,
                                                example_phase_angles).mu0)
        self.assertTrue(truth)

    def test_mu_is_same_shape_as_emission_angles(self):
        random_int = np.random.randint(100, 200 + 1)
        example_solar_zenith_angles = np.linspace(10, 25, num=random_int)
        example_emission_angles = np.linspace(10, 20, num=random_int)
        example_phase_angles = np.linspace(10, 30, num=random_int)
        self.assertEqual(example_emission_angles.shape,
                         self.observation(self.example_short_wavelength, self.example_long_wavelength,
                                          example_solar_zenith_angles, example_emission_angles,
                                          example_phase_angles).mu.shape)

    def test_mu_is_cosine_of_emission_angles(self):
        random_int = np.random.randint(100, 200 + 1)
        example_solar_zenith_angles = np.linspace(10, 25, num=random_int)
        example_emission_angles = np.linspace(10, 20, num=random_int)
        example_phase_angles = np.linspace(10, 30, num=random_int)
        truth = np.array_equal(np.cos(np.radians(example_emission_angles)),
                               self.observation(self.example_short_wavelength, self.example_long_wavelength,
                                                example_solar_zenith_angles, example_emission_angles,
                                                example_phase_angles).mu)

        self.assertTrue(truth)

    def test_phi0_is_always_0(self):
        self.assertEqual(0, self.observation(self.example_short_wavelength, self.example_long_wavelength,
                                             self.example_solar_zenith_angle, self.example_emission_angle,
                                             self.example_phase_angle).phi0)

    def test_phi_is_same_shape_as_input_angles(self):
        random_int = np.random.randint(100, 200 + 1)
        example_solar_zenith_angles = np.linspace(10, 25, num=random_int)
        example_emission_angles = np.linspace(10, 20, num=random_int)
        example_phase_angles = np.linspace(10, 30, num=random_int)
        self.assertTrue(
            example_solar_zenith_angles.shape == example_emission_angles.shape == example_phase_angles.shape ==
            self.observation(
                self.example_short_wavelength, self.example_long_wavelength, example_solar_zenith_angles,
                example_emission_angles, example_phase_angles).phi.shape)

    # There's an unknown issue here... if I input
    #example_solar_zenith_angles = np.linspace(1, 179, num=random_int)
    #example_emission_angles = np.linspace(10, 20, num=random_int)
    #example_phase_angles = np.linspace(10, 30, num=random_int)
    # to self.observation I get an error
