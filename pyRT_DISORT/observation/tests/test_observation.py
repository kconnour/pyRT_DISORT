# Built-in imports
from unittest import TestCase

# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.observation.observation import Observation, Wavelengths


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
        self.assertTrue(hasattr(test_observation, 'solar_zenith_angle'))
        self.assertTrue(hasattr(test_observation, 'emission_angle'))
        self.assertTrue(hasattr(test_observation, 'phase_angle'))
        self.assertTrue(hasattr(test_observation, 'mu0'))
        self.assertTrue(hasattr(test_observation, 'mu'))
        self.assertTrue(hasattr(test_observation, 'phi0'))
        self.assertTrue(hasattr(test_observation, 'phi'))
        self.assertTrue(len(test_observation.__dict__.keys()), 11)

    def test_observation_inputs_cannot_be_ints(self):
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


class TestWavelength(TestCase):
    def setUp(self):
        self.wavelengths = Wavelengths


class TestWavelengthInit(TestWavelength):
    def test_wavelength_has_4_known_attributes(self):
        short_wavelengths = np.linspace(1, 10, num=5)
        long_wavelengths = short_wavelengths + 1
        test_wavelengths = self.wavelengths(short_wavelengths, long_wavelengths)
        self.assertTrue(hasattr(test_wavelengths, 'short_wavelength'))
        self.assertTrue(hasattr(test_wavelengths, 'long_wavelength'))
        self.assertTrue(hasattr(test_wavelengths, 'low_wavenumber'))
        self.assertTrue(hasattr(test_wavelengths, 'high_wavenumber'))
        self.assertEqual(4, len(test_wavelengths.__dict__.keys()))

    def test_wavelength_inputs_cannot_be_ints(self):
        self.assertRaises(TypeError, lambda: self.wavelengths(1, 2))

    def test_wavelength_inputs_cannot_be_floats(self):
        self.assertRaises(TypeError, lambda: self.wavelengths(1.5, 2.5))

    def test_input_wavelengths_must_have_same_shape(self):
        short_wavelengths = np.linspace(1, 10, num=5)
        long_wavelengths = np.linspace(10, 15, num=4)
        self.assertRaises(ValueError, lambda: self.wavelengths(short_wavelengths, long_wavelengths))

    def test_short_wavelength_must_be_positive(self):
        long_wavelength = np.array([1])
        smallest_negative_float = np.array([np.nextafter(0, -1)])
        self.assertRaises(ValueError, lambda: self.wavelengths(smallest_negative_float, long_wavelength))
        self.assertRaises(ValueError, lambda: self.wavelengths(np.array([0]), long_wavelength))

    def test_long_wavelength_cannot_be_less_than_short_wavelength(self):
        short_wavelengths = np.linspace(1, 10, num=10)
        long_wavelengths = short_wavelengths + 0.5
        long_wavelengths[0] = 0.5
        self.assertRaises(ValueError, lambda: self.wavelengths(short_wavelengths, long_wavelengths))

    def test_wavelengths_cannot_be_too_small(self):
        long_wavelength = np.array([1])
        smallest_positive_float = np.array([np.nextafter(0, 1)])
        self.assertRaises(ValueError, lambda: self.wavelengths(smallest_positive_float, long_wavelength))

    def test_wavelengths_must_be_finite(self):
        infinite_wavelength = np.array([np.inf])
        test_wavelength = np.array([1])
        self.assertRaises(ValueError, lambda: self.wavelengths(test_wavelength, infinite_wavelength))
        self.assertRaises(ValueError, lambda: self.wavelengths(infinite_wavelength, test_wavelength))

    def test_long_wavelength_cannot_equal_short_wavelength(self):
        short_wavelengths = np.linspace(1, 10, num=10)
        long_wavelengths = short_wavelengths + 0.5
        long_wavelengths[0] = 1
        self.assertRaises(ValueError, lambda: self.wavelengths(short_wavelengths, long_wavelengths))

    def test_short_wavelength_is_converted_to_high_wavenumber(self):
        long_wavelength = np.array([1000])
        self.assertEqual(np.array([10 ** 4]), self.wavelengths(np.array([1]), long_wavelength).high_wavenumber)
        self.assertEqual(np.array([10 ** 2]), self.wavelengths(np.array([100]), long_wavelength).high_wavenumber)

    def test_long_wavelength_is_converted_to_low_wavenumber(self):
        short_wavelength = np.array([0.1])
        self.assertEqual(np.array([10 ** 4]), self.wavelengths(short_wavelength, np.array([1])).low_wavenumber)
        self.assertEqual(np.array([10 ** 2]), self.wavelengths(short_wavelength ,np.array([100])).low_wavenumber)
