# Built-in imports
from unittest import TestCase

# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.observation.observation import Wavelengths

class TestWavelength(TestCase):
    def setUp(self):
        self.wavelengths = Wavelengths


class TestWavelengthInit(TestWavelength):
    def test_wavelength_has_4_known_attributes(self):
        short_wavelengths = np.linspace(1, 10, num=5)
        long_wavelengths = short_wavelengths + 1
        test_wavelengths = self.wavelengths(short_wavelengths, long_wavelengths)
        self.assertTrue(hasattr(test_wavelengths, 'short_wavelengths'))
        self.assertTrue(hasattr(test_wavelengths, 'long_wavelengths'))
        self.assertTrue(hasattr(test_wavelengths, 'low_wavenumbers'))
        self.assertTrue(hasattr(test_wavelengths, 'high_wavenumbers'))
        self.assertEqual(4, len(test_wavelengths.__dict__.keys()))

    def test_wavelength_inputs_cannot_be_ints(self):
        self.assertRaises(TypeError, lambda: self.wavelengths(1, 2))

    def test_wavelength_inputs_cannot_be_floats(self):
        self.assertRaises(TypeError, lambda: self.wavelengths(1.5, 2.5))

    def test_wavelengths_must_be_1d(self):
        test_1d_wavelengths = np.linspace(1, 10, num=10)
        test_2d_wavelengths = np.multiply.outer(test_1d_wavelengths, test_1d_wavelengths) + 0.5
        self.assertRaises(IndexError, lambda: self.wavelengths(test_1d_wavelengths, test_2d_wavelengths))
        test_2d_wavelengths -= 1
        self.assertRaises(IndexError, lambda: self.wavelengths(test_2d_wavelengths, test_1d_wavelengths))

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
        self.assertEqual(np.array([10 ** 4]), self.wavelengths(np.array([1]), long_wavelength).high_wavenumbers)
        self.assertEqual(np.array([10 ** 2]), self.wavelengths(np.array([100]), long_wavelength).high_wavenumbers)

    def test_long_wavelength_is_converted_to_low_wavenumber(self):
        short_wavelength = np.array([0.1])
        self.assertEqual(np.array([10 ** 4]), self.wavelengths(short_wavelength, np.array([1])).low_wavenumbers)
        self.assertEqual(np.array([10 ** 2]), self.wavelengths(short_wavelength ,np.array([100])).low_wavenumbers)
