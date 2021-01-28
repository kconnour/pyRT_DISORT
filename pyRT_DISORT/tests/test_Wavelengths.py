from pyRT_DISORT.observation import Wavelengths
from unittest import TestCase
import numpy as np
import numpy.testing as npt


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
