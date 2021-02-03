from unittest import TestCase
import numpy as np
from pyRT_DISORT.utilities.array_checks import ArrayChecker


class TestArrayChecker(TestCase):
    def setUp(self) -> None:
        self.zeros = np.zeros(10)
        self.int_zeros = ArrayChecker(np.zeros(10, dtype=int))
        self.float_zeros = ArrayChecker(np.zeros(10, dtype=float))
        self.mono_dec = ArrayChecker(np.linspace(50.0, 1.0))
        self.mono_inc = ArrayChecker(np.linspace(1.0, 50.0))
        self.zeros_2d = ArrayChecker(np.zeros((10, 10)))
        self.zeros_3d = ArrayChecker(np.zeros((10, 10, 10)))
        zeros = np.copy(self.zeros)
        zeros[-1] = np.inf
        self.one_inf = ArrayChecker(zeros)
        zeros = np.copy(self.zeros)
        zeros[0] = np.nextafter(0, -1)
        self.small_neg = ArrayChecker(zeros)


class TestInit(TestArrayChecker):
    def test_int_input_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            ArrayChecker(1)

    def test_float_input_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            ArrayChecker(2.5)

    def test_list_input_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            ArrayChecker([1, 2.5])

    def test_input_array_of_strs_raises_value_error(self) -> None:
        array = np.zeros(10, dtype=str)
        with self.assertRaises(ValueError):
            ArrayChecker(array)


class TestDetermineIfArrayContainsOnlyInts(TestArrayChecker):
    def test_array_of_ints_returns_true(self) -> None:
        self.assertTrue(self.int_zeros.determine_if_array_contains_only_ints())

    def test_array_of_floats_returns_false(self) -> None:
        self.assertFalse(
            self.float_zeros.determine_if_array_contains_only_ints())


class TestDetermineIfArrayIs1d(TestArrayChecker):
    def test_1d_array_returns_true(self) -> None:
        self.assertTrue(self.int_zeros.determine_if_array_is_1d())

    def test_2d_array_returns_false(self) -> None:
        self.assertFalse(self.zeros_2d.determine_if_array_is_1d())


class TestDetermineIfArrayIs2d(TestArrayChecker):
    def test_2d_array_returns_true(self) -> None:
        self.assertTrue(self.zeros_2d.determine_if_array_is_2d())

    def test_1d_array_returns_false(self) -> None:
        self.assertFalse(self.int_zeros.determine_if_array_is_2d())

    def test_3d_array_returns_false(self) -> None:
        self.assertFalse(self.zeros_3d.determine_if_array_is_2d())


class TestDetermineIfArrayIsFinite(TestArrayChecker):
    def test_array_with_all_finite_values_returns_true(self) -> None:
        self.assertTrue(self.int_zeros.determine_if_array_is_finite())

    def test_array_with_all_infinite_values_returns_false(self) -> None:
        ac = ArrayChecker(self.zeros + np.inf)
        self.assertFalse(ac.determine_if_array_is_finite())

    def test_array_with_one_infinite_values_returns_false(self) -> None:
        self.assertFalse(self.one_inf.determine_if_array_is_finite())


class TestDetermineIfArrayIsInRange(TestArrayChecker):
    def test_array_in_known_range_returns_true(self) -> None:
        self.assertTrue(self.mono_inc.determine_if_array_is_in_range(1.0, 50.0))

    def test_array_of_0_is_in_range_0(self) -> None:
        self.assertTrue(self.int_zeros.determine_if_array_is_in_range(0.0, 0.0))

    def test_int_input_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            self.int_zeros.determine_if_array_is_in_range(0.0, 0)


class TestDetermineIfArrayIsMonotonicallyDecreasing(TestArrayChecker):
    def test_monotonically_decreasing_array_returns_true(self) -> None:
        self.assertTrue(
            self.mono_dec.determine_if_array_is_monotonically_decreasing())

    def test_monotonically_increasing_array_returns_false(self) -> None:
        self.assertFalse(
            self.mono_inc.determine_if_array_is_monotonically_decreasing())

    def test_array_of_same_value_returns_false(self) -> None:
        self.assertFalse(
            self.int_zeros.determine_if_array_is_monotonically_decreasing())


class TestDetermineIfArrayIsMonotonicallyIncreasing(TestArrayChecker):
    def test_monotonically_increasing_array_returns_true(self) -> None:
        self.assertTrue(
            self.mono_inc.determine_if_array_is_monotonically_increasing())

    def test_monotonically_decreasing_array_returns_false(self) -> None:
        self.assertFalse(
            self.mono_dec.determine_if_array_is_monotonically_increasing())

    def test_array_of_same_value_returns_false(self) -> None:
        self.assertFalse(
            self.int_zeros.determine_if_array_is_monotonically_increasing())


class TestDetermineIfArrayIsNonNegative(TestArrayChecker):
    def test_array_of_zeros_returns_true(self):
        self.assertTrue(self.int_zeros.determine_if_array_is_non_negative())

    def test_array_with_one_negative_value_returns_false(self):
        self.assertFalse(self.small_neg.determine_if_array_is_non_negative())

    def test_array_with_positive_infinity_returns_true(self):
        self.assertTrue(self.one_inf.determine_if_array_is_non_negative())


class TestDetermineIfArrayIsPositive(TestArrayChecker):
    def test_array_with_all_positive_values_returns_true(self):
        self.assertTrue(self.mono_inc.determine_if_array_is_positive())

    def test_array_with_zero_returns_false(self):
        self.assertFalse(self.int_zeros.determine_if_array_is_positive())

    def test_array_with_one_negative_value_returns_false(self):
        self.assertFalse(self.small_neg.determine_if_array_is_non_negative())

    def test_array_with_positive_infinity_returns_true(self):
        self.assertTrue(self.one_inf.determine_if_array_is_non_negative())


class TestDetermineIfArrayIsPositiveFinite(TestArrayChecker):
    def test_array_of_positive_finite_values_returns_true(self):
        self.assertTrue(self.mono_inc.determine_if_array_is_positive_finite())

    def test_array_with_positive_infinity_returns_false(self):
        array = np.linspace(1.0, 50.0)
        array[-1] = np.inf
        ac = ArrayChecker(array)
        self.assertFalse(ac.determine_if_array_is_positive_finite())

    def test_array_wth_zeros_returns_false(self):
        self.assertFalse(self.int_zeros.determine_if_array_is_positive_finite())


class TestDetermineIfArrayMatchesDimensionality(TestArrayChecker):
    def test_1d_array_with_dimensionality_of_1_returns_true(self):
        self.assertTrue(
            self.int_zeros.determine_if_array_matches_dimensionality(1))

    def test_1d_array_with_dimensionality_of_2_returns_false(self):
        self.assertFalse(
            self.int_zeros.determine_if_array_matches_dimensionality(2))

    def test_2d_array_with_dimensionality_of_1_returns_false(self):
        self.assertFalse(
            self.zeros_2d.determine_if_array_matches_dimensionality(1))

    def test_2d_array_with_dimensionality_of_2_returns_true(self):
        self.assertTrue(
            self.zeros_2d.determine_if_array_matches_dimensionality(2))

    def test_3d_array_with_dimensionality_of_3_returns_true(self):
        self.assertTrue(
            self.zeros_3d.determine_if_array_matches_dimensionality(3))
