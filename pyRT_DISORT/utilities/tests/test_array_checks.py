from unittest import TestCase
import numpy as np
import numpy.testing as npt
from pyRT_DISORT.utilities.array_checks import ArrayChecker


class TestArrayChecker(TestCase):
    def setUp(self) -> None:
        self.ac = ArrayChecker


class TestInit(TestArrayChecker):
    def test_int_input_raises_type_error(self):
        with self.assertRaises(TypeError):
            ArrayChecker(1)

    def test_float_input_raises_type_error(self):
        with self.assertRaises(TypeError):
            ArrayChecker(2.5)

    def test_list_input_raises_type_error(self):
        with self.assertRaises(TypeError):
            ArrayChecker([1, 2.5])
