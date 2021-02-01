from unittest import TestCase
import numpy as np
import numpy.testing as npt
from pyRT_DISORT.eos import ModelEquationOfState


class TestModelEquationOfState(TestCase):
    def setUp(self) -> None:
        self.eos = TestModelEquationOfState


class TestInit(TestModelEquationOfState):
    def test_2d_altitude_grid_raises_value_error(self):
        with self.assertRaises(ValueError):
            a = np.ones((10, 5))
            junk = np.linspace(1, 10)
            ModelEquationOfState(a, junk, junk, junk, junk)

    def test_mono_increasing_altitude_model_raises_value_error(self):
        a = np.linspace(1, 50)
        ModelEquationOfState(a, a, a, a, np.flip(a))
        with self.assertRaises(ValueError):
            ModelEquationOfState(a, a, a, a, a)

    def test_list_altitude_grid_raises_type_error(self):
        altitude_grid = [0, 10, 20]
        junk = np.linspace(20, 10)
        with self.assertRaises(TypeError):
            ModelEquationOfState(altitude_grid, junk, junk, junk, junk)

    def test_index_error_raised_if_grids_do_not_have_same_shape(self):
        a = np.linspace(10, 20)
        with self.assertRaises(IndexError):
            ModelEquationOfState(a, a, a[:-1], a, np.flip(a))
