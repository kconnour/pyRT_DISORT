from unittest import TestCase
import numpy as np
from pyRT_DISORT.controller import ComputationalParameters


class TestComputationalParameters(TestCase):
    def setUp(self) -> None:
        self.cp = ComputationalParameters(10, 30, 20, 40, 50, 60)


class TestComputationalParametersInit(TestComputationalParameters):
    def test_str_n_layers_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            ComputationalParameters('foo', 30, 20, 40, 50, 60)

    def test_str_n_moments_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            ComputationalParameters(10, 'foo', 30, 40, 50, 60)

    def test_n_moments_equal_to_n_streams_is_ok(self) -> None:
        ComputationalParameters(10, 30, 30, 40, 50, 60)

    def test_str_n_streams_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            ComputationalParameters(10, 30, 'foo', 40, 50, 60)

    def test_odd_n_streams_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            ComputationalParameters(10, 30, 25, 40, 50, 60)

    def test_str_n_umu_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            ComputationalParameters(10, 30, 20, 'foo', 50, 60)

    def test_str_n_phi_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            ComputationalParameters(10, 30, 20, 40, 'foo', 60)

    def test_str_n_user_levels_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            ComputationalParameters(10, 30, 20, 40, 50, 'foo')


class TestNLayers(TestComputationalParameters):
    def test_n_layers_is_unchanged(self) -> None:
        self.assertEqual(10, self.cp.n_layers)

    def test_n_layers_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.cp.n_layers = 100


class TestNMoments(TestComputationalParameters):
    def test_n_moments_is_unchanged(self) -> None:
        self.assertEqual(30, self.cp.n_moments)

    def test_n_moments_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.cp.n_moments = 100


class TestNStreams(TestComputationalParameters):
    def test_n_streams_is_unchanged(self) -> None:
        self.assertEqual(20, self.cp.n_streams)

    def test_n_streams_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.cp.n_streams = 100


class TestNUmu(TestComputationalParameters):
    def test_n_umu_is_unchanged(self) -> None:
        self.assertEqual(40, self.cp.n_umu)

    def test_n_umu_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.cp.n_umu = 100


class TestNPhi(TestComputationalParameters):
    def test_n_phi_is_unchanged(self) -> None:
        self.assertEqual(50, self.cp.n_phi)

    def test_n_phi_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.cp.n_phi = 100


class TestNUserLevels(TestComputationalParameters):
    def test_n_user_levels_is_unchanged(self) -> None:
        self.assertEqual(60, self.cp.n_user_levels)

    def test_n_user_levels_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.cp.n_user_levels = 100
