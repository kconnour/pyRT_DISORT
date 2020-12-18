# Built-in imports
from unittest import TestCase

# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.observation.observation import Observation


class TestObservation(TestCase):
    def setUp(self):
        self.observation = Observation


class TestObservationInit(TestObservation):
    def test_observation_has_11_attributes(self):
        test_short_wavelength = np.array([1])
        test_long_wavelength = np.array([2])
        test_angle = np.array([10])
        test_observation = self.observation(test_short_wavelength, test_long_wavelength,
                                            test_angle, test_angle, test_angle)
        self.assertEqual(11, len(test_observation.__dict__.keys()))
