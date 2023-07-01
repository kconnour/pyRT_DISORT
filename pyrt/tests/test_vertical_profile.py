import numpy as np

from pyrt.vertical_profile import conrath


class TestConrath:
    def test_profile_is_same_shape_as_altitude(self):
        altitude = np.linspace(100, 0, num=101)

        profile = conrath(altitude, 1, 10, 0.1)

        assert profile.shape == altitude.shape
        