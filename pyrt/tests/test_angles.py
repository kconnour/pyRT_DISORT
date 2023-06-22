import numpy as np
import pytest

from pyrt.angles import azimuth


class TestAzimuth:
    def test_trivial_case_produces_0(self):
        incidence = 0
        emission = 0
        phase = 0

        az = azimuth(incidence, emission, phase)

        assert az == 0

    def test_max_phase_angle_produces_0(self):
        # Mike says this should be true for any case where
        #  incidence + emission = phase
        incidence = 10
        emission = 20
        phase = 30

        az = azimuth(incidence, emission, phase)

        assert az == pytest.approx(0, abs=1e-5)

    def test_n_dimensional_arrays_keep_same_shape(self):
        shape = (19, 47)
        incidence = np.zeros(shape)
        emission = np.zeros(shape)
        phase = np.zeros(shape)

        az = azimuth(incidence, emission, phase)

        assert az.shape == shape
