import numpy as np
import pytest
from pyRT_DISORT.rayleigh import Rayleigh, RayleighCO2


class TestRayleigh:
    def setup(self):
        self.altitude_grid = np.linspace(0, 50, num=11)
        self.wavenumbers_1d = np.linspace(1, 2, num=10) * 10000
        self.wavenumbers_2d = np.broadcast_to(self.wavenumbers_1d, (5, 10))


class TestRayleighInit(TestRayleigh):
    def test_1d_wavenumbers_makes_3d_phase_function(self) -> None:
        r = Rayleigh(self.altitude_grid, self.wavenumbers_1d)
        assert np.ndim(r.phase_function) == 3

    def test_2d_wavenumbers_makes_4d_phase_function(self) -> None:
        r = Rayleigh(self.altitude_grid, self.wavenumbers_2d)
        assert np.ndim(r.phase_function) == 4

    def test_all_0th_coefficient_equals_1(self) -> None:
        r = Rayleigh(self.altitude_grid, self.wavenumbers_1d)
        assert np.all(r.phase_function[0, :, :] == 1)


class TestPhaseFunction(TestRayleigh):
    def test_phase_function_is_read_only(self):
        r = Rayleigh(self.altitude_grid, self.wavenumbers_1d)
        with pytest.raises(AttributeError):
            r.phase_function = self.altitude_grid


class TestRayleighCO2Init(TestRayleigh):
    def test_ffo(self):
        cden = np.ones(11)
        r = RayleighCO2(self.wavenumbers_2d, self.wavenumbers_2d, cden)
        print(r.phase_function.shape)
        print(r.scattering_optical_depth.shape)

