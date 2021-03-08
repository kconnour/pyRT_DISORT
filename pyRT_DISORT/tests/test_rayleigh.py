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
    def test_phase_function_is_read_only(self) -> None:
        r = Rayleigh(self.altitude_grid, self.wavenumbers_1d)
        with pytest.raises(AttributeError):
            r.phase_function = self.altitude_grid


class TestRayleighCO2Init(TestRayleigh):
    def test_altitudes_and_column_density_can_be_different_sizes(self) -> None:
        column_densities = np.ones(11)
        altitudes = np.linspace(0, 50, num=10)
        RayleighCO2(altitudes, self.wavenumbers_2d, column_densities)

    def test_scattering_optical_depth_is_correct_shape(self) -> None:
        column_den = np.ones(11)
        r = RayleighCO2(self.altitude_grid, self.wavenumbers_2d, column_den)
        assert r.scattering_optical_depth.shape == (11, 5, 10)

    def test_scattering_optical_depth_matches_known_input(self) -> None:
        # This was taken from the Rayleigh-only integrated test
        altitude_grid = np.linspace(80, 0, num=20)
        wavelengths = np.array([1, 9.3])
        wavenumbers = 1 / wavelengths * 10**4
        column_density = np.array([3.68230187e+20, 5.94065367e+20,
                                   9.68880126e+20, 1.58051881e+21,
                                   2.57563277e+21, 4.18727702e+21,
                                   6.78079484e+21, 1.09003489e+22,
                                   1.73071997e+22, 2.70092591e+22,
                                   4.15818655e+22, 6.31703015e+22,
                                   9.44743677e+22, 1.38849116e+23,
                                   1.99743729e+23, 2.81441206e+23,
                                   3.89008286e+23, 5.31013473e+23,
                                   7.14045817e+23])
        r = RayleighCO2(altitude_grid, wavenumbers, column_density)
        known_sum = np.array([2.59701496e-07, 3.42070917e-11])
        test_sum = np.sum(r.scattering_optical_depth, axis=0)
        assert np.allclose(known_sum, test_sum)


class TestScatteringOpticalDepth(TestRayleigh):
    def test_scattering_optical_depth_is_read_only(self) -> None:
        r = RayleighCO2(self.altitude_grid, self.wavenumbers_1d, np.ones(11))
        with pytest.raises(AttributeError):
            r.scattering_optical_depth = self.altitude_grid
