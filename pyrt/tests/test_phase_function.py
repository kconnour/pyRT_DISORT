import numpy as np
import pytest
from pyrt.phase_function import construct_hg, decompose_hg, decompose, fit_asymmetry_parameter, set_negative_coefficients_to_0


class TestConstructHG:
    def test_function_is_normalized_to_1(self):
        g = 0.5
        sa = np.linspace(0, 180, num=1801)

        pf = construct_hg(g, sa)

        assert np.sum(pf * np.sin(np.radians(sa))) * 2 * np.pi * np.pi / len(sa) == pytest.approx(1, 0.001)

    def test_g_equals_0_gives_same_value_everywhere(self):
        g = 0
        sa = np.linspace(0, 180, num=1801)

        pf = construct_hg(g, sa)

        assert np.all(pf == 1 / (4 * np.pi))


class TestDecomposeHG:
    def test_first_moment_is_1(self):
        legendre = decompose_hg(0.5, 200)

        assert legendre[0] == 1

    def test_function_gives_n_moments(self):
        legendre = decompose_hg(0.5, 200)

        assert legendre[0] == 1


class TestDecompose:
    def test_function_matches_hg_result(self):
        ang = np.linspace(0, 180, num=181)
        pf = construct_hg(0.5, ang) * 4 * np.pi  # normalize it
        coeff = decompose_hg(0.5, 129)

        lc = decompose(pf, ang, 129)

        assert np.amax(np.abs(lc - coeff)) < 1e-10


class TestFitAsymmetryParamter:
    def test_function_reproduces_hg_phase_function(self):
        g = 0.8
        sa = np.linspace(0, 180, num=18001)
        pf = construct_hg(g, sa) * 4 * np.pi

        fit_g = fit_asymmetry_parameter(pf, sa)

        assert 0 < abs(g - fit_g) < 0.01


class TestSetNegativeCoefficientsTo0:
    def test_function_does_nothing_if_no_negative_coefficients(self):
        coeff = np.linspace(1, 50)

        new_coeff = set_negative_coefficients_to_0(coeff)

        assert np.array_equal(new_coeff, coeff)

    def test_function_produces_expected_result(self):
        coeff = np.linspace(10, -10, num=50)

        new_coeff = set_negative_coefficients_to_0(coeff)

        assert np.array_equal(new_coeff[:25], coeff[:25])
        assert np.all(new_coeff[25:] == 0)
