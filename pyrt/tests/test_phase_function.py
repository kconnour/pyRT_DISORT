import numpy as np
import pytest
from pyrt.phase_function import decompose, fit_asymmetry_parameter, \
    set_negative_coefficients_to_0, construct_henyey_greenstein, \
    henyey_greenstein_legendre_coefficients


class TestDecompose:
    def test_function_matches_hg_result(self):
        ang = np.linspace(0, 180, num=181)
        pf = construct_henyey_greenstein(0.5, ang) * 4 * np.pi  # normalize it
        coeff = henyey_greenstein_legendre_coefficients(0.5, 129)

        lc = decompose(pf, ang, 129)

        assert np.amax(np.abs(lc - coeff)) < 1e-10


class TestFitAsymmetryParamter:
    def test_function_reproduces_hg_phase_function(self):
        g = 0.8
        sa = np.linspace(0, 180, num=18001)
        pf = construct_henyey_greenstein(g, sa) * 4 * np.pi

        fit_g = fit_asymmetry_parameter(pf, sa)

        assert 0 < abs(g - fit_g) < 0.01


class TestSetNegativeCoefficientsTo0:
    def test_function_output_is_equal_to_input_if_no_negative_coefficients(self):
        coeff = np.linspace(1, 50)

        new_coeff = set_negative_coefficients_to_0(coeff)

        assert np.array_equal(new_coeff, coeff)

    def test_function_does_nothing_to_coefficients_before_first_negative(self):
        coeff = np.linspace(10, -10, num=50)

        new_coeff = set_negative_coefficients_to_0(coeff)

        assert np.array_equal(new_coeff[:25], coeff[:25])

    def test_function_zeros_coefficients_after_first_negative(self):
        coeff = np.linspace(10, -10, num=50)

        new_coeff = set_negative_coefficients_to_0(coeff)

        assert np.all(new_coeff[25:] == 0)


class TestConstructHenyeyGreenstein:
    def test_function_is_normalized_to_1(self):
        # Check the integral equals 1 via Riemann summation
        g = 0.5
        sa = np.linspace(0, 180, num=1801)
        step_size = np.pi / len(sa)

        pf = construct_henyey_greenstein(g, sa)

        assert np.sum(pf * np.sin(np.radians(sa))) * 2 * np.pi * step_size == pytest.approx(1, 0.001)

    def test_g_equals_0_gives_same_value_everywhere(self):
        g = 0
        sa = np.linspace(0, 180, num=1801)

        pf = construct_henyey_greenstein(g, sa)

        assert np.all(pf == 1 / (4 * np.pi))


class TestHenyeyGreensteinLegendreCoefficients:
    def test_first_moment_is_1(self):
        g = 0.5
        n_coeff = 200

        legendre = henyey_greenstein_legendre_coefficients(g, n_coeff)

        assert legendre[0] == 1

    def test_function_gives_n_moments(self):
        g = 0.5
        n_coeff = 200

        legendre = henyey_greenstein_legendre_coefficients(g, n_coeff)

        assert len(legendre) == n_coeff

    def test_returned_coefficients_are_monotonically_decreasing_except_0(self):
        g = 0.5
        n_coeff = 200

        legendre = henyey_greenstein_legendre_coefficients(g, n_coeff)

        assert np.all(np.diff(legendre[1:]) < 0)