import numpy as np
import pytest
from pyrt import Column


class TestColumn:
    def test_optical_depth_property_is_unchanged_from_input(self):
        od = np.linspace(0.1, 1, num=15)
        ssa = np.ones((15,))
        pmom = np.ones((2, 15))

        col = Column(od, ssa, pmom)

        assert np.array_equal(col.optical_depth, od)

    def test_single_scattering_albedo_property_is_unchanged_from_input(self):
        od = np.linspace(0.1, 1, num=15)
        ssa = np.linspace(0.1, 0.9, num=15)
        pmom = np.ones((2, 15))

        col = Column(od, ssa, pmom)

        assert np.array_equal(col.single_scattering_albedo, ssa)

    def test_legendre_coefficient_property_is_normalized_from_input(self):
        od = np.linspace(0.1, 1, num=15)
        ssa = np.linspace(0.1, 0.9, num=15)
        pmom = np.ones((128, 15))
        norm = np.arange(128) * 2 + 1

        col = Column(od, ssa, pmom)

        assert np.array_equal(col.legendre_coefficients, (pmom.T / norm).T)

    def test_negative_optical_depth_raises_value_error(self):
        od = np.linspace(-0.1, 1, num=15)
        ssa = np.ones((15,))
        pmom = np.ones((2, 15))

        with pytest.raises(ValueError):
            Column(od, ssa, pmom)

    def test_negative_single_scattering_albedo_raises_value_error(self):
        od = np.linspace(0.1, 1, num=15)
        ssa = np.linspace(-0.1, 0.9, num=15)
        pmom = np.ones((2, 15))

        with pytest.raises(ValueError):
            Column(od, ssa, pmom)

    def test_adding_two_columns_give_sum_of_optical_depths(self):
        od0 = np.linspace(0.1, 1, num=15)
        od1 = np.linspace(0.1, 1, num=15) * 2
        ssa = np.ones((15,)) * 0.5
        pmom = np.ones((2, 15))
        col0 = Column(od0, ssa, pmom)
        col1 = Column(od1, ssa, pmom)
        answer = od0 + od1

        col = col0 + col1

        assert np.array_equal(col.optical_depth, answer)

    def test_adding_two_columns_give_weighted_average_of_single_scattering_albedo(self):
        od0 = np.linspace(0.1, 1, num=15)
        od1 = np.linspace(0.1, 1, num=15) * 2
        ssa0 = np.ones((15,)) * 0.5
        ssa1 = np.ones((15,)) * 0.75
        pmom = np.ones((2, 15))
        col0 = Column(od0, ssa0, pmom)
        col1 = Column(od1, ssa1, pmom)
        answer = np.ones((15,)) * (0.75*2 + 0.5) / 3

        col = col0 + col1

        assert np.allclose(col.single_scattering_albedo, answer)

    def test_adding_two_identical_columns_gives_expected_legendre_coefficients(self):
        od = np.linspace(0.1, 1, num=15)
        ssa = np.ones((15,)) * 0.5
        pmom = np.ones((128, 15))
        col0 = Column(od, ssa, pmom)
        col1 = Column(od, ssa, pmom)
        answer = col0.legendre_coefficients

        col = col0 + col1

        assert np.allclose(col.legendre_coefficients, answer, rtol=1e-01)
