import numpy as np
import pytest
from pyRT_DISORT.vertical_profile import Conrath


class TestConrath:
    def setup(self):
        self.grid_1d = np.linspace(0, 50, num=11)
        self.grid_2d = np.broadcast_to(self.grid_1d, (2, 11)).T
        self.q0_0d = 1
        self.q0_1d = np.array([1, 10])
        self.H_0d = 10
        self.H_1d = np.array([10, 15])
        self.nu_0d = 0.5
        self.nu_1d = np.array([0.5, 0.6])


class TestConrathInit(TestConrath):
    def test_1d_profile_makes_1d_output(self):
        c = Conrath(self.grid_1d, self.q0_0d, self.H_0d, self.nu_0d)
        assert np.ndim(c.profile) == 1

    def test_2d_profile_makes_2d_output(self):
        c = Conrath(self.grid_2d, self.q0_0d, self.H_0d, self.nu_0d)
        assert np.ndim(c.profile) == 2

    def test_2d_profile_and_1d_scale_height_makes_2d_output(self):
        c = Conrath(self.grid_2d, self.q0_0d, self.H_1d, self.nu_0d)
        assert np.ndim(c.profile) == 2

    def test_2d_profile_and_1d_nu_makes_2d_output(self):
        c = Conrath(self.grid_2d, self.q0_0d, self.H_0d, self.nu_1d)
        assert np.ndim(c.profile) == 2

    def test_2d_profile_and_1d_parameters_makes_2d_output(self):
        c = Conrath(self.grid_2d, self.q0_1d, self.H_1d, self.nu_1d)
        assert np.ndim(c.profile) == 2

    def test_2d_profile_and_list_parameters_makes_2d_output(self):
        c = Conrath(self.grid_2d, self.q0_1d.tolist(),
                    self.H_1d.tolist(), self.nu_1d.tolist())
        assert np.ndim(c.profile) == 2

    def test_surface_profile_is_q0(self) -> None:
        c = Conrath(self.grid_1d, self.q0_0d, self.H_0d, self.nu_0d)
        assert c.profile[0] == self.q0_0d

    def test_all_values_are_equal_when_nu_equals_0(self):
        q0 = 10
        c = Conrath(self.grid_2d, q0, self.H_0d, 0)
        assert np.all(c.profile == q0)

    def test_non_broadcastable_arrays_raises_value_error(self):
        q0 = np.array([5, 6, 7])
        with pytest.raises(ValueError):
            Conrath(self.grid_2d, q0, self.H_1d, self.nu_1d)


class TestProfile(TestConrath):
    def test_profile_is_read_only(self):
        c = Conrath(self.grid_2d, self.q0_1d, self.H_1d, self.nu_1d)
        with pytest.raises(AttributeError):
            c.profile = self.grid_2d
