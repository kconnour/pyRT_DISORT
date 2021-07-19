import os
import numpy as np
import pytest
from pyRT_DISORT.phase_function import StaticTabularLegendreCoefficients, \
    RadialSpectralTabularLegendreCoefficients
from astropy.io import fits


class TestStaticTabularLegendreCoefficientsInit:
    def setup(self):
        self.coefficients = np.linspace(1, 0.001, num=65)
        self.altitudes = np.linspace(80, 0, num=20)
        self.wavelengths = np.linspace(1, 10, num=15)

    def test_phase_function_has_proper_shape(self) -> None:
        w = np.zeros((3, 17))
        stlc = StaticTabularLegendreCoefficients(
            self.coefficients, self.altitudes, w, max_moments=60)
        assert stlc.phase_function.shape == (60, 20, 3, 17)

    def test_phase_function_is_independent_of_layer(self) -> None:
        stlc = StaticTabularLegendreCoefficients(
            self.coefficients, self.altitudes, self.wavelengths)
        assert np.array_equal(stlc.phase_function[:, 0, :],
                              stlc.phase_function[:, -1, :])

    def test_0th_coefficient_is_unchanged(self) -> None:
        stlc = StaticTabularLegendreCoefficients(
            self.coefficients, self.altitudes, self.wavelengths)
        assert np.all(stlc.phase_function[0, :, :] == 1)

    def test_phase_function_is_read_only(self) -> None:
        stlc = StaticTabularLegendreCoefficients(
            self.coefficients, self.altitudes, self.wavelengths)
        with pytest.raises(AttributeError):
            stlc.phase_function = 0


class TestRadialSpectralTabularLegendreCoefficientsInit:
    def setup(self) -> None:
        path = os.path.realpath(os.path.join(
            os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'),
            'tests/aux/dust_phase_function.fits'))
        hdulist = fits.open(path)
        self.phase_function = hdulist['primary'].data
        self.particle_size_grid = hdulist['particle_sizes'].data
        self.wavelength_grid = hdulist['wavelengths'].data
        self.altitude_grid = np.linspace(80, 0, num=20)
        self.particle_sizes = np.linspace(1, 1.5, num=15)
        self.wavelengths_1d = np.linspace(1, 10, num=10)
        self.wavelengths_2d = np.ones((5, 10)) * self.wavelengths_1d

    def test_1d_wavelengths_give_correcly_shaped_phase_function(self) -> None:
        rstlc = RadialSpectralTabularLegendreCoefficients(
            self.phase_function, self.particle_size_grid, self.wavelength_grid,
            self.altitude_grid[:-1], self.wavelengths_1d, self.particle_sizes)
        assert rstlc.phase_function.shape == (65, 15, 10)

    def test_2d_wavelengths_give_correcly_shaped_phase_function(self) -> None:
        rstlc = RadialSpectralTabularLegendreCoefficients(
            self.phase_function, self.particle_size_grid, self.wavelength_grid,
            self.altitude_grid, self.wavelengths_2d, self.particle_sizes)
        assert rstlc.phase_function.shape == (65, 15, 5, 10)

    def test_max_of_phase_function_is_1(self) -> None:
        rstlc = RadialSpectralTabularLegendreCoefficients(
            self.phase_function, self.particle_size_grid, self.wavelength_grid,
            self.altitude_grid, self.wavelengths_1d, self.particle_sizes)
        assert np.amax(rstlc.phase_function) == 1

    def test_phase_function_is_read_only(self) -> None:
        rstlc = RadialSpectralTabularLegendreCoefficients(
            self.phase_function, self.particle_size_grid, self.wavelength_grid,
            self.altitude_grid, self.wavelengths_1d, self.particle_sizes)
        with pytest.raises(AttributeError):
            rstlc.phase_function = 0
