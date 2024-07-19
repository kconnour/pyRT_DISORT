import numpy as np

from pyrt.rayleigh import rayleigh_legendre, rayleigh_co2


class TestRayleighLegendre:
    def test_moment_0_is_always_1(self):
        n_layers = 15
        n_wavelengths = 20

        coefficients = rayleigh_legendre(n_layers, n_wavelengths)

        assert np.all(coefficients[0, :, :] == 1)

    def test_moment_1_is_always_0(self):
        n_layers = 15
        n_wavelengths = 20

        coefficients = rayleigh_legendre(n_layers, n_wavelengths)

        assert np.all(coefficients[1, :, :] == 0)

    def test_moment_2_is_always_half(self):
        n_layers = 15
        n_wavelengths = 20

        coefficients = rayleigh_legendre(n_layers, n_wavelengths)

        assert np.all(coefficients[2, :, :] == 0.5)


class TestRayleighCO2:
    def test_optical_depth_is_monotonically_decreasing(self):
        column_density = np.linspace(1, 10, num=15) * 10**26
        wavelengths = np.linspace(0.2, 50, num=100)

        column = rayleigh_co2(column_density, wavelengths)

        assert np.all(np.diff(np.sum(column.optical_depth, axis=0)) < 0)

    def test_single_scattering_albedo_is_always_1(self):
        column_density = np.linspace(1, 10, num=15) * 10**26
        wavelengths = np.linspace(0.2, 50, num=100)

        column = rayleigh_co2(column_density, wavelengths)

        assert np.all(column.single_scattering_albedo == 1)

    def test_legendre_moment_0_is_always_1(self):
        column_density = np.linspace(1, 10, num=15) * 10**26
        wavelengths = np.linspace(0.2, 50, num=100)

        column = rayleigh_co2(column_density, wavelengths)

        assert np.all(column.legendre_coefficients[0] == 1)

    def test_legendre_moment_1_is_always_0(self):
        column_density = np.linspace(1, 10, num=15) * 10**26
        wavelengths = np.linspace(0.2, 50, num=100)

        column = rayleigh_co2(column_density, wavelengths)

        assert np.all(column.legendre_coefficients[1] == 0)

    def test_legendre_moment_2_is_always_0_1(self):
        # Moment 2 is 0.5, which turns into 0.1 when divided by 2k + 1
        column_density = np.linspace(1, 10, num=15) * 10**26
        wavelengths = np.linspace(0.2, 50, num=100)

        column = rayleigh_co2(column_density, wavelengths)

        assert np.all(column.legendre_coefficients[2] == 0.1)
