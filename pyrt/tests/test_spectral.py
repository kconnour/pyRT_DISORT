import numpy as np

from pyrt.spectral import wavenumber


class TestWavenumber:
    def test_1_micron_wavelength_gives_expected_result(self):
        wavelength = 1
        expected_wavenumber = 10000

        result = wavenumber(wavelength)

        assert result == expected_wavenumber

    def test_10_microns_wavelength_gives_expected_result(self):
        wavelength = 10
        expected_wavenumber = 1000

        result = wavenumber(wavelength)

        assert result == expected_wavenumber

    def test_output_array_is_same_shape_as_input(self):
        wavelengths = np.ones((4, 5, 6))

        wavenumbers = wavenumber(wavelengths)

        assert wavenumbers.shape == wavelengths.shape
