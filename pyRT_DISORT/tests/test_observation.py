import numpy as np
import pytest
from pyRT_DISORT.observation import Angles, Spectral


class TestAngles:
    def setup(self) -> None:
        self.dummy_angles = np.outer(np.linspace(5, 10, num=15),
                                     np.linspace(5, 8, num=20))
        self.angles = Angles(self.dummy_angles, self.dummy_angles,
                             self.dummy_angles)


class TestAnglesInit(TestAngles):
    def test_angles_contains_7_attributes(self) -> None:
        assert len(self.angles.__dict__.items()) == 7


class TestIncidence(TestAngles):
    def test_incidence_angle_is_unchanged(self) -> None:
        incidence_angle = self.dummy_angles + 10
        angles = Angles(incidence_angle, self.dummy_angles, self.dummy_angles)
        assert np.array_equal(incidence_angle, angles.incidence)

    def test_incidence_angle_is_read_only(self) -> None:
        with pytest.raises(AttributeError):
            self.angles.incidence = self.dummy_angles

    def test_incidence_angle_outside_0_to_180_raises_value_error(self) -> None:
        Angles(np.array([0]), np.array([1]), np.array([1]))
        Angles(np.array([180]), np.array([1]), np.array([1]))
        too_low = np.array([np.nextafter(0, -1)])
        too_high = np.array([np.nextafter(181, 181)])
        with pytest.raises(ValueError):
            Angles(too_low, np.array([1]), np.array([1]))
        with pytest.raises(ValueError):
            Angles(too_high, np.array([1]), np.array([1]))

    def test_array_of_string_incidence_angle_raises_type_error(self) -> None:
        str_angles = self.dummy_angles.astype('str')
        with pytest.raises(TypeError):
            Angles(str_angles, self.dummy_angles, self.dummy_angles)

    def test_list_incidence_angle_raises_type_error(self) -> None:
        list_angles = self.dummy_angles.tolist()
        with pytest.raises(TypeError):
            Angles(list_angles, self.dummy_angles, self.dummy_angles)

    def test_oddly_shaped_incidence_angles_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            incidence = self.dummy_angles[:, :-1]
            Angles(incidence, self.dummy_angles, self.dummy_angles)


class TestEmission(TestAngles):
    def test_emission_angle_is_unchanged(self) -> None:
        emission_angle = self.dummy_angles + 5
        angles = Angles(self.dummy_angles, emission_angle, self.dummy_angles)
        assert np.array_equal(emission_angle, angles.emission)

    def test_emission_angle_is_read_only(self) -> None:
        with pytest.raises(AttributeError):
            self.angles.emission = self.dummy_angles

    def test_emission_angle_outside_0_to_90_raises_value_error(self) -> None:
        Angles(np.array([1]), np.array([0]), np.array([1]))
        Angles(np.array([1]), np.array([90]), np.array([1]))
        too_low = np.array([np.nextafter(0, -1)])
        too_high = np.array([np.nextafter(90, 91)])
        with pytest.raises(ValueError):
            Angles(np.array([1]), too_low, np.array([1]))
        with pytest.raises(ValueError):
            Angles(np.array([1]), too_high, np.array([1]))

    def test_array_of_string_emission_angle_raises_type_error(self) -> None:
        str_angles = self.dummy_angles.astype('str')
        with pytest.raises(TypeError):
            Angles(self.dummy_angles, str_angles, self.dummy_angles)

    def test_list_emission_angle_raises_type_error(self) -> None:
        list_angles = self.dummy_angles.tolist()
        with pytest.raises(TypeError):
            Angles(self.dummy_angles, list_angles, self.dummy_angles)

    def test_oddly_shaped_emission_angles_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            emission = self.dummy_angles[:, :-1]
            Angles(self.dummy_angles, emission, self.dummy_angles)


class TestPhase(TestAngles):
    def test_phase_angle_is_unchanged(self) -> None:
        phase_angle = self.dummy_angles + 10
        angles = Angles(self.dummy_angles, self.dummy_angles, phase_angle)
        assert np.array_equal(phase_angle, angles.phase)

    def test_phase_angle_is_read_only(self) -> None:
        with pytest.raises(AttributeError):
            self.angles.phase = self.dummy_angles

    def test_phase_angle_outside_0_to_180_raises_value_error(self) -> None:
        Angles(np.array([0]), np.array([1]), np.array([1]))
        Angles(np.array([180]), np.array([1]), np.array([1]))
        too_low = np.array([np.nextafter(0, -1)])
        too_high = np.array([np.nextafter(181, 181)])
        with pytest.raises(ValueError):
            Angles(np.array([1]), np.array([1]), too_low)
        with pytest.raises(ValueError):
            Angles(np.array([1]), np.array([1]), too_high)

    def test_array_of_string_phase_angle_raises_type_error(self) -> None:
        str_angles = self.dummy_angles.astype('str')
        with pytest.raises(TypeError):
            Angles(self.dummy_angles, self.dummy_angles, str_angles)

    def test_list_phase_angle_raises_type_error(self) -> None:
        list_angles = self.dummy_angles.tolist()
        with pytest.raises(TypeError):
            Angles(self.dummy_angles, self.dummy_angles, list_angles)

    def test_oddly_shaped_phase_angles_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            phase = self.dummy_angles[:, :-1]
            Angles(self.dummy_angles, self.dummy_angles, phase)


class TestMu0(TestAngles):
    def test_mu0_matches_analytic_solution(self) -> None:
        incidence = np.array([0, 45, 60, 90, 180])
        expected_mu0 = np.array([1, np.sqrt(2) / 2, 0.5, 0, -1])
        dummy_angles = np.ones(5)
        angles = Angles(incidence, dummy_angles, dummy_angles)
        assert np.allclose(angles.mu0, expected_mu0)

    def test_mu0_is_read_only(self) -> None:
        with pytest.raises(AttributeError):
            self.angles.mu0 = self.dummy_angles

    def test_mu0_is_same_shape_as_incidence_angles(self) -> None:
        assert self.angles.mu0.shape == self.angles.incidence.shape


class TestMu(TestAngles):
    def test_mu_matches_analytic_solution(self) -> None:
        emission = np.array([0, 45, 60, 90])
        expected_mu = np.array([1, np.sqrt(2) / 2, 0.5, 0])
        dummy_angles = np.ones(4)
        angles = Angles(dummy_angles, emission, dummy_angles)
        assert np.allclose(angles.mu, expected_mu)

    def test_mu_is_read_only(self) -> None:
        with pytest.raises(AttributeError):
            self.angles.mu = self.dummy_angles

    def test_mu_is_same_shape_as_emission_angles(self) -> None:
        assert self.angles.mu.shape == self.angles.emission.shape


class TestPhi0(TestAngles):
    def test_phi0_is_always_0(self) -> None:
        assert np.all(self.angles.phi0 == 0)

    def test_phi0_is_read_only(self) -> None:
        with pytest.raises(AttributeError):
            self.angles.phi0 = self.dummy_angles

    def test_phi0_is_same_shape_as_phase_angles(self) -> None:
        assert self.angles.phi0.shape == self.angles.phase.shape


class TestPhi(TestAngles):
    # TODO: once I better understand phi, use analytic cases
    def test_phi_matches_disort_multi_computations(self) -> None:
        dummy_angles = np.array([0, 10, 70])
        expected_phi = np.array([0, 119.747139, 104.764977])
        angles = Angles(dummy_angles, dummy_angles, dummy_angles)
        assert np.allclose(angles.phi, expected_phi)

    def test_phi_is_read_only(self) -> None:
        with pytest.raises(AttributeError):
            self.angles.phi = self.dummy_angles

    def test_phi_is_same_shape_as_phase_angles(self) -> None:
        assert self.angles.phi.shape == self.angles.phase.shape


class TestSpectral:
    def setup(self) -> None:
        dummy_wavelengths = np.array([1, 10, 15, 20])
        grid = np.broadcast_to(dummy_wavelengths, (20, 15, 4))
        width = 0.05
        self.short_wavelength = grid - width
        self.long_wavelength = grid + width
        self.spectral = Spectral(self.short_wavelength, self.long_wavelength)


class TestSpectralInit(TestSpectral):
    def test_spectral_contains_4_attributes(self) -> None:
        assert len(self.spectral.__dict__.items()) == 4


class TestShortWavelength(TestSpectral):
    def test_short_wavelength_is_unchanged(self) -> None:
        dummy_wavs = np.array([1, 10, 15, 20])
        width = 0.05
        short = dummy_wavs - width
        long = dummy_wavs + width
        wavelengths = Spectral(short, long)
        assert np.array_equal(short, wavelengths.short_wavelength)

    def test_short_wavelength_is_read_only(self) -> None:
        with pytest.raises(AttributeError):
            self.spectral.short_wavelength = self.short_wavelength

    def test_non_positive_short_wavelength_raises_value_error(self) -> None:
        positive_short = np.copy(self.short_wavelength)
        zero_short = np.copy(self.short_wavelength)
        negative_short = np.copy(self.short_wavelength)
        positive_short[0, 0, 0] = 1
        zero_short[0, 0, 0] = 0
        negative_short[0, 0, 0] = -1

        Spectral(positive_short, self.long_wavelength)
        with pytest.raises(ValueError):
            Spectral(zero_short, self.long_wavelength)
        with pytest.raises(ValueError):
            Spectral(negative_short, self.long_wavelength)

    def test_nan_short_wavelength_raises_value_error(self) -> None:
        nan_short = np.copy(self.short_wavelength)
        nan_short[0, 0, 0] = np.nan
        with pytest.raises(ValueError):
            Spectral(nan_short, self.long_wavelength)

    def test_inf_short_wavelength_raises_value_error(self) -> None:
        inf_short = np.copy(self.short_wavelength)
        inf_short[0, 0, 0] = np.inf
        with pytest.raises(ValueError):
            Spectral(inf_short, self.long_wavelength)

    def test_list_short_wavelength_raises_type_error(self) -> None:
        list_short = np.copy(self.short_wavelength).tolist()
        with pytest.raises(TypeError):
            Spectral(list_short, self.long_wavelength)

    def test_str_short_wavelength_raises_type_error(self) -> None:
        str_short = np.copy(self.short_wavelength).astype('str')
        with pytest.raises(TypeError):
            Spectral(str_short, self.long_wavelength)

    def test_differently_shaped_wavelengths_raises_value_error(self) -> None:
        short = np.ones(10)
        long = np.ones(11) + 1
        with pytest.raises(ValueError):
            Spectral(short, long)

    def test_same_wavelengths_raises_value_error(self) -> None:
        short = np.linspace(1, 50, num=50)
        with pytest.raises(ValueError):
            Spectral(short, short)

    def test_longer_short_wavelength_raises_value_error(self) -> None:
        short = np.linspace(1, 50, num=50)
        with pytest.raises(ValueError):
            Spectral(short, short - 0.5)


class TestLongWavelength(TestSpectral):
    def test_long_wavelength_is_unchanged(self) -> None:
        dummy_wavs = np.array([1, 10, 15, 20])
        width = 0.05
        short = dummy_wavs - width
        long = dummy_wavs + width
        wavelengths = Spectral(short, long)
        assert np.array_equal(long, wavelengths.long_wavelength)

    def test_long_wavelength_is_read_only(self) -> None:
        with pytest.raises(AttributeError):
            self.spectral.long_wavelength = self.long_wavelength

    def test_nan_long_wavelength_raises_value_error(self) -> None:
        nan_long = np.copy(self.long_wavelength)
        nan_long[0, 0, 0] = np.nan
        with pytest.raises(ValueError):
            Spectral(self.short_wavelength, nan_long)

    def test_inf_long_wavelength_raises_value_error(self) -> None:
        inf_long = np.copy(self.long_wavelength)
        inf_long[0, 0, 0] = np.inf
        with pytest.raises(ValueError):
            Spectral(self.short_wavelength, inf_long)

    def test_list_long_wavelength_raises_type_error(self) -> None:
        list_long = np.copy(self.long_wavelength).tolist()
        with pytest.raises(TypeError):
            Spectral(self.short_wavelength, list_long)

    def test_str_long_wavelength_raises_type_error(self) -> None:
        str_long = np.copy(self.long_wavelength).astype('str')
        with pytest.raises(TypeError):
            Spectral(self.short_wavelength, str_long)


class TestHighWavenumber(TestSpectral):
    def test_high_wavenumber_matches_known_values(self):
        wavelengths = np.array([1, 10, 15])
        wavenumbers = np.array([10000, 1000, 666.66666])
        spec = Spectral(wavelengths, wavelengths + 1)
        assert np.allclose(spec.high_wavenumber, wavenumbers)

    def test_high_wavenumbers_is_read_only(self):
        with pytest.raises(AttributeError):
            self.spectral.high_wavenumber = 10**4 / self.short_wavelength

    def test_high_wavenumber_is_same_shape_as_short_wavelength(self) -> None:
        assert self.spectral.high_wavenumber.shape == \
               self.spectral.short_wavelength.shape

    def test_all_pixels_have_same_high_wavenumber(self) -> None:
        assert np.all(self.spectral.high_wavenumber ==
                      self.spectral.high_wavenumber[0])


class TestLowWavenumber(TestSpectral):
    def test_low_wavenumber_matches_known_values(self):
        wavelengths = np.array([1, 10, 15])
        wavenumbers = np.array([10000, 1000, 666.66666])
        spec = Spectral(wavelengths - 0.5, wavelengths)
        assert np.allclose(spec.low_wavenumber, wavenumbers)

    def test_low_wavenumbers_is_read_only(self):
        with pytest.raises(AttributeError):
            self.spectral.low_wavenumber = 10**4 / self.long_wavelength

    def test_low_wavenumber_is_same_shape_as_long_wavelength(self) -> None:
        assert self.spectral.low_wavenumber.shape == \
               self.spectral.long_wavelength.shape

    def test_all_pixels_have_same_low_wavenumber(self) -> None:
        assert np.all(self.spectral.low_wavenumber ==
                      self.spectral.low_wavenumber[0])
