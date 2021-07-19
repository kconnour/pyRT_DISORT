import numpy as np
import pytest
from pyRT_DISORT.observation import Angles, Spectral


class TestAngles:
    @pytest.fixture
    def dummy_angles(self) -> np.ndarray:
        yield np.outer(np.linspace(5, 10, num=15), np.linspace(5, 8, num=20))


class TestAnglesInit(TestAngles):
    @pytest.fixture
    def angles(self, dummy_angles: pytest.fixture) -> Angles:
        yield Angles(dummy_angles, dummy_angles, dummy_angles)

    def test_angles_contains_7_attributes(self, angles: pytest.fixture) -> None:
        assert len(angles.__dict__.items()) == 7


class TestIncidence(TestAngles):
    @pytest.fixture
    def dummy_incidence(self, dummy_angles: pytest.fixture) -> np.ndarray:
        yield dummy_angles + 10

    @pytest.fixture
    def incidence(self, dummy_incidence: pytest.fixture,
                  dummy_angles: pytest.fixture) -> Angles:
        yield Angles(dummy_incidence, dummy_angles, dummy_angles)

    def test_incidence_is_unchanged(self, dummy_incidence: pytest.fixture,
                                    incidence: pytest.fixture) -> None:
        assert np.array_equal(dummy_incidence, incidence.incidence)

    def test_incidence_is_read_only(self, incidence: pytest.fixture) -> None:
        with pytest.raises(AttributeError):
            incidence.incidence = 0

    def test_incidence_of_0_raises_no_errors(self) -> None:
        Angles(np.array([0]), np.array([1]), np.array([1]))

    def test_incidence_less_than_0_raises_value_error(self) -> None:
        incidence = np.array([np.nextafter(0, -1)])
        with pytest.raises(ValueError):
            Angles(incidence, np.array([1]), np.array([1]))

    def test_incidence_of_180_raises_no_errors(self) -> None:
        Angles(np.array([180]), np.array([1]), np.array([1]))

    def test_nan_incidence_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Angles(np.array([np.nan]), np.array([1]), np.array([1]))

    def test_incidence_greater_than_180_raises_value_error(self) -> None:
        incidence = np.array([np.nextafter(180, 181)])
        with pytest.raises(ValueError):
            Angles(incidence, np.array([1]), np.array([1]))

    def test_list_incidence_raises_type_error(
            self, dummy_angles: pytest.fixture) -> None:
        with pytest.raises(TypeError):
            Angles(dummy_angles.tolist(), dummy_angles, dummy_angles)

    def test_oddly_shaped_incidence_raises_value_error(
            self, dummy_angles: pytest.fixture) -> None:
        incidence = dummy_angles[:, :-1]
        with pytest.raises(ValueError):
            Angles(incidence, dummy_angles, dummy_angles)


class TestEmission(TestAngles):
    @pytest.fixture
    def dummy_emission(self, dummy_angles: pytest.fixture) -> np.ndarray:
        yield dummy_angles + 5

    @pytest.fixture
    def emission(self, dummy_emission: pytest.fixture,
                 dummy_angles: pytest.fixture) -> Angles:
        yield Angles(dummy_angles, dummy_emission, dummy_angles)

    def test_emission_is_unchanged(self, dummy_emission: pytest.fixture,
                                   emission: pytest.fixture) -> None:
        assert np.array_equal(dummy_emission, emission.emission)

    def test_emission_is_read_only(self, emission: pytest.fixture) -> None:
        with pytest.raises(AttributeError):
            emission.emission = 0

    def test_emission_of_0_raises_no_errors(self) -> None:
        Angles(np.array([1]), np.array([0]), np.array([1]))

    def test_emission_less_than_0_raises_value_error(self) -> None:
        emission = np.array([np.nextafter(0, -1)])
        with pytest.raises(ValueError):
            Angles(np.array([1]), emission, np.array([1]))

    def test_emission_of_90_raises_no_errors(self) -> None:
        Angles(np.array([1]), np.array([90]), np.array([1]))

    def test_emission_greater_than_90_raises_value_error(self) -> None:
        emission = np.array([np.nextafter(90, 91)])
        with pytest.raises(ValueError):
            Angles(np.array([1]), emission, np.array([1]))

    def test_nan_emission_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Angles(np.array([1]), np.array([np.nan]), np.array([1]))

    def test_list_emission_raises_type_error(
            self, dummy_angles: pytest.fixture) -> None:
        with pytest.raises(TypeError):
            Angles(dummy_angles, dummy_angles.tolist(), dummy_angles)

    def test_oddly_shaped_emission_raises_value_error(
            self, dummy_angles: pytest.fixture) -> None:
        emission = dummy_angles[:, :-1]
        with pytest.raises(ValueError):
            Angles(dummy_angles, emission, dummy_angles)


class TestPhase(TestAngles):
    @pytest.fixture
    def dummy_phase(self, dummy_angles: pytest.fixture) -> np.ndarray:
        yield dummy_angles + 10

    @pytest.fixture
    def phase(self, dummy_phase: pytest.fixture,
              dummy_angles: pytest.fixture) -> Angles:
        yield Angles(dummy_angles, dummy_angles, dummy_phase)

    def test_phase_is_unchanged(self, dummy_phase: pytest.fixture,
                                phase: pytest.fixture) -> None:
        assert np.array_equal(dummy_phase, phase.phase)

    def test_phase_is_read_only(self, phase: pytest.fixture) -> None:
        with pytest.raises(AttributeError):
            phase.phase = 0

    def test_phase_of_0_raises_no_errors(self) -> None:
        Angles(np.array([1]), np.array([1]), np.array([0]))

    def test_phase_less_than_0_raises_value_error(self) -> None:
        phase = np.array([np.nextafter(0, -1)])
        with pytest.raises(ValueError):
            Angles(np.array([1]), np.array([1]), phase)

    def test_phase_of_180_raises_no_errors(self) -> None:
        Angles(np.array([1]), np.array([1]), np.array([180]))

    def test_phase_greater_than_180_raises_value_error(self) -> None:
        phase = np.array([np.nextafter(180, 181)])
        with pytest.raises(ValueError):
            Angles(np.array([1]), np.array([1]), phase)

    def test_nan_phase_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Angles(np.array([1]), np.array([1]), np.array([np.nan]))

    def test_list_phase_raises_type_error(
            self, dummy_angles: pytest.fixture) -> None:
        with pytest.raises(TypeError):
            Angles(dummy_angles, dummy_angles, dummy_angles.tolist())

    def test_oddly_shaped_phase_raises_value_error(
            self, dummy_angles: pytest.fixture) -> None:
        phase = dummy_angles[:, :-1]
        with pytest.raises(ValueError):
            Angles(dummy_angles, dummy_angles, phase)


class TestMu0(TestAngles):
    @pytest.fixture
    def known_incidence(self) -> np.ndarray:
        yield np.array([0, 45, 60, 90, 180])

    @pytest.fixture
    def known_mu0(self) -> np.ndarray:
        yield np.array([1, np.sqrt(2) / 2, 0.5, 0, -1])

    @pytest.fixture
    def angles(self, known_incidence: pytest.fixture) -> Angles:
        dummy_angles = np.ones(known_incidence.shape)
        yield Angles(known_incidence, dummy_angles, dummy_angles)

    def test_mu0_matches_analytic_solution(self, angles: pytest.fixture,
                                           known_mu0: pytest.fixture) -> None:
        assert np.allclose(angles.mu0, known_mu0)

    def test_mu0_is_read_only(self, angles: pytest.fixture) -> None:
        with pytest.raises(AttributeError):
            angles.mu0 = 0

    def test_mu0_is_same_shape_as_incidence_angles(
            self, angles: pytest.fixture,
            known_incidence: pytest.fixture) -> None:
        assert angles.mu0.shape == known_incidence.shape


class TestMu(TestAngles):
    @pytest.fixture
    def known_emission(self) -> np.ndarray:
        yield np.array([0, 45, 60, 90])

    @pytest.fixture
    def known_mu(self) -> np.ndarray:
        yield np.array([1, np.sqrt(2) / 2, 0.5, 0])

    @pytest.fixture
    def angles(self, known_emission: pytest.fixture) -> Angles:
        dummy_angles = np.ones(known_emission.shape)
        yield Angles(dummy_angles, known_emission, dummy_angles)

    def test_mu_matches_analytic_solution(self, angles: pytest.fixture,
                                          known_mu: pytest.fixture) -> None:
        assert np.allclose(angles.mu, known_mu)

    def test_mu_is_read_only(self, angles: pytest.fixture) -> None:
        with pytest.raises(AttributeError):
            angles.mu = 0

    def test_mu_is_same_shape_as_emission_angles(
            self, angles: pytest.fixture,
            known_emission: pytest.fixture) -> None:
        assert angles.mu.shape == known_emission.shape


class TestPhi0(TestAngles):
    @pytest.fixture
    def known_phase(self) -> np.ndarray:
        yield np.array([0, 45, 60, 90, 180])

    @pytest.fixture
    def known_phi0(self, known_phase: pytest.fixture) -> np.ndarray:
        yield np.zeros(known_phase.shape)

    @pytest.fixture
    def angles(self, known_phase: pytest.fixture) -> Angles:
        dummy_angles = np.ones(known_phase.shape)
        yield Angles(dummy_angles, dummy_angles, known_phase)

    def test_phi0_is_always_0(self, angles: pytest.fixture) -> None:
        assert np.all(angles.phi0 == 0)

    def test_phi0_is_read_only(self, angles: pytest.fixture) -> None:
        with pytest.raises(AttributeError):
            angles.phi0 = 0

    def test_phi0_is_same_shape_as_phase_angles(
            self, angles: pytest.fixture, known_phase: pytest.fixture) -> None:
        assert angles.phi0.shape == known_phase.shape


class TestPhi(TestAngles):
    @pytest.fixture
    def known_phase(self) -> np.ndarray:
        yield np.array([0, 10, 70])

    @pytest.fixture
    def known_phi(self) -> np.ndarray:
        yield np.array([0, 119.747139, 104.764977])

    @pytest.fixture
    def angles(self, known_phase: pytest.fixture) -> Angles:
        yield Angles(known_phase, known_phase, known_phase)

    # TODO: once I better understand phi, use analytic cases
    def test_phi_matches_disort_multi_computations(
            self, angles: pytest.fixture, known_phi: pytest.fixture) -> None:
        assert np.allclose(angles.phi, known_phi)

    def test_phi_is_read_only(self, angles: pytest.fixture) -> None:
        with pytest.raises(AttributeError):
            self.angles.phi = self.dummy_angles

    def test_phi_is_same_shape_as_phase_angles(
            self, angles: pytest.fixture, known_phase: pytest.fixture) -> None:
        assert angles.phi.shape == known_phase.shape


class TestSpectral:
    @pytest.fixture
    def dummy_wavelengths(self) -> np.ndarray:
        yield np.array([1, 10, 15, 20], dtype=float)

    @pytest.fixture
    def wavelength_grid(self, dummy_wavelengths: pytest.fixture) -> np.ndarray:
        yield np.broadcast_to(dummy_wavelengths, (20, 15, 4)).T

    @pytest.fixture
    def width(self) -> float:
        yield 0.05

    @pytest.fixture
    def spectral(self, wavelength_grid: pytest.fixture,
                 width: pytest.fixture) -> Spectral:
        yield Spectral(wavelength_grid - width, wavelength_grid + width)

    @pytest.fixture
    def known_wavelengths(self) -> np.ndarray:
        yield np.array([1, 10, 15])

    @pytest.fixture
    def known_wavenumbers(self) -> np.ndarray:
        yield np.array([10000, 1000, 666.66666])


class TestSpectralInit(TestSpectral):
    def test_spectral_contains_4_attributes(
            self, spectral: pytest.fixture) -> None:
        assert len(spectral.__dict__.items()) == 4


class TestShortWavelength(TestSpectral):
    def test_short_wavelength_is_unchanged(
            self, spectral: pytest.fixture, wavelength_grid: pytest.fixture,
            width: pytest.fixture) -> None:
        assert np.array_equal(wavelength_grid - width,
                              spectral.short_wavelength)

    def test_short_wavelength_is_read_only(
            self, spectral: pytest.fixture) -> None:
        with pytest.raises(AttributeError):
            spectral.short_wavelength = 0

    def test_short_wavelength_of_01_raises_no_errors(
            self, wavelength_grid: pytest.fixture) -> None:
        wg = np.copy(wavelength_grid)
        wg[0, 0, 0] = 0.1
        Spectral(wg, wg + 1)

    def test_short_wavelength_below_01_raises_value_error(
            self, wavelength_grid: pytest.fixture) -> None:
        wg = np.copy(wavelength_grid)
        wg[0, 0, 0] = np.nextafter(0.1, 0)
        with pytest.raises(ValueError):
            Spectral(wg, wg + 1)

    def test_nan_short_wavelength_raises_value_error(
            self, wavelength_grid: pytest.fixture) -> None:
        nan_short = np.copy(wavelength_grid)
        nan_short[0, 0, 0] = np.nan
        with pytest.raises(ValueError):
            Spectral(nan_short, nan_short + 1)

    def test_inf_short_wavelength_raises_value_error(
            self, wavelength_grid: pytest.fixture) -> None:
        inf_short = np.copy(wavelength_grid)
        inf_short[0, 0, 0] = np.inf
        with pytest.raises(ValueError):
            Spectral(inf_short, inf_short + 1)

    def test_list_short_wavelength_raises_type_error(
            self, wavelength_grid: pytest.fixture) -> None:
        with pytest.raises(TypeError):
            Spectral(wavelength_grid.tolist(), wavelength_grid + 1)

    def test_differently_shaped_wavelengths_raises_value_error(self) -> None:
        short = np.ones(10)
        long = np.ones(11) + 1
        with pytest.raises(ValueError):
            Spectral(short, long)

    def test_same_wavelengths_raises_value_error(
            self, wavelength_grid: pytest.fixture) -> None:
        with pytest.raises(ValueError):
            Spectral(wavelength_grid, wavelength_grid)

    def test_longer_short_wavelength_raises_value_error(
            self, wavelength_grid: pytest.fixture) -> None:
        with pytest.raises(ValueError):
            Spectral(wavelength_grid + 1, wavelength_grid)


class TestLongWavelength(TestSpectral):
    def test_long_wavelength_is_unchanged(
            self, spectral: pytest.fixture, wavelength_grid: pytest.fixture,
            width: pytest.fixture) -> None:
        assert np.array_equal(wavelength_grid + width, spectral.long_wavelength)

    def test_long_wavelength_is_read_only(
            self, spectral: pytest.fixture) -> None:
        with pytest.raises(AttributeError):
            spectral.long_wavelength = 0

    def test_long_wavelength_of_50_raises_no_errors(
            self, wavelength_grid: pytest.fixture) -> None:
        wg = np.copy(wavelength_grid)
        wg[-1, -1, -1] = 50.
        Spectral(wg - 0.5, wg)

    def test_long_wavelength_above_50_raises_value_error(
            self, wavelength_grid: pytest.fixture) -> None:
        wg = np.copy(wavelength_grid)
        wg[0, 0, 0] = np.nextafter(50, 51)
        with pytest.raises(ValueError):
            Spectral(wg - 0.5, wg)

    def test_nan_long_wavelength_raises_value_error(
            self, wavelength_grid: pytest.fixture) -> None:
        nan_long = np.copy(wavelength_grid)
        nan_long[0, 0, 0] = np.nan
        with pytest.raises(ValueError):
            Spectral(nan_long - 0.5, nan_long)

    def test_inf_long_wavelength_raises_value_error(
            self, wavelength_grid: pytest.fixture) -> None:
        inf_long = np.copy(wavelength_grid)
        inf_long[0, 0, 0] = np.inf
        with pytest.raises(ValueError):
            Spectral(inf_long - 1, inf_long)

    def test_list_long_wavelength_raises_type_error(
            self, wavelength_grid: pytest.fixture) -> None:
        with pytest.raises(TypeError):
            Spectral(wavelength_grid - 1, wavelength_grid.tolist())


class TestHighWavenumber(TestSpectral):
    @pytest.fixture
    def known_spectral(self, known_wavelengths: pytest.fixture) -> Spectral:
        yield Spectral(known_wavelengths, known_wavelengths + 1)

    def test_high_wavenumber_matches_known_values(
            self, known_spectral: pytest.fixture,
            known_wavenumbers: pytest.fixture) -> None:
        assert np.allclose(known_spectral.high_wavenumber, known_wavenumbers)

    def test_high_wavenumber_is_read_only(
            self, spectral: pytest.fixture) -> None:
        with pytest.raises(AttributeError):
            spectral.high_wavenumber = 0

    def test_high_wavenumber_is_same_shape_as_short_wavelength(
            self, spectral: pytest.fixture) -> None:
        assert spectral.high_wavenumber.shape == spectral.short_wavelength.shape


class TestLowWavenumber(TestSpectral):
    @pytest.fixture
    def known_spectral(self, known_wavelengths: pytest.fixture) -> Spectral:
        yield Spectral(known_wavelengths - 0.5, known_wavelengths)

    def test_low_wavenumber_matches_known_values(
            self, known_spectral: pytest.fixture,
            known_wavenumbers: pytest.fixture) -> None:
        assert np.allclose(known_spectral.low_wavenumber, known_wavenumbers)

    def test_low_wavenumber_is_read_only(
            self, spectral: pytest.fixture) -> None:
        with pytest.raises(AttributeError):
            spectral.low_wavenumber = 0

    def test_low_wavenumber_is_same_shape_as_long_wavelength(
            self, spectral: pytest.fixture) -> None:
        assert spectral.low_wavenumber.shape == spectral.long_wavelength.shape
