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


'''class TestSpectral:
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
                      self.spectral.low_wavenumber[0])'''
