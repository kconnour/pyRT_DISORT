# 3rd-party imports
import numpy as np

# Local imports
from preprocessing.model.phase_function import EmpiricalPhaseFunction, HenyeyGreensteinPhaseFunction, RayleighPhaseFunction


def test_empirical():
    legendre_file = '/preprocessing/planets/mars/aux/legendre_coeff_dust.npy'
    n_moments = 95
    n_layers = 20
    n_wavelengths = 100
    empirical = EmpiricalPhaseFunction(legendre_file, n_moments)
    phase_function = empirical.make_phase_function(n_layers, n_wavelengths)
    return np.all(phase_function[-1, :, :] == phase_function[-1, :, :])


def test_henyey_greenstein():
    n_moments = 128
    n_layers = 20
    g_values = np.array([-1, -0.5, -0.25, 0.1, np.pi/4, 1])
    hg = HenyeyGreensteinPhaseFunction(n_moments)
    phase_function = hg.make_phase_function(n_layers, g_values)
    last_moment = (2*(n_moments - 1) + 1) * g_values**(n_moments - 1)
    return np.all(phase_function[-1, 0, :] == last_moment)


def test_rayleigh():
    n_moments = 128
    n_layers = 20
    n_wavelengths = 100
    rayleigh = RayleighPhaseFunction(n_moments)
    phase_function = rayleigh.make_phase_function(n_layers, n_wavelengths)
    return np.all(phase_function[0, :, :] == 1) and np.all(phase_function[1:, :, :] == 0)


print(test_empirical())
print(test_henyey_greenstein())
print(test_rayleigh())
