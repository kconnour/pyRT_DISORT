import numpy as np
from scipy.constants import Boltzmann

from pyrt.eos import column_density


class TestColumnDensity:
    def test_output_matches_analytic_result(self):
        surface_pressure = 600
        scale_height = 10
        altitude = np.flip(np.arange(100))
        pressure = surface_pressure * np.exp(-altitude / 10)
        temperature = np.ones(altitude.shape) * 180
        # This is N = P_0 * H / (k_B * T) * (1 - exp(-z / H) for the special
        #  case where the lower altitude is 0.
        analytical_result = surface_pressure * scale_height * 1000 * (1 - np.exp(-altitude[0] / scale_height)) / 180 / Boltzmann

        colden = column_density(pressure, temperature, altitude)

        assert np.isclose((np.sum(colden) - np.sum(analytical_result)) / np.sum(analytical_result), 0, atol=1e-3)
