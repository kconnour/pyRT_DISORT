"""The phase function module contains structures to create a variety of phase
function arrays required by DISORT.
"""
import numpy as np


class TabularLegendreCoefficients:
    """An abstract class from which all coefficient classes are derived.

    TabularLegendreCoefficients holds a variety of properties that are needed by
    all derived classes. It is not meant to be instantiated.

    """

    def __init__(self, coefficients: np.ndarray, altitude_grid: np.ndarray,
                 wavelengths: np.ndarray, max_moments: int = None) -> None:
        self._coefficients = coefficients
        self._altitude_grid = altitude_grid
        self._wavelengths = wavelengths
        self._max_moments = max_moments
        self._n_layers = len(self._altitude_grid)
        self._n_moments = coefficients.shape[0] if max_moments is None \
            else max_moments

    def _normalize_coefficients(self, unnormalized_coefficients):
        moment_indices = np.linspace(0, self._n_moments-1, num=self._n_moments)
        normalization = moment_indices * 2 + 1
        return (unnormalized_coefficients.T / normalization).T



