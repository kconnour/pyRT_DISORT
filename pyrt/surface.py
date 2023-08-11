"""The :code:`surface` module contains structures for creating arrays related to
DISORT's surface treatment.
"""
import numpy as np
from disort import disobrdf


def _make_empty_bemst(n_streams) -> np.ndarray:
    return np.zeros(int(0.5 * n_streams))


def _make_empty_emust(n_polar) -> np.ndarray:
    return np.zeros(n_polar)


def _make_empty_rho_accurate(n_polar, n_azimuth) -> np.ndarray:
    return np.zeros((n_polar, n_azimuth))


def _make_empty_rhoq(n_streams) -> np.ndarray:
    return np.zeros((int(0.5 * n_streams), int(0.5 * n_streams + 1), n_streams))


def _make_empty_rhou(n_streams) -> np.ndarray:
    return np.zeros((n_streams, int(0.5 * n_streams + 1), n_streams))


def _call_disobrdf(user_angles: bool, only_fluxes: bool, n_polar: int, n_azimuth: int, n_streams: int, mu: float,
                   mu0: float, phi: float, phi0: float,
                   beam_flux: float, albedo: float, phase_function_number: int,
                   brdf_arg: np.ndarray, n_mug: int):
    bemst = _make_empty_bemst(n_streams)
    _emust = _make_empty_emust(n_polar)
    _rho_accurate = _make_empty_rho_accurate(n_polar, n_azimuth)
    _rhou = _make_empty_rhou(n_streams)
    _rhoq = _make_empty_rhoq(n_streams)

    # rhoq, rhou, emust, bemst, rho_accurate
    return disobrdf(user_angles, mu, beam_flux, mu0, False, albedo,
                    only_fluxes, _rhoq, _rhou, _emust,
                    bemst, False, phi, phi0, _rho_accurate,
                    phase_function_number, brdf_arg, n_mug,
                    nstr=n_streams, numu=n_polar,
                    nphi=n_azimuth)


def make_hapke_surface(user_angles: bool, only_fluxes: bool, n_polar: int,
                       n_azimuth: int, n_streams: int, mu: float, mu0: float,
                       phi: float, phi0: float, beam_flux: float, n_mug: int,
                       b0: float, h: float, w: float) -> list[np.ndarray]:
    """Make a basic Hapke surface.

    Parameters
    ----------
    user_angles
        True if this should be returned at user angles; False otherwise.
    only_fluxes
    n_polar
    n_azimuth
    n_streams
    mu
        The cosine of emission.
    mu0
        The cosine of incidence.
    phi
        The azimuth angle.
    phi0
        Phi0
    beam_flux
        The incident beam flux.
    n_mug
        The number of angle quadrature points.
    b0
        The strength of the opposition surge.
    h
        The width of the opposition surge.
    w
        The hapke w parameter.

    """
    brdf_arg = np.array([b0, h, w, 0, 0, 0])
    return _call_disobrdf(user_angles, only_fluxes, n_polar, n_azimuth,
                          n_streams, mu, mu0, phi, phi0, beam_flux,
                          0, 1, brdf_arg, n_mug)


def make_hapkeHG2_surface(user_angles: bool, only_fluxes: bool, n_polar: int,
                          n_azimuth: int, n_streams: int, mu: float, mu0: float,
                          phi: float, phi0: float, beam_flux: float, n_mug: int,
                          b0: float, h: float, w: float, asym: float, frac: float) -> list[np.ndarray]:
    """Make a 2-lobed Henyey-Greenstein Hapke surface.

    Parameters
    ----------
    user_angles
        True if this should be returned at user angles; False otherwise.
    only_fluxes
    n_polar
    n_azimuth
    n_streams
    mu
        The cosine of emission.
    mu0
        The cosine of incidence.
    phi
        The azimuth angle.
    phi0
        Phi0
    beam_flux
        The incident beam flux.
    n_mug
        The number of angle quadrature points.
    b0
        The strength of the opposition surge.
    h
        The width of the opposition surge.
    w
        The surface single scattering albedo.
    asym
        The asymmetry parameter (b)
    frac
        The forward scattering fraction (c)

    """
    brdf_arg = np.array([b0, h, w, asym, frac, 0])
    return _call_disobrdf(user_angles, only_fluxes, n_polar, n_azimuth,
                          n_streams, mu, mu0, phi, phi0, beam_flux,
                          0, 5, brdf_arg, n_mug)


def make_hapkeHG2roughness_surface(user_angles: bool, only_fluxes: bool, n_polar: int,
                       n_azimuth: int, n_streams: int, mu: float, mu0: float,
                       phi: float, phi0: float, beam_flux: float, n_mug: int,
                       b0: float, h: float, w: float, asym: float, frac: float,
                       roughness: float) -> list[np.ndarray]:
    """Make a 2-lobed Henyey-Greenstein Hapke surface with roughness parameter.

        Parameters
        ----------
        user_angles
            True if this should be returned at user angles; False otherwise.
        only_fluxes
        n_polar
        n_azimuth
        n_streams
        mu
            The cosine of emission.
        mu0
            The cosine of incidence.
        phi
            The azimuth angle.
        phi0
            Phi0
        beam_flux
            The incident beam flux.
        n_mug
            The number of angle quadrature points.
        b0
            The strength of the opposition surge.
        h
            The width of the opposition surge.
        w
            The surface single scattering albedo.
        asym
            The asymmetry parameter (b).
        frac
            The forward scattering fraction (c).
        roughness
            The roughness parameter.
        """
    brdf_arg = np.array([b0, h, w, asym, frac, roughness])
    return _call_disobrdf(user_angles, only_fluxes, n_polar, n_azimuth,
                          n_streams, mu, mu0, phi, phi0, beam_flux,
                          0, 6, brdf_arg, n_mug)
