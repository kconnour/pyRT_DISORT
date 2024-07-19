"""The :code:`surface` module contains structures for creating arguments related
to DISORT's surface treatment.
"""
import numpy as np


def make_empty_bemst(n_streams: int) -> np.ndarray:
    """Make an empty bemst array.

    Parameters
    ----------
    n_streams
        The number of streams.

    Returns
    -------
    Array of zeros

    """
    return np.zeros(int(0.5 * n_streams))


def make_empty_emust(n_polar: int) -> np.ndarray:
    """Make an empty emust array.

    Parameters
    ----------
    n_polar
        The number of polar angles.

    Returns
    -------
    Array of zeros

    """
    return np.zeros(n_polar)


def make_empty_rho_accurate(n_polar: int, n_azimuth: int) -> np.ndarray:
    """Make an empty rho accurate array.

    Parameters
    ----------
    n_polar
        The number of polar angles.
    n_azimuth
        The number of azimuth angles.

    Returns
    -------
    Array of zeros

    """
    return np.zeros((n_polar, n_azimuth))


def make_empty_rhoq(n_streams: int) -> np.ndarray:
    """Make an empty rhoq array.

    Parameters
    ----------
    n_streams
        The number of streams.

    Returns
    -------
    Array of zeros

    """
    return np.zeros((int(0.5 * n_streams), int(0.5 * n_streams + 1), n_streams))


def make_empty_rhou(n_streams: int) -> np.ndarray:
    """Make an empty rhou array.

    Parameters
    ----------
    n_streams
        The number of streams

    Returns
    -------
    Array of zeros

    """
    return np.zeros((n_streams, int(0.5 * n_streams + 1), n_streams))


def _make_disobrdf_arg(user_angles: bool, only_fluxes: bool, n_polar: int,
                       n_azimuth: int, n_streams: int, mu: float,
                       mu0: float, phi: float, phi0: float,
                       beam_flux: float, albedo: float,
                       phase_function_number: int,
                       brdf_arg: np.ndarray, n_mug: int):
    bemst = make_empty_bemst(n_streams)
    _emust = make_empty_emust(n_polar)
    _rho_accurate = make_empty_rho_accurate(n_polar, n_azimuth)
    _rhou = make_empty_rhou(n_streams)
    _rhoq = make_empty_rhoq(n_streams)

    return [user_angles, mu, beam_flux, mu0, False, albedo,
                    only_fluxes, _rhoq, _rhou, _emust,
                    bemst, False, phi, phi0, _rho_accurate,
                    phase_function_number, brdf_arg, n_mug,
                    n_streams, n_polar,
                    n_azimuth]


def make_hapke_surface(user_angles: bool, only_fluxes: bool, n_polar: int,
                       n_azimuth: int, n_streams: int, mu: float, mu0: float,
                       phi: float, phi0: float, beam_flux: float, n_mug: int,
                       b0: float, h: float, w: float) -> list:
    """Make a basic Hapke surface.

    Parameters
    ----------
    user_angles
        True if this should be returned at user angles; False otherwise.
    only_fluxes
        True of only fluxes should be returned; False otherwise. Note that I
        don't know why this would be relevant to this function, but DISORT
        requires it.
    n_polar
        The number of polar angles.
    n_azimuth
        The number of azimuth angles.
    n_streams
        The number of streams.
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

    Returns
    -------
    list
        The arguments to apply to disobrdf.

    Notes
    -----
    The output of this function can be easily piped into disort.disobrdf with
    ``disort.disobrdf(*make_hapke_surface(<args>))``.

    """
    brdf_arg = np.array([b0, h, w, 0, 0, 0])
    return _make_disobrdf_arg(user_angles, only_fluxes, n_polar, n_azimuth,
                          n_streams, mu, mu0, phi, phi0, beam_flux,
                          0, 1, brdf_arg, n_mug)


def make_hapkeHG2_surface(user_angles: bool, only_fluxes: bool, n_polar: int,
                          n_azimuth: int, n_streams: int, mu: float, mu0: float,
                          phi: float, phi0: float, beam_flux: float, n_mug: int,
                          b0: float, h: float, w: float, asym: float, frac: float) -> list:
    """Make a 2-lobed Henyey-Greenstein Hapke surface.

    Parameters
    ----------
    user_angles
        True if this should be returned at user angles; False otherwise.
    only_fluxes
        True of only fluxes should be returned; False otherwise. Note that I
        don't know why this would be relevant to this function, but DISORT
        requires it.
    n_polar
        The number of polar angles.
    n_azimuth
        The number of azimuth angles.
    n_streams
        The number of streams.
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

    Returns
    -------
    list
        The arguments to apply to disobrdf.

    Notes
    -----
    The output of this function can be easily piped into disort.disobrdf with
    ``disort.disobrdf(*make_hapkeHG2_surface(<args>))``.

    """
    brdf_arg = np.array([b0, h, w, asym, frac, 0])
    return _make_disobrdf_arg(user_angles, only_fluxes, n_polar, n_azimuth,
                          n_streams, mu, mu0, phi, phi0, beam_flux,
                          0, 5, brdf_arg, n_mug)


def make_hapkeHG2roughness_surface(user_angles: bool, only_fluxes: bool, n_polar: int,
                       n_azimuth: int, n_streams: int, mu: float, mu0: float,
                       phi: float, phi0: float, beam_flux: float, n_mug: int,
                       b0: float, h: float, w: float, asym: float, frac: float,
                       roughness: float) -> list:
    """Make a 2-lobed Henyey-Greenstein Hapke surface with roughness parameter.

    Parameters
    ----------
    user_angles
        True if this should be returned at user angles; False otherwise.
    only_fluxes
        True of only fluxes should be returned; False otherwise. Note that I
        don't know why this would be relevant to this function, but DISORT
        requires it.
    n_polar
        The number of polar angles.
    n_azimuth
        The number of azimuth angles.
    n_streams
        The number of streams.
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

    Returns
    -------
    list
        The arguments to apply to disobrdf.

    Notes
    -----
    The output of this function can be easily piped into disort.disobrdf with
    ``disort.disobrdf(*make_hapkeHG2roughness_surface(<args>))``.

    """
    brdf_arg = np.array([b0, h, w, asym, frac, roughness])
    return _make_disobrdf_arg(user_angles, only_fluxes, n_polar, n_azimuth,
                          n_streams, mu, mu0, phi, phi0, beam_flux,
                          0, 6, brdf_arg, n_mug)
