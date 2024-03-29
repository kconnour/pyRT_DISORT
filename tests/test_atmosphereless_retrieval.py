import numpy as np
import pyrt
import disort


def test_retrieval_without_atmosphere_gives_expected_result():
    """ This test case comes from Mike Wolff

    Returns
    -------
    None
    """

    # Define computational parameters
    n_streams = 32
    n_polar = 1
    n_azimuth = 1

    # Define angles
    solar_zenith_angle = 45
    emission_angle = 55
    phase_angle = 72.25
    azimuth = pyrt.azimuth(solar_zenith_angle, emission_angle, phase_angle)
    mu0 = np.cos(np.radians(solar_zenith_angle))
    mu = np.cos(np.radians(emission_angle))

    # Define an empty atmosphere
    z = np.linspace(100000, 0, num=15)
    optical_depth = np.zeros((z.shape[0] - 1))
    ssa = np.ones((z.shape[0] - 1))
    pmom = np.zeros((128, z.shape[0]-1))
    pmom[0] = 1

    # Define output arrays
    n_user_levels = z.shape[0]
    albedo_medium = pyrt.empty_albedo_medium(n_polar)
    diffuse_up_flux = pyrt.empty_diffuse_up_flux(n_user_levels)
    diffuse_down_flux = pyrt.empty_diffuse_down_flux(n_user_levels)
    direct_beam_flux = pyrt.empty_direct_beam_flux(n_user_levels)
    flux_divergence = pyrt.empty_flux_divergence(n_user_levels)
    intensity = pyrt.empty_intensity(n_polar, n_user_levels, n_azimuth)
    mean_intensity = pyrt.empty_mean_intensity(n_user_levels)
    transmissivity_medium = pyrt.empty_transmissivity_medium(n_polar)

    # Define miscellaneous variables
    user_od_output = np.zeros(n_user_levels)
    temper = np.zeros(n_user_levels)
    h_lyr = np.zeros(n_user_levels)

    # Make the surface
    rhoq, rhou, emust, bemst, rho_accurate = pyrt.make_hapkeHG2roughness_surface(
        True, False, n_polar, n_azimuth, n_streams, mu, mu0,
        azimuth, 0, np.pi, 150, 1, 0.06, 0.7, 0.26, 0.3, 15)

    # Call DISORT
    rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed = \
        disort.disort(True, False, False, False, [False, False, False, False, False],
                      False, False, True, False,
                      optical_depth, ssa, pmom,
                      temper, 1, 1, user_od_output,
                      mu0, 0, mu, azimuth,
                      np.pi, 0, 0.1, 0, 0, 1, 3400000, h_lyr,
                      rhoq, rhou, rho_accurate, bemst, emust,
                      0, '', direct_beam_flux,
                      diffuse_down_flux, diffuse_up_flux, flux_divergence,
                      mean_intensity, intensity, albedo_medium,
                      transmissivity_medium, maxcmu=n_streams, maxulv=n_user_levels, maxmom=127)
    answer = uu[0, 0, 0]

    assert np.isclose(answer, 0.1695, atol=1e-3)
