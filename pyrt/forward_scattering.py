import numpy as np

from pyrt.grid import regrid


def extinction_ratio(extinction_cross_section, particle_size_grid, wavelength_grid, wavelength_reference: float) -> np.ndarray:
    """Make a grid of extinction cross section ratios.

    This is the extinction cross section at the input wavelengths divided by
    the extinction cross section at the reference wavelength.

    Parameters
    ----------
    extinction_cross_section
    particle_size_grid
    wavelength_grid
    wavelength_reference

    Returns
    -------

    """
    cext_slice = np.squeeze(regrid(extinction_cross_section, particle_size_grid, wavelength_grid, particle_size_grid, wavelength_reference))
    return (extinction_cross_section.T / cext_slice).T


def optical_depth(q_prof, column_density, extinction_ratio, column_integrated_od):
    """Make the optical depth in each layer.

    Parameters
    ----------
    q_prof
    column_density
    extinction_ratio
    column_integrated_od

    Returns
    -------

    """
    normalization = np.sum(q_prof * column_density)
    profile = q_prof * column_density * column_integrated_od / normalization
    return (profile * extinction_ratio.T).T


if __name__ == '__main__':
    from pyrt import column_density as cd
    from pyrt import scale_height
    from pyrt import conrath

    w = np.array([1, 2, 3, 4, 5])
    altitude_grid = np.linspace(100, 0, num=15)
    pressure_profile = 500 * np.exp(-altitude_grid / 10)
    temperature_profile = np.linspace(150, 250, num=15)
    mass = 7.3 * 10 ** -26
    gravity = 3.7

    column_density = cd(pressure_profile, temperature_profile, altitude_grid)
    H_LYR = 10

    z_midpoint = (altitude_grid[:-1] + altitude_grid[1:]) / 2
    q0 = 1
    nu = 0.01

    dust_profile = conrath(z_midpoint, q0, H_LYR, nu)

    particle_size_grid = np.linspace(0.5, 10, num=50)
    wavelength_grid = np.linspace(0.2, 50, num=20)
    extinction_cross_section = np.ones((50, 20))
    scattering_cross_section = np.ones((50, 20)) * 0.5

    particle_size_gradient = np.linspace(1, 1.5, num=14)

    ext = extinction_ratio(extinction_cross_section, particle_size_grid, wavelength_grid, 9.3)
    ext = regrid(ext, particle_size_grid, wavelength_grid, particle_size_gradient, w)
    od = optical_depth(dust_profile, column_density, ext, 1)
    print(od.shape)
    print(np.sum(od, axis=0))

    ssa = regrid(scattering_cross_section / extinction_cross_section, particle_size_grid, wavelength_grid, particle_size_gradient, w)
    print(ssa.shape)

    dust_pmom = np.ones((128, 50, 20))
    dust_legendre = regrid(dust_pmom, particle_size_grid, wavelength_grid, particle_size_gradient, w)

    print(dust_legendre.shape)