import numpy as np
from astropy.io import fits
from pyRT_DISORT.observation import Angles, Spectral
from pyRT_DISORT.eos import Hydrostatic
from pyRT_DISORT.rayleigh import RayleighCO2
from pyRT_DISORT.vertical_profile import Conrath
from pyRT_DISORT.forward_scattering import NearestNeighborSingleScatteringAlbedo
from pyRT_DISORT.optical_depth import OpticalDepth

# observation module
dummy_angles = np.outer(np.linspace(5, 10, num=15), np.linspace(5, 8, num=20))
angles = Angles(dummy_angles, dummy_angles, dummy_angles)

dummy_wavelengths = np.array([1, 2, 3, 4, 5])
pixel_wavelengths = np.broadcast_to(dummy_wavelengths, (20, 15, 5)).T
width = 0.05

spectral = Spectral(pixel_wavelengths - width, pixel_wavelengths + width)

# eos module
altitude_grid = np.linspace(100, 0, num=51)
pressure_profile = 500 * np.exp(-altitude_grid / 10)
temperature_profile = np.linspace(150, 250, num=51)
mass = 7.3 * 10**-26
gravity = 3.7

z_grid = np.linspace(100, 0, num=15)
hydro = Hydrostatic(altitude_grid, pressure_profile, temperature_profile, z_grid, mass, gravity)

# rayleigh module
rco2 = RayleighCO2(pixel_wavelengths[:, 0, 0], hydro.column_density)

# vertical_profile module
z_midpoint = ((z_grid[:-1] + z_grid[1:]) / 2)[:, np.newaxis]
q0 = np.array([1])
H = np.array([10])
nu = np.array([0.01])

conrath = Conrath(z_midpoint, q0, H, nu)

# forward_scattering module
f = '/home/kyle/repos/pyRT_DISORT/tests/aux/dust_properties.fits'
hdul = fits.open(f)
cext = hdul['primary'].data[:, :, 0]
csca = hdul['primary'].data[:, :, 1]
wavs = hdul['wavelengths'].data
psizes = hdul['particle_sizes'].data

pgrad = np.linspace(1, 1.5, num=14)

nnssa = NearestNeighborSingleScatteringAlbedo(csca, cext, psizes, wavs, pgrad, spectral.short_wavelength[:, 0, 0])

# optical_depth module
od = OpticalDepth(np.squeeze(conrath.profile), hydro.column_density, nnssa.make_extinction_grid(9.3), 1)

print(np.sum(od.total, axis=0))