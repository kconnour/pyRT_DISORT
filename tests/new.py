import numpy as np
from astropy.io import fits
from pyRT_DISORT.observation import Angles, Spectral
from pyRT_DISORT.eos import Hydrostatic
from pyRT_DISORT.rayleigh import RayleighCO2
from pyRT_DISORT.vertical_profile import Conrath
from pyRT_DISORT.forward_scattering import NearestNeighborSingleScatteringAlbedo
from pyRT_DISORT.optical_depth import OpticalDepth
from pyRT_DISORT.phase_function import RadialSpectralTabularLegendreCoefficients
from pyRT_DISORT.model_atmosphere import ModelAtmosphere
from pyRT_DISORT.controller import ComputationalParameters, ModelBehavior
from pyRT_DISORT.surface import Lambertian

# observation module
dummy_angles = np.outer(np.linspace(5, 10, num=15), np.linspace(5, 8, num=20))
angles = Angles(dummy_angles, dummy_angles, dummy_angles)
mu = angles.mu[0, 0]
mu0 = angles.mu0[0, 0]
phi = angles.phi[0, 0]
phi0 = angles.phi0[0, 0]

dummy_wavelengths = np.array([1, 2, 3, 4, 5])
pixel_wavelengths = np.broadcast_to(dummy_wavelengths, (20, 15, 5)).T
width = 0.05

spectral = Spectral(pixel_wavelengths - width, pixel_wavelengths + width)
low_wavenumber = spectral.low_wavenumber[:, 0, 0]
high_wavenumber = spectral.high_wavenumber[:, 0, 0]

# eos module
altitude_grid = np.linspace(100, 0, num=51)
pressure_profile = 500 * np.exp(-altitude_grid / 10)
temperature_profile = np.linspace(150, 250, num=51)
mass = 7.3 * 10**-26
gravity = 3.7

z_grid = np.linspace(100, 0, num=15)
hydro = Hydrostatic(altitude_grid, pressure_profile, temperature_profile, z_grid, mass, gravity)
temper = hydro.temperature
h_lyr = hydro.scale_height

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

# the phase_function module
dust_phsfn_file = fits.open('/home/kyle/repos/pyRT_DISORT/tests/aux/dust_phase_function.fits')
coeff = dust_phsfn_file['primary'].data
pf_wavs = dust_phsfn_file['wavelengths'].data
pf_psizes = dust_phsfn_file['particle_sizes'].data
pf = RadialSpectralTabularLegendreCoefficients(coeff, pf_psizes, pf_wavs, z_grid, spectral.short_wavelength[:, 0, 0], pgrad)

# the model_atmosphere module
model = ModelAtmosphere()
rayleigh_info = (rco2.scattering_optical_depth, rco2.ssa, rco2.phase_function)
dust_info = (od.total, nnssa.single_scattering_albedo, pf.phase_function)
model.add_constituent(rayleigh_info)
model.add_constituent(dust_info)

dtauc = model.optical_depth[:, 0]
ssalb = model.single_scattering_albedo[:, 0]
pmom = model.legendre_moments[:, :, 0]

# The controller module
cp = ComputationalParameters(hydro.n_layers, model.legendre_moments.shape[0], 16, 1, 1, 80)

mb = ModelBehavior()
accur = mb.accuracy
deltamplus = mb.delta_m_plus
dopseudosphere = mb.do_pseudo_sphere
header = mb.header
prnt = mb.print_variables
radius = mb.radius

# The surface module
lamb = Lambertian(0.1, cp)
albedo = lamb.albedo
lamber = lamb.lambertian
rhou = lamb.rhou
rhoq = lamb.rhoq
bemst = lamb.bemst
emust = lamb.emust
rho_accurate = lamb.rho_accurate

# the radiation module
from pyRT_DISORT.radiation import IncidentFlux, ThermalEmission

flux = IncidentFlux()
fbeam = flux.beam_flux
fisot = flux.isotropic_flux

te = ThermalEmission()
plank = te.thermal_emission
btemp = te.bottom_temperature
ttemp = te.top_temperature
temis = te.top_emissivity

# the output module
from pyRT_DISORT.output import OutputArrays, OutputBehavior, UserLevel

ob = OutputBehavior()
ibcnd = ob.incidence_beam_conditions
onlyfl = ob.only_fluxes
usrang = ob.user_angles
usrtau = ob.user_optical_depths

oa = OutputArrays(cp.n_polar, cp.n_user_levels, cp.n_azimuth)
albmed = oa.albedo_medium
flup = oa.diffuse_up_flux
rfldn = oa.diffuse_down_flux
rfldir = oa.direct_beam_flux
dfdt = oa.flux_divergence
uu = oa.intensity
uavg = oa.mean_intensity
trnmed = oa.transmissivity_medium

ulv = UserLevel(cp.n_user_levels)
utau = ulv.optical_depth_output

# Run the model
import disort

rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed = \
    disort.disort(usrang, usrtau, ibcnd, onlyfl, prnt, plank, lamber,
                  deltamplus, dopseudosphere, dtauc, ssalb,
                  pmom, temper, low_wavenumber,
                  high_wavenumber, utau, mu0, phi0, mu, phi, fbeam, fisot,
                  albedo, btemp, ttemp, temis, radius, h_lyr, rhoq, rhou,
                  rho_accurate, bemst, emust, accur, header, rfldir,
                  rfldn, flup, dfdt, uavg, uu, albmed, trnmed)

print(uu[0, 0, 0])   # shape: (1, 81, 1)
