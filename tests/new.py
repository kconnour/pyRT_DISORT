# observation module
import numpy as np
from pyRT_DISORT.observation import Angles, Spectral

dummy_angles = np.outer(np.linspace(5, 10, num=15), np.linspace(5, 8, num=20))

angles = Angles(dummy_angles, dummy_angles, dummy_angles)

incidence = angles.incidence
emission = angles.emission
phase = angles.phase
mu = angles.mu
mu0 = angles.mu0
phi = angles.phi
phi0 = angles.phi0

UMU = mu[0, 0]
UMU0 = mu0[0, 0]
PHI = phi[0, 0]
PHI0 = phi0[0, 0]

pixel_wavelengths = np.array([1, 2, 3, 4, 5])
width = 0.05

spectral = Spectral(pixel_wavelengths - width, pixel_wavelengths + width)

short_wavelength = spectral.short_wavelength
long_wavelength = spectral.long_wavelength
WVNMHI = spectral.high_wavenumber
WVNMLO = spectral.low_wavenumber

# eos module
atmfile = np.load('/home/kyle/repos/pyRT_DISORT/tests/aux/marsatm.npy')
#altitude_grid = np.flip(atmfile[:, 0])
#pressure_profile = np.flip(atmfile[:, 1])
#temperature_profile = np.flip(atmfile[:, 2])

altitude_grid = np.linspace(100, 0, num=51)
pressure_profile = 500 * np.exp(-altitude_grid / 10)
temperature_profile = np.linspace(150, 250, num=51)
mass = 7.3 * 10**-26
gravity = 3.7

z_grid = np.linspace(100, 0, num=15)

from pyRT_DISORT.eos import Hydrostatic

hydro = Hydrostatic(altitude_grid, pressure_profile, temperature_profile,
                    z_grid, mass, gravity)

altitude = hydro.altitude
pressure = hydro.pressure
TEMPER = hydro.temperature
number_density = hydro.number_density
column_density = hydro.column_density
n_layers = hydro.n_layers
H_LYR = hydro.scale_height

# rayleigh module
from pyRT_DISORT.rayleigh import RayleighCO2

rco2 = RayleighCO2(pixel_wavelengths, hydro.column_density)

rayleigh_od = rco2.optical_depth
rayleigh_ssa = rco2.single_scattering_albedo
rayleigh_pf = rco2.phase_function

print(np.sum(rayleigh_od, axis=0))

# aerosol module
from pyRT_DISORT.aerosol import Conrath

z_midpoint = ((z_grid[:-1] + z_grid[1:]) / 2)
q0 = 1
H = 10
nu = 0.01

conrath = Conrath(z_midpoint, q0, H, nu)
dust_profile = conrath.profile

from astropy.io import fits
f = '/home/kyle/repos/pyRT_DISORT/tests/aux/dust_properties.fits'
hdul = fits.open(f)
cext = hdul['primary'].data[:, :, 0]
csca = hdul['primary'].data[:, :, 1]
wavs = hdul['wavelengths'].data
psizes = hdul['particle_sizes'].data

pgrad = np.linspace(1.5, 1.5, num=14)
wave_ref = 9.3

from pyRT_DISORT.aerosol import ForwardScattering

fs = ForwardScattering(csca, cext, psizes, wavs, pgrad, pixel_wavelengths, wave_ref)
fs.make_nn_properties()

nn_ext_cs = fs.extinction_cross_section
nn_sca_cs = fs.scattering_cross_section
dust_ssa = fs.single_scattering_albedo
dust_ext = fs.extinction

from pyRT_DISORT.aerosol import OpticalDepth

od = OpticalDepth(dust_profile, hydro.column_density, fs.extinction, 1)
dust_od = od.total

from pyRT_DISORT.aerosol import TabularLegendreCoefficients

dust_phsfn_file = fits.open('/home/kyle/repos/pyRT_DISORT/tests/aux/dust_phase_function.fits')
coeff = dust_phsfn_file['primary'].data
pf_wavs = dust_phsfn_file['wavelengths'].data
pf_psizes = dust_phsfn_file['particle_sizes'].data

pf = TabularLegendreCoefficients(coeff, pf_psizes, pf_wavs, pgrad, pixel_wavelengths)
pf.make_nn_phase_function()

dust_pf = pf.phase_function

# the atmosphere module
from pyRT_DISORT.atmosphere import ModelAtmosphere

model = ModelAtmosphere()

rayleigh_info = (rayleigh_od, rayleigh_ssa, rayleigh_pf)
dust_info = (dust_od, dust_ssa, dust_pf)

model.add_constituent(rayleigh_info)
model.add_constituent(dust_info)

DTAUC = model.optical_depth
SSALB = model.single_scattering_albedo
PMOM = model.legendre_moments

# The controller module
from pyRT_DISORT.controller import ComputationalParameters, ModelBehavior

cp = ComputationalParameters(hydro.n_layers, model.legendre_moments.shape[0],
                             16, 1, 1, 80)

mb = ModelBehavior()
ACCUR = mb.accuracy
DELTAMPLUS = mb.delta_m_plus
DO_PSEUDO_SPHERE = mb.do_pseudo_sphere
HEADER = mb.header
PRNT = mb.print_variables
EARTH_RADIUS = mb.radius

# the radiation module
from pyRT_DISORT.radiation import IncidentFlux, ThermalEmission

flux = IncidentFlux()
FBEAM = flux.beam_flux
FISOT = flux.isotropic_flux

te = ThermalEmission()
PLANK = te.thermal_emission
BTEMP = te.bottom_temperature
TTEMP = te.top_temperature
TEMIS = te.top_emissivity

# the output module
from pyRT_DISORT.output import OutputArrays, OutputBehavior, UserLevel

oa = OutputArrays(cp.n_polar, cp.n_user_levels, cp.n_azimuth)
ALBMED = oa.albedo_medium
FLUP = oa.diffuse_up_flux
RFLDN = oa.diffuse_down_flux
RFLDIR = oa.direct_beam_flux
DFDT = oa.flux_divergence
UU = oa.intensity
UAVG = oa.mean_intensity
TRNMED = oa.transmissivity_medium

ob = OutputBehavior()
IBCND = ob.incidence_beam_conditions
ONLYFL = ob.only_fluxes
USRANG = ob.user_angles
USRTAU = ob.user_optical_depths

ulv = UserLevel(cp.n_user_levels)
UTAU = ulv.optical_depth_output

# The surface module
from pyRT_DISORT.surface import Lambertian

lamb = Lambertian(0.1, cp.n_streams, cp.n_polar, cp.n_azimuth, ob.user_angles,
                  ob.only_fluxes)
ALBEDO = lamb.albedo
LAMBER = lamb.lambertian
RHOU = lamb.rhou
RHOQ = lamb.rhoq
BEMST = lamb.bemst
EMUST = lamb.emust
RHO_ACCURATE = lamb.rho_accurate

# Run the model
import disort

test_run = np.zeros(pixel_wavelengths.shape)

for ind, w in enumerate(pixel_wavelengths):
    rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed = \
        disort.disort(USRANG, USRTAU, IBCND, ONLYFL, PRNT, PLANK, LAMBER,
                      DELTAMPLUS, DO_PSEUDO_SPHERE, DTAUC[:, ind], SSALB[:, ind],
                      PMOM[:, :, ind], TEMPER, WVNMLO, WVNMHI,
                      UTAU, UMU0, PHI0, UMU, PHI, FBEAM, FISOT,
                      ALBEDO, BTEMP, TTEMP, TEMIS, EARTH_RADIUS, H_LYR, RHOQ, RHOU,
                      RHO_ACCURATE, BEMST, EMUST, ACCUR, HEADER, RFLDIR,
                      RFLDN, FLUP, DFDT, UAVG, UU, ALBMED, TRNMED)

    test_run[ind] = uu[0, 0, 0]

print(test_run)
