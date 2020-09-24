import disort

import numpy as np
from preprocessing.model.model_atmosphere import ModelAtmosphere
from preprocessing.model.aerosol import Aerosol
from preprocessing.model.atmosphere import Layers
from preprocessing.model.aerosol_column import Column
from preprocessing.observation import Observation
from preprocessing.controller.output import Output
from preprocessing.model.phase_function import EmpiricalPhaseFunction, NearestNeighborPhaseFunction
from preprocessing.controller.size import Size
from preprocessing.controller.unsure import Unsure
from preprocessing.controller.control import Control
from preprocessing.model.boundary_conditions import BoundaryConditions
from preprocessing.model.rayleigh import RayleighCo2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the model atmosphere
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define some files I'll need
phase = '/home/kyle/repos/pyRT_DISORT/preprocessing/planets/mars/aux/phase_functions.npy'
phase_radii = '/home/kyle/repos/pyRT_DISORT/preprocessing/planets/mars/aux/phase_function_radii.npy'
phase_wavs = '/home/kyle/repos/pyRT_DISORT/preprocessing/planets/mars/aux/phase_function_wavelengths.npy'
dustfile = '/home/kyle/repos/pyRT_DISORT/preprocessing/planets/mars/aux/dust.npy'
atm = '/home/kyle/repos/pyRT_DISORT/preprocessing/planets/mars/aux/mars_atm.npy'

# I make an aerosol that was observed at these wavelengths
wavs = np.array([1, 9.3])
dust = Aerosol(dustfile, wavs, 9.3)     # wavelength reference

# Make a column of that aerosol
lay = Layers(atm)
dust_column = Column(dust, lay, 10, 0.5, np.array([1, 1.2, 1.4]), np.array([0.3, 0.5, 0.2]))  # column r, column OD

# Make the phase function
e = EmpiricalPhaseFunction(phase, phase_radii, phase_wavs)
n_moments = 65
nn = NearestNeighborPhaseFunction(e, dust_column, n_moments)    # 128 moments

# Make Rayleigh stuff
rco2 = RayleighCo2(wavs, lay, n_moments)
rayleigh_info = (rco2.hyperspectral_optical_depths, rco2.hyperspectral_optical_depths, rco2.hyperspectral_layered_phase_function)

# Make the model
model = ModelAtmosphere()
dust_info = (dust_column.hyperspectral_total_optical_depths, dust_column.hyperspectral_scattering_optical_depths,
             nn.layered_hyperspectral_nearest_neighbor_phase_functions)

# Add dust and Rayleigh scattering to the model
model.add_constituent(dust_info)
model.add_constituent(rayleigh_info)

# Once everything is in the model, compute the model. Then, slice off the wavelength dimension
model.compute_model()
optical_depths = model.hyperspectral_total_optical_depths[:, 1]
ssa = model.hyperspectral_total_single_scattering_albedos[:, 1]
polynomial_moments = model.hyperspectral_legendre_moments[:, :, 1]

#print(np.amax(optical_depths))
#print(np.amax(ssa))
#print(np.amax(polynomial_moments))
#raise SystemExit(2)

# Get a miscellaneous variable that I'll need later
temperatures = lay.temperature_boundaries

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make a fake observation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
short_wav = 1    # microns
long_wav = 1.1
sza = 50
emission_angle = 40
phase_angle = 20
latitude = 10
longitude = 30
altitude_map = '/home/kyle/repos/pyRT_DISORT/preprocessing/planets/mars/aux/altitude_map.npy'
solar_spec = '/home/kyle/repos/pyRT_DISORT/preprocessing/aux/solar_spectrum.npy'
obs = Observation(short_wav, long_wav, sza, emission_angle, phase_angle, latitude, longitude, altitude_map, solar_spec)
phi = np.array([obs.phi])
low_wavenumber = obs.low_wavenumber
high_wavenumber = obs.high_wavenumber
phi0 = obs.phi0
umu0 = obs.mu0
umu = np.array([obs.mu])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the size of the arrays
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

n_layers = lay.n_layers
n_streams = 16
n_umu = 1
n_phi = 1
n_user_levels = 81
size = Size(n_layers, n_moments, n_streams, n_umu, n_phi, n_user_levels)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the control class
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
control = Control(print_variables=np.array([True, True, True, True, True]))
usrtau = control.user_optical_depths
usrang = control.user_angles
onlyfl = control.only_fluxes
accur = control.accuracy
prnt = control.print_variables
header = control.header
do_pseudo_sphere = control.do_pseudo_sphere
planet_radius = control.radius
deltamplus = control.delta_m_plus

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the boundary conditions class
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
boundary = BoundaryConditions(bottom_temperature=270, top_emissivity=1)
ibcnd = boundary.ibcnd
fbeam = boundary.beam_flux
fisot = boundary.fisot
lamber = boundary.lambertian
plank = boundary.plank
surface_temp = boundary.bottom_temperature
top_temp = boundary.top_temperature
top_emissivity = boundary.top_emissivity

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the arrays I'm unsure about (for now)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
uns = Unsure(size)
h_lyr = uns.make_h_lyr()
rhoq = uns.make_rhoq()
rhou = uns.make_rhou()
rho_accurate = uns.make_rho_accurate()
bemst = uns.make_bemst()
emust = uns.make_emust()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the output arrays
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
n_boundaries = 15
output = Output(size)
direct_beam_flux = output.make_direct_beam_flux()
diffuse_down_flux = output.make_diffuse_down_flux()
diffuse_up_flux = output.make_diffuse_up_flux()
flux_divergence = output.make_flux_divergence()
mean_intensity = output.make_mean_intensity()
intensity = output.make_intensity()
albedo_medium = output.make_albedo_medium()
transmissivity_medium = output.make_transmissivity_medium()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Unsorted crap
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
utau = np.zeros(n_user_levels)
# Get albedo (it probably shouldn't go here though...)
albedo_map = '/home/kyle/repos/pyRT_DISORT/preprocessing/planets/mars/aux/albedo_map.npy'
albedo = 0.5  #Albedo(albedo_map, obs.latitude, obs.longitude).interpolate_albedo()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run the model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed = disort.disort(usrang, usrtau, ibcnd, onlyfl, prnt, plank, lamber,
                                                                    deltamplus, do_pseudo_sphere, optical_depths,
                               ssa, polynomial_moments, temperatures, low_wavenumber, high_wavenumber, utau, umu0, phi0,
                                                                    umu, phi, fbeam, fisot, albedo,
                               surface_temp, top_temp, top_emissivity, planet_radius, h_lyr, rhoq, rhou, rho_accurate,
                                                                    bemst, emust, accur,
                               header, direct_beam_flux, diffuse_down_flux, diffuse_up_flux, flux_divergence,
                                                                    mean_intensity,
                               intensity, albedo_medium, transmissivity_medium)

print(uu)   # shape: (1, 81, 1)
