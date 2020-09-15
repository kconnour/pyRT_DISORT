import disort

import numpy as np
from generic.model_atmosphere import ModelAtmosphere
from generic.atmosphere import Atmosphere
from generic.aerosol import Aerosol
from generic.aerosol_column import Column
from generic.observation import Observation
from generic.output import Output
from planets.mars.map import Albedo
from generic.phase_function import EmpiricalPhaseFunction
from generic.size import Size
from generic.unsure import Unsure
from generic.control import Control
from generic.boundary_conditions import BoundaryConditions

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the model atmsophere
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the absolute paths to some files I'll need
atmfile = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/mars_atm.npy'
dustfile = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/dust.npy'
polyfile = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/legendre_coeff_dust.npy'

# Make an atmosphere and add it to the model
atm = Atmosphere(atmfile)
model = ModelAtmosphere(atm)

# Now the atmosphere doesn't have anything in it... create a column of dust
# First, make a phase function
wavelengths = np.array([1, 2])    # Suppose you observe at these 2 wavelengths
phase = EmpiricalPhaseFunction(polyfile, 128)  # 128 moments
# Then pass that phase function to Aerosol. Aerosol only keeps track of the aerosol's properties (c_ext, c_sca, g, etc.)
dust = Aerosol(dustfile, phase, wavelengths, 9.3)   # 9.3 is the reference wavelength
# Next, make a column of aerosols
dust_column = Column(dust, 10, 0.5, 1)   # 10=scale height, 0.5=Conrath nu, 1 = column optical depth

# Once I make columns this way, I can add them to the model
model.add_rayleigh_co2_optical_depths(wavelengths)
model.add_column(dust_column)

# After I've added Rayleigh scattering and all the columns I want, it can get the "big 3" arrays
optical_depths = model.calculate_column_optical_depths()
ssa = model.calculate_single_scattering_albedos()
polynomial_moments = model.calculate_polynomial_moments()
#print(optical_depths.shape)        # n_layers x n_wavelengths
#print(ssa.shape)                   # n_layers x n_wavelengths
#print(polynomial_moments.shape)    # n_moments x n_layers x n_wavelengths
# Or I can access anything that went into the model
temperatures = model.atmosphere.temperature_boundaries

# DISORT cannot natively handle the wavelength dimension, so reduce those here for testing
optical_depths = optical_depths[:, 0]
ssa = ssa[:, 0]
polynomial_moments = polynomial_moments[:, :, 0]

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
altitude_map = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/altitude_map.npy'
solar_spec = '/home/kyle/repos/pyRT_DISORT/aux/solar_spectrum.npy'
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

n_layers = model.atmosphere.n_layers
n_moments = 128
n_cmu = 4   # I'm unsure about this variable...
n_umu = 1
n_phi = 1
n_user_levels = 20
size = Size(n_layers, n_moments, n_cmu, n_umu, n_phi, n_user_levels)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the control class
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
control = Control()
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
albedo_map = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/albedo_map.npy'
albedo = Albedo(albedo_map, obs.latitude, obs.longitude).interpolate_albedo()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run the model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dfdt, uavg, uu = disort.disort(usrang, usrtau, ibcnd, onlyfl, prnt, plank, lamber, deltamplus, do_pseudo_sphere, optical_depths,
                               ssa, polynomial_moments, temperatures, low_wavenumber, high_wavenumber, utau, umu0, phi0, umu, phi, fbeam, fisot, albedo,
                               surface_temp, top_temp, top_emissivity, planet_radius, h_lyr, rhoq, rhou, rho_accurate, bemst, emust, accur,
                               header, direct_beam_flux, diffuse_down_flux, diffuse_up_flux, flux_divergence, mean_intensity,
                               intensity, albedo_medium, transmissivity_medium)

print(uu)
