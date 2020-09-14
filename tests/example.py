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

# Define the absolute paths to some files I'll need
atmfile = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/mars_atm.npy'
dustfile = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/dust.npy'
polyfile = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/legendre_coeff_dust.npy'

# Make an atmosphere and add it to the model
atm = Atmosphere(atmfile)
model = ModelAtmosphere(atm)

# Now the atmosphere doesn't have anything in it... create a column of dust
# First, make a phase function
wavelengths = np.array([1, 2, 8, 9.3])    # Suppose you observe at these 4 wavelengths
phase = EmpiricalPhaseFunction(polyfile, 128)  # 128 moments
# Then pass that phase function to Aerosol. Aerosol only keeps track of the aerosol's properties (c_ext, c_sca, g, etc.)
dust = Aerosol(dustfile, phase, wavelengths, 9.3)   # 9.3 is the reference wavelength
# Next, make a column of aerosols
dust_column = Column(dust, 10, 0.5, 1)   # 10=scale height, 0.5=Conrath nu, 1 = column optical depth

# Once I make columns this way, I can add them to the model
model.add_rayleigh_co2_optical_depths(wavelengths)
model.add_column('')

# After I've added Rayleigh scattering and all the columns I want, it can get the "big 3" arrays
optical_depths = model.calculate_column_optical_depths()
ssa = model.calculate_single_scattering_albedos()
polynomial_moments = model.calculate_polynomial_moments()
#print(optical_depths.shape)        # n_layers x n_wavelengths
#print(ssa.shape)                   # n_layers x n_wavelengths
#print(polynomial_moments.shape)    # n_moments x n_layers x n_wavelengths
# Or I can access anything that went into the model
temperatures = model.atmosphere.temperature_boundaries

raise SystemExit(12)

# DISORT cannot natively handle the wavelength dimension, so reduce those here for testing
optical_depths = optical_depths[:, 0]
ssa = ssa[:, 0]
polynomial_moments = polynomial_moments[:, :, 0]

# Make up a fake observation
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
phi = np.array([obs.calculate_phi()])
low_wavenumber = obs.calculate_low_wavenumber()
high_wavenumber = obs.calculate_high_wavenumber()
phi0 = obs.phi0
umu0 = obs.calculate_mu0()
umu = np.array([obs.calculate_mu()])
fbeam = np.pi

# Just define a bunch of variables that should go into a conrol class
usrtau = False
usrang = True
fisot = 0
onlyfl = True
n_umu = 1
n_phi = 1
ibcnd = 0
prnt = np.array([True, False, False, False, True])
accur = 0
plank = True
lamber = True
n_streams = 8
surface_temp = 270
top_temp = 0
top_emissivity = 1
n_user_levels = 20
n_cmu = 4   # I'm really unsure about this variable...
do_pseudo_sphere = False
earth_radius = 6371
deltamplus = False
header = 'myHeader'
utau = np.zeros(n_user_levels)

# Get albedo (it probably shouldn't go here though...)
albedo_map = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/albedo_map.npy'
albedo = Albedo(albedo_map, obs.latitude, obs.longitude).interpolate_albedo()

# Make the output variables (note: I learned later that these aren't all output variables...)
n_boundaries = 15
output = Output(n_user_levels, n_phi, n_umu, n_boundaries, n_cmu)
h_lyr = output.make_h_lyr()
rhoq = output.make_rhoq()
rhou = output.make_rhou()
rho_accurate = output.make_rho_accurate()
bemst = output.make_bemst()
emust = output.make_emust()
direct_beam_flux = output.make_direct_beam_flux()
diffuse_down_flux = output.make_diffuse_down_flux()
diffuse_up_flux = output.make_diffuse_up_flux()
flux_divergence = output.make_flux_divergence()
mean_intensity = output.make_mean_intensity()
intensity = output.make_intensity()
albedo_medium = output.make_albedo_medium()
transmissivity_medium = output.make_transmissivity_medium()

# Run the model
dfdt, uavg, uu = disort.disort(usrang, usrtau, ibcnd, onlyfl, prnt, plank, lamber, deltamplus, do_pseudo_sphere, optical_depths,
                               ssa, polynomial_moments, temperatures, low_wavenumber, high_wavenumber, utau, umu0, phi0, umu, phi, fbeam, fisot, albedo,
                               surface_temp, top_temp, top_emissivity, earth_radius, h_lyr, rhoq, rhou, rho_accurate, bemst, emust, accur,
                               header, direct_beam_flux, diffuse_down_flux, diffuse_up_flux, flux_divergence, mean_intensity,
                               intensity, albedo_medium, transmissivity_medium)

print(uu)
