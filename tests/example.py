import disort

import numpy as np
from preprocessing.model.model_atmosphere import ModelAtmosphere
from preprocessing.model.atmosphere import Atmosphere
from preprocessing.model.aerosol import Aerosol
from preprocessing.model.aerosol_column import Column
from preprocessing.observation import Observation
from preprocessing.controller.output import Output
from preprocessing.model.phase_function import EmpiricalPhaseFunction
from preprocessing.controller.size import Size
from preprocessing.controller.unsure import Unsure
from preprocessing.controller.control import Control
from preprocessing.model.boundary_conditions import BoundaryConditions
from preprocessing.utilities.rayleigh_co2 import calculate_rayleigh_co2_optical_depths

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the model atmosphere
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the absolute paths to some files I'll need
atmfile = '/home/kyle/repos/pyRT_DISORT/preprocessing/planets/mars/aux/disortMultiPseudoMatch.npy'
dustfile = '/home/kyle/repos/pyRT_DISORT/preprocessing/planets/mars/aux/dust.npy'
icefile = '/home/kyle/repos/pyRT_DISORT/preprocessing/planets/mars/aux/ice.npy'
dust_polyfile = '/home/kyle/repos/pyRT_DISORT/preprocessing/planets/mars/aux/legendre_coeff_dust.npy'
ice_polyfile = '/home/kyle/repos/pyRT_DISORT/preprocessing/planets/mars/aux/legendre_coeff_h2o_ice.npy'

wavelengths = np.array([9.3, 11])    # Suppose you observe at these 2 wavelengths
n_moments = 128

# Make an atmosphere and add it to the model
atm = Atmosphere(atmfile)
model = ModelAtmosphere(atm, len(wavelengths), n_moments)

# Now the atmosphere doesn't have anything in it... create a column of dust
phase = EmpiricalPhaseFunction(dust_polyfile)
dust = Aerosol(dustfile, phase, wavelengths, 9.3)   # 9.3 is the reference wavelength
dust_column = Column(dust, 10, 0.5, 1)   # 10=scale height, 0.5=Conrath nu, 1 = column optical depth

# Make some ice
ice_phase = EmpiricalPhaseFunction(ice_polyfile)
ice = Aerosol(icefile, ice_phase, wavelengths, 12.1)
ice_column = Column(ice, 10, 0.5, 0.5)

# Once I make columns this way, I can add them to the model
model.add_column(dust_column)
#model.add_column(ice_column)

# Add in Rayleigh stuff
co2_OD = calculate_rayleigh_co2_optical_depths(wavelengths, atm.column_density_layers)
model.add_rayleigh_constituent(co2_OD)

# Now that the everything's in the model, compute everything
model.compute_model()

# The model now knows everything, so these are examples but unnecessary---except for removing the wavelength dimension
optical_depths = model.total_optical_depths[:, 0]
ssa = model.total_single_scattering_albedo[:, 0]
polynomial_moments = model.polynomial_moments[:, :, 0]

# After I've added Rayleigh scattering and all the columns I want, it can get the "big 3" arrays
#model.calculate_rayleigh_optical_depths()
#optical_depths = model.calculate_column_optical_depths()
#ssa = model.calculate_single_scattering_albedos()
#polynomial_moments = model.calculate_polynomial_moments()

# Or I can access anything that went into the model
temperatures = model.atmosphere.temperature_boundaries

# DISORT cannot natively handle the wavelength dimension, so reduce those here for testing
#optical_depths = optical_depths[:, 0]
#ssa = ssa[:, 0]
#polynomial_moments = polynomial_moments[:, :, 0]
print(optical_depths)
print(ssa)

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

n_layers = model.atmosphere.n_layers
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
