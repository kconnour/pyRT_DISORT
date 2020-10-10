import disort
import os

import numpy as np
from pyRT_DISORT.preprocessing.model.model_atmosphere import ModelAtmosphere
from pyRT_DISORT.preprocessing.model.aerosol import Aerosol
from pyRT_DISORT.preprocessing.model.atmosphere import Layers
from pyRT_DISORT.preprocessing.model.aerosol_column import Column
from pyRT_DISORT.preprocessing.observation import Observation
from pyRT_DISORT.preprocessing.controller.output import Output
from pyRT_DISORT.preprocessing.model.phase_function import EmpiricalPhaseFunction, NearestNeighborPhaseFunction
from pyRT_DISORT.preprocessing.controller.size import Size
from pyRT_DISORT.preprocessing.controller.unsure import Unsure
from pyRT_DISORT.preprocessing.controller.control import Control
from pyRT_DISORT.preprocessing.model.boundary_conditions import BoundaryConditions
from pyRT_DISORT.preprocessing.model.rayleigh import RayleighCo2
from data.get_data import get_data_path
from pyRT_DISORT.preprocessing.model.surface import Hapke

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the model atmosphere
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define some files I'll need

phase = os.path.join(get_data_path(), 'planets/mars/aux/phase_functions.npy')
phase_radii = os.path.join(get_data_path(), 'planets/mars/aux/phase_function_radii.npy')
phase_wavs = os.path.join(get_data_path(), 'planets/mars/aux/phase_function_wavelengths.npy')
dustfile = os.path.join(get_data_path(), 'planets/mars/aux/dust.npy')
atm = os.path.join(get_data_path(), 'planets/mars/aux/mars_atm.npy')
altitude_map = os.path.join(get_data_path(), 'planets/mars/aux/altitude_map.npy')
solar_spec = os.path.join(get_data_path(), 'aux/solar_spectrum.npy')
albedo_map = os.path.join(get_data_path(), 'planets/mars/aux/albedo_map.npy')

# Make an aerosol that was observed at these wavelengths
wavs = np.array([1, 9.3])
dust = Aerosol(dustfile, wavs, 9.3)     # 9.3 is the wavelength reference

# Make a column of that aerosol
lay = Layers(atm)
# 10 = scale heigh, 0.5 = Conrath nu, then you can input an array of r_effective and an array of the column ODs at those r_effective. I throw an error if the lengths don't match
dust_column = Column(dust, lay, 10, 0.5, np.array([1]), np.array([0.8]))  # here, use r_eff = 1 micron and its column OD = 0.8. Just a test case

# Make the phase function
e = EmpiricalPhaseFunction(phase, phase_radii, phase_wavs)
n_moments = 65
nn = NearestNeighborPhaseFunction(e, dust_column, n_moments) # You can use more moments than you have coefficients for, it'll just add 0s to the end

# Make Rayleigh stuff
rco2 = RayleighCo2(wavs, lay, n_moments)

# Make the model
model = ModelAtmosphere()
dust_info = (dust_column.hyperspectral_total_optical_depths, dust_column.hyperspectral_scattering_optical_depths,
             nn.layered_hyperspectral_nearest_neighbor_phase_functions)
rayleigh_info = (rco2.hyperspectral_optical_depths, rco2.hyperspectral_optical_depths, rco2.hyperspectral_layered_phase_function)

# Add dust and Rayleigh scattering to the model
model.add_constituent(dust_info)
model.add_constituent(rayleigh_info)

# Once everything is in the model, compute the model. Then, slice off the wavelength dimension
model.compute_model()
optical_depths = model.hyperspectral_total_optical_depths[:, 1]
ssa = model.hyperspectral_total_single_scattering_albedos[:, 1]
polynomial_moments = model.hyperspectral_legendre_moments[:, :, 1]

# Test case 0: Only Rayleigh scattering. Just comment out line 51: model.add_constituent(dust_info) and 88: raise SystemExit
# All I'll say is that at many wavelengths UU matched disort_multi to at least 3 decimal places

# Test case 1.1: Rayleigh scattering is negligible (such as at 9.3 microns) so dust should dominate the ODs
#print(rco2.hyperspectral_optical_depths[:, 1])
#print(dust_column.hyperspectral_total_optical_depths[:, 1])
#print(model.hyperspectral_total_optical_depths[:, 1])

# Test case 1.2: Rayleigh scattering is negligible (such as at 9.3 microns) so dust should dominate the SSAs
#print(model.hyperspectral_total_single_scattering_albedos[:, 1])
#print(dust.hyperspectral_single_scattering_albedos[1])

# Test case 2.1: Rayleigh scattering is significant (such as at 150 nm) so dust + Rayleigh should contribute to the ODs
#print(rco2.hyperspectral_optical_depths[:, 0])
#print(dust_column.hyperspectral_total_optical_depths[:, 0])
#print(model.hyperspectral_total_optical_depths[:, 0])

# Test case 2.2: Rayleigh scattering is significant (such as at 150 nm) so the SSA should be somewhat greater than the dust SSA
#print(rco2.hyperspectral_optical_depths[:, 0])     # Rayleigh OD is 100% scattering OD
#print(dust_column.hyperspectral_scattering_optical_depths[:, 0])    # Get the scattering portion of the dust OD
#print(model.hyperspectral_total_single_scattering_albedos[:, 0])    # Where Rayleigh OD dominates, SSA=1
#print(dust.hyperspectral_single_scattering_albedos[0])   # Rayleigh SSA=1 so this number should always be less than the column SSA

# Test case 3: dust + Rayleigh
# Comment out the SystemExit on line 88. My UU[0] = 0.06378016. disort_multi gives 0.0637801662
# I'm running ./disort_multi -dust_conrath 0.5, 10 -dust_phsfn 98 < testInput.txt
# dust_phsfn98.dat contain the 65 moments at reff = 1 micron and wavelength = 9.3 microns
# testInput.txt is: 9.3, 0.5, 10, 30, 50, 40, 20, 0.8, 0, 0
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
n_phi = len(phi)
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
boundary = BoundaryConditions(bottom_temperature=270, top_emissivity=1, lambertian_bottom_boundary=False)
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
albedo = 0.5  #Albedo(albedo_map, obs.latitude, obs.longitude).interpolate_albedo()
hapke = Hapke(size, obs, control, boundary, albedo)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run the model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed = disort.disort(usrang, usrtau, ibcnd, onlyfl, prnt, plank, lamber,
                                                                    deltamplus, do_pseudo_sphere, optical_depths,
                               ssa, polynomial_moments, temperatures, low_wavenumber, high_wavenumber, utau, umu0, phi0,
                                                                    umu, phi, fbeam, fisot, albedo,
                               surface_temp, top_temp, top_emissivity, planet_radius, h_lyr, hapke.rhoq, hapke.rhou, hapke.rho_accurate,
                                                                    hapke.bemst, hapke.emust, accur,
                               header, direct_beam_flux, diffuse_down_flux, diffuse_up_flux, flux_divergence,
                                                                    mean_intensity,
                               intensity, albedo_medium, transmissivity_medium)

print(uu[0, :15, 0])   # shape: (1, 81, 1)
