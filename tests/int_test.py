from pyRT_DISORT.model_atmosphere.aerosol import ForwardScatteringProperty, ForwardScatteringPropertyCollection
from pyRT_DISORT.utilities.external_files import ExternalFile
from pyRT_DISORT.model_atmosphere.atmosphere_grid import ModelGrid
from pyRT_DISORT.model_atmosphere.aerosol_column import Column
from pyRT_DISORT.model_atmosphere.vertical_profiles import Conrath
from pyRT_DISORT.model_atmosphere.phase_function import TabularLegendreCoefficients
from pyRT_DISORT.model_atmosphere.rayleigh import RayleighCo2
from pyRT_DISORT.model_atmosphere.model_atmosphere import ModelAtmosphere
from pyRT_DISORT.observation.observation import Observation
from pyRT_DISORT.model_controller.size import Size
from pyRT_DISORT.model_controller.control import Control
from pyRT_DISORT.model_controller.output import Output
from pyRT_DISORT.model_controller.unsure import Unsure
from pyRT_DISORT.model_atmosphere.boundary_conditions import BoundaryConditions
from pyRT_DISORT.model_atmosphere.surface import HapkeHG2Roughness
import disort

import os
import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Preprocessing steps
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read in the atmosphere file
data_path = '/data/'
#data_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')), 'data')  # This hack sucks but I figure we need a quick resolution
atmFile = ExternalFile(os.path.join(data_path, 'planets/mars/aux/mars_atm.npy'))
z_boundaries = np.linspace(80, 0, num=20)    # Define the boundaries I want to use. Note that I'm sticking with DISORT's convention of starting from TOA
model_grid = ModelGrid(atmFile.array, z_boundaries)
temperatures = model_grid.boundary_temperatures  # Define an oddball variable for use in the disort call

# Read in a 3D dust file
dustFile = ExternalFile(os.path.join(data_path, 'planets/mars/aux/dust_properties.fits'))
wavs = dustFile.array['wavelengths'].data
sizes = dustFile.array['particle_sizes'].data

# Add the columns from dustFile to dust_properties
c_ext = ForwardScatteringProperty(dustFile.array['primary'].data[:, :, 0], particle_size_grid=sizes, wavelength_grid=wavs)
c_sca = ForwardScatteringProperty(dustFile.array['primary'].data[:, :, 1], particle_size_grid=sizes, wavelength_grid=wavs)
dust_properties = ForwardScatteringPropertyCollection()
dust_properties.add_property(c_ext, 'c_extinction')
dust_properties.add_property(c_sca, 'c_scattering')

# Make a dust Conrath profile
conrath_profile = Conrath(model_grid, 10, 0.5)

# Define a smooth gradient of particle sizes from 0.5 to 1.5 microns
p_sizes = np.linspace(0.5, 1.5, num=len(conrath_profile.profile))

# Define the wavelengths where dust was observed
wavelengths = np.array([1, 9.3])

# Make a phase function. I'm allowing negative coefficients here
dust_phsfn_file = ExternalFile(os.path.join(data_path, 'planets/mars/aux/dust_phase_function.fits'))
dust_phsfn = TabularLegendreCoefficients(dust_phsfn_file.array['primary'].data,
                                         dust_phsfn_file.array['particle_sizes'].data,
                                         dust_phsfn_file.array['wavelengths'].data)

# Make the new Column where wave_ref = 9.3 microns and OD = 1
dust_col = Column(dust_properties, model_grid, conrath_profile.profile, p_sizes, wavelengths, 9.3, 1, dust_phsfn)
# Then you can access total_optical_depth, scattering_optical_depth, and phase_function as attributes

# ~~~~~~~~~~~~~~~~~~~~~ New things
# Make Rayleigh stuff
n_moments = 1000
rco2 = RayleighCo2(wavelengths, model_grid, n_moments)

# Make the model
model = ModelAtmosphere()
dust_info = (dust_col.total_optical_depth, dust_col.scattering_optical_depth, dust_col.scattering_optical_depth * dust_col.phase_function)
rayleigh_info = (rco2.scattering_optical_depths, rco2.scattering_optical_depths, rco2.phase_function)  # This works since scattering OD = total OD

# Add dust and Rayleigh scattering to the model
model.add_constituent(dust_info)
model.add_constituent(rayleigh_info)

# Once everything is in the model, compute the model. Then, slice off the wavelength dimension
model.compute_model()
optical_depths = model.hyperspectral_total_optical_depths[:, 1]
ssa = model.hyperspectral_total_single_scattering_albedos[:, 1]
polynomial_moments = model.hyperspectral_legendre_moments[:, :, 1]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make a fake observation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
short_wav = np.array([1])    # This is kinda clunky but real data will take flattened
long_wav = np.array([1.1])
sza = np.array([50])
emission_angle = np.array([40])
phase_angle = np.array([20])
obs = Observation(short_wav, long_wav, sza, emission_angle, phase_angle)
phi = np.array([obs.phi])
low_wavenumber = obs.low_wavenumbers
high_wavenumber = obs.high_wavenumbers
phi0 = obs.phi0
umu0 = obs.mu0
umu = np.array([obs.mu])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the size of the arrays
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
n_layers = model_grid.n_layers
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

# Choose which Hapke surface to use: the default 3 parameter one that comes with DISORT, a 2-lobed HG without roughness,
# or a 2-lobed HG with roughness. The purpose of these classes is to make the rhou, rhoq, bemst, emust, ... arrays
#hapke = Hapke(size, obs, control, boundary, albedo)
#hapke = HapkeHG2(size, obs, control, boundary, albedo, w=0.12, asym=0.75, frac=0.9, b0=1, hh=0.04, n_mug=200)
hapke = HapkeHG2Roughness(size, obs, control, boundary, albedo, w=0.12, asym=0.75, frac=0.5, b0=1, hh=0.04, n_mug=200, roughness=0.5)

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

print(uu[0, :20, 0])   # shape: (1, 81, 1)

# This gives: 0.01475769 using python3.8
