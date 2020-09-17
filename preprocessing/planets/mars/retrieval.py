# Make a really crappy "main" file to do a retrieval for an atmosphere of only Martian dust
import numpy as np

# Local imports
#from planets.mars.mars_aerosols import MarsDust
from preprocessing.observation import Observation
import disort
from preprocessing.planets.mars.map import Albedo
from preprocessing.controller.output import Output
from atmosphere import Atmosphere
from preprocessing.planets.mars.mars_atmosphere import DustAtmosphere

# Read in the atmosphere and define variables   NOTE: I should have an option to read in OR create an atm
mars_atm_file = '/preprocessing/planets/mars/aux/mars_atm.npy'
dust_file = '/preprocessing/planets/mars/aux/dust.npy'
atmosphere = Atmosphere(mars_atm_file)
n_boundaries = len(atmosphere.z)
n_layers = n_boundaries - 1
temperatures = atmosphere.T

# Add dust to the atmosphere
marsdust = DustAtmosphere(mars_atm_file, dust_file, 10000, 0.5)
column_OD = 2
tau_dust = marsdust.calculate_dust_optical_depths(column_OD)

# DISORT variables that Mike just defined
usrtau = False
usrang = True
fisot = 0
onlyfl = True
n_umu = 0
n_phi = 1
ibcnd = 0
prnt = np.array([True, False, False, False, True])
accur = 0

# DISORT variables that can change, but these seem like reasonable test cases
plank = True
lamber = True
n_streams = 8
surface_temp = 270
top_temp = 0
top_emissivity = 1
n_user_levels = 20
n_cmu = 1

# DISORT variables that I'm defining
do_pseudo_sphere = False
earth_radius = 6371
deltamplus = False
header = 'myHeader'

# Get the phase function moments
dust_phsfn_file = '/preprocessing/planets/mars/aux/legendre_coeff_dust.npy'
poly_dust_phase_moments = np.load(dust_phsfn_file, allow_pickle=True)
n_moments = len(poly_dust_phase_moments)
# I NEED TO MAKE A 2D GRID OF PMOM ``````````````````````````````````````````````````````

# Make up a "fake observation"
short_wav = 1000
long_wav = 1100
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
fbeam = obs.calculate_solar_flux()

# Make the output variables
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

# Get albedo (it probably shouldn't go here though...)
albedo_map = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/albedo_map.npy'
albedo = Albedo(obs.latitude, obs.longitude, albedo_map)

dfdt, uavg, uu = disort.disort(usrang, usrtau, ibcnd, onlyfl, prnt, plank, lamber, deltamplus, do_pseudo_sphere, tau_dust,
                               ssalb, pmom, temperatures, low_wavenumber, high_wavenumber, utau, umu0, phi0, umu, phi, fbeam, fisot, albedo,
                               surface_temp, top_temp, top_emissivity, earth_radius, h_lyr, rhoq, rhou, rho_accurate, bemst, emust, accur,
                               header, direct_beam_flux, diffuse_down_flux, diffuse_up_flux, flux_divergence, mean_intensity,
                               intensity, albedo_medium, transmissivity_medium, maxcly=n_layers, maxmom=n_moments,
                               maxcmu=n_streams, maxumu=n_umu, maxphi=n_phi, maxulv=n_user_levels)