# Make a really crappy "main" file to do a retrieval for an atmosphere of only Martian dust
import numpy as np

# Local imports
from planets.mars.aerosols import MarsDust
from observation import Observation
import disort
from map import Albedo

# Read in the atmosphere and define variables   NOTE: I should have an option to read in OR create an atm
mars_atm_file = '/planets/mars/aux/mars_atm.npy'
atmosphere = np.load(mars_atm_file, allow_pickle=True)
n_boundaries = atmosphere.shape[0]
n_layers = n_boundaries - 1
temperatures = atmosphere[:, 2]

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

# DISORT variables that can change, but these are good test cases
plank = True
lamber = True
n_streams = 8
surface_temp = 270
top_temp = 0
top_emissivity = 1

# DISORT variables that I'm defining
do_pseudo_sphere = False
earth_radius = 6371
deltamplus = False
header = 'myHeader'

# Get the phase function moments
dust_phsfn_file = '/planets/mars/aux/legendre_coeff_dust.npy'
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
phi = np.array([obs.phi()])
low_wavenumber = obs.calculate_low_wavenumber()
high_wavenumber = obs.calculate_high_wavenumber()
phi0 = obs.phi0
umu0 = obs.mu0()
umu = np.array([obs.mu()])
fbeam = obs.calculate_solar_flux()

# Make the output variables. I still don't know what the damn nmaxulv is
direct_beam_flux = np.zeros(n_boundaries)
diffuse_down_flux = np.zeros(n_boundaries)
diffuse_up_flux = np.zeros(n_boundaries)
flux_divergence = np.zeros(n_boundaries)
mean_intensity = np.zeros(n_boundaries)
intensity = np.zeros((n_umu, n_boundaries, n_phi))
albedo_medium = np.zeros(n_umu)
transmissivity_medium = np.zeros(n_umu)

# Get albedo (it probably shouldn't go here though...)
albedo_map = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/albedo_map.npy'
albedo = Albedo(obs.latitude, obs.longitude, albedo_map)

# Unsure what's happening here
rho_accurate = np.zeros((n_umu, n_phi))




dust_aerosol_file = '/planets/mars/aux/dust.npy'
wavelength = 250  # nm
theta = 0.5  # radians; angle at which to evaluate phase function
dust = MarsDust(dust_phsfn_file, dust_aerosol_file, theta, wavelength)

'''Remaining
dtauc
ntau
ssalb
pmom
utau
h_lyr
rhoq
rhou
bemst
emust
'''



dfdt, uavg, uu = disort.disort(usrang, usrtau, ibcnd, onlyfl, prnt, plank, lamber, deltamplus, do_pseudo_sphere, dtauc,
                               ssalb, pmom, temperatures, low_wavenumber, high_wavenumber, utau, umu0, phi0, umu, phi, fbeam, fisot, albedo,
                               surface_temp, top_temp, top_emissivity, earth_radius, h_lyr, rhoq, rhou, rho_accurate, bemst, emust, accur,
                               header, direct_beam_flux, diffuse_down_flux, diffuse_up_flux, flux_divergence, mean_intensity,
                               intensity, albedo_medium, transmissivity_medium, maxcly=n_layers, maxmom=n_moments,
                               maxcmu=n_streams, maxumu=n_umu, maxphi=n_phi, maxulv=ntau)