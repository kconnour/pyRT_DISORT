import disort

#from atmosphere import Atmosphere
#from planets.mars.mars_atmosphere import AtmosphericDust
import numpy as np
from generic.atmosphere import Atmosphere
from generic.aerosol import Aerosol
from generic.aerosol_column import Column
from generic.observation import Observation
from generic.output import Output
from planets.mars.map import Albedo

# For now I'm just passing files in where necessary
atmfile = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/mars_atm.npy'
dustfile = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/dust.npy'
polyfile = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/legendre_coeff_dust.npy'

'''
# Create an atmosphere object, which holds z, P, T, n, and N---equation of state variables
# It finds what variables are absent and makes them based on optional arguments. Useful for exoplanet
# (where you know nothing) to GCM (where you know everything)
atm = Atmosphere(atmfile)

# The values of the 5 variables are stored in the object
temperatures = atm.T
# number_density = atm.n   # etc...

# If desired, you can add Rayleigh scattering optical depth to the atmosphere at a wavelength (microns)
atm.add_rayleigh_co2_optical_depth(9.3)

# Then, you can create aerosol objects which you then add to the atmosphere
# (..., scale height, Conrath nu, column OD, wavelength)
# I'm still working to make a single input file
dust = AtmosphericDust(atmfile, dustfile, polyfile, 10, 0.5, 1, 9.3)
atm.add_aerosol(dust)

# After adding all the aerosols to the atmosphere, you can make the relevant arrays
dust_od = atm.calculate_column_optical_depth()
ssa = atm.calculate_single_scattering_albedo()
moments = atm.calculate_polynomial_moments()

print(dust_od)
print(ssa)
raise SystemExit('')'''

atm = Atmosphere('/home/kyle/repos/pyRT_DISORT/planets/mars/aux/mars_atm.npy')
atm.add_rayleigh_co2_optical_depth(np.array([1, 2, 5, 15, 24, 49]))
dust = Aerosol(128, 'emp', 1, np.array([1, 2, 5, 15, 24, 49]), g=0.5,
               aerosol_file='/home/kyle/repos/pyRT_DISORT/planets/mars/aux/dust.npy',
               legendre_file='/home/kyle/repos/pyRT_DISORT/planets/mars/aux/legendre_coeff_dust.npy')
c = Column(dust, 10, 0.5, 1)
atm.add_column(c)

# Make the disort variables
temperatures = atm.T
dust_od = atm.calculate_column_optical_depth()[:, 0]
ssa = atm.calculate_single_scattering_albedo()[:, 0]
moments = atm.calculate_polynomial_moments()[:, :, 0]
print(moments.shape)

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

# Just define a bunch of variables
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

# Make the output variables
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

dfdt, uavg, uu = disort.disort(usrang, usrtau, ibcnd, onlyfl, prnt, plank, lamber, deltamplus, do_pseudo_sphere, dust_od,
                               ssa, moments, temperatures, low_wavenumber, high_wavenumber, utau, umu0, phi0, umu, phi, fbeam, fisot, albedo,
                               surface_temp, top_temp, top_emissivity, earth_radius, h_lyr, rhoq, rhou, rho_accurate, bemst, emust, accur,
                               header, direct_beam_flux, diffuse_down_flux, diffuse_up_flux, flux_divergence, mean_intensity,
                               intensity, albedo_medium, transmissivity_medium)

print(uu)
