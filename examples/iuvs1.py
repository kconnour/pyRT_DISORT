# Parallelize finding the best fit parameters for IUVS data assuming just ice and dust

# Built-in imports
import os
import time
from tempfile import mkdtemp

# 3rd-party imports
import numpy as np
from astropy.io import fits
from scipy import optimize
import joblib

# Local imports
import disort
from pyRT_DISORT.data.get_data import get_data_path
from pyRT_DISORT.preprocessing.utilities.external_files import ExternalFile
from pyRT_DISORT.preprocessing.model.model_atmosphere import ModelAtmosphere
from pyRT_DISORT.preprocessing.model.aerosol import Aerosol
from pyRT_DISORT.preprocessing.model.atmosphere import Layers
from pyRT_DISORT.preprocessing.model.aerosol_column import Column, Conrath, GCMProfile
from pyRT_DISORT.preprocessing.observation import Observation
from pyRT_DISORT.preprocessing.controller.output import Output
from pyRT_DISORT.preprocessing.model.phase_function import TabularLegendreCoefficients
from pyRT_DISORT.preprocessing.controller.size import Size
from pyRT_DISORT.preprocessing.controller.unsure import Unsure
from pyRT_DISORT.preprocessing.controller.control import Control
from pyRT_DISORT.preprocessing.model.boundary_conditions import BoundaryConditions
from pyRT_DISORT.preprocessing.model.rayleigh import RayleighCo2
from pyRT_DISORT.preprocessing.model.surface import HapkeHG2Roughness
from pyRT_DISORT.preprocessing.utilities.shared_array import SharedArray

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Do the calculations I'll only need to do once when iteratively solving
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define files I'll need
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
observation_file = '/media/kyle/Samsung_T5/IUVS_data/orbit03400/mvn_iuv_l1b_apoapse-orbit03453-muv_20160708T051356_v13_r01.fits.gz'
dust_phase_function = ExternalFile(os.path.join(get_data_path(), 'planets/mars/aux/dust_phase_functions.npy'))
dust_phase_radii = ExternalFile(os.path.join(get_data_path(), 'planets/mars/aux/dust_phase_function_radii.npy'))
dust_phase_wavs = ExternalFile(os.path.join(get_data_path(), 'planets/mars/aux/dust_phase_function_wavelengths.npy'))
ice_coeff = ExternalFile(os.path.join(get_data_path(), 'planets/mars/aux/legendre_coeff_h2o_ice.npy'))
dustfile = ExternalFile(os.path.join(get_data_path(), 'planets/mars/aux/dust.npy'))
icefile = ExternalFile(os.path.join(get_data_path(), 'planets/mars/aux/ice.npy'))
atm = ExternalFile(os.path.join(get_data_path(), 'planets/mars/aux/mars_atm_copy.npy'))
altitude_map = ExternalFile(os.path.join(get_data_path(), 'planets/mars/aux/altitude_map.npy'))
solar_spec = ExternalFile(os.path.join(get_data_path(), 'aux/solar_spectrum.npy'))
albedo_map = ExternalFile(os.path.join(get_data_path(), 'planets/mars/aux/albedo_map.npy'))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define observation-related quantities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read in data from the file and store values as numpy array (for readability only)
f = '/media/kyle/Samsung_T5/IUVS_data/orbit03400/mvn_iuv_l1b_apoapse-orbit03453-muv_20160708T051356_v13_r01.fits.gz'
hdulist = fits.open(f)
wavs = np.squeeze(hdulist['observation'].data['wavelength'])[0, :] / 1000   # Convert to microns
diff = np.diff(wavs)
diff = np.concatenate((diff, np.array([diff[0]])))
short_wavs = (wavs - diff/2)
long_wavs = (wavs + diff/2)
szas = hdulist['pixelgeometry'].data['pixel_solar_zenith_angle'][:2, :].flatten()
emission_angles = hdulist['pixelgeometry'].data['pixel_emission_angle'][:2, :].flatten()
phase_angles = hdulist['pixelgeometry'].data['pixel_phase_angle'][:2, :].flatten()

# Put the arrays into the Observation class
obs = Observation(short_wavs, long_wavs, szas, emission_angles, phase_angles)

# Access the variables that I care about
low_wavenumber = obs.low_wavenumber
high_wavenumber = obs.high_wavenumber
phi0 = obs.phi0
umu = obs.mu
umu0 = obs.mu0
phi = obs.phi

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read in the atmosphere
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
lay = Layers(atm.array)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the size of the arrays
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
n_layers = lay.n_layers
n_moments = 1000
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
albedo = 0.02
hapke = HapkeHG2Roughness(size, obs, control, boundary, albedo, w=0.12, asym=0.75, frac=0.5, b0=1, hh=0.04, n_mug=200,
                          roughness=0.5)
temperatures = lay.temperature_boundaries

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get aerosol properties at the instrument's wavelength
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dust = Aerosol(dustfile.array, wavs, 9.3)     # 9.3 is the wavelength reference
ice = Aerosol(icefile.array, wavs, 12.1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make model-invariant quantitites
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
rco2 = RayleighCo2(wavs, lay, n_moments)
dust_conrath = Conrath(lay, 10, 0.5)   # 10 = scale height, 0.5 = Conrath nu
# This is a somewhat cryptic way of making a constant profile at altitudes between 25--75 km
iceprof = np.where((25 < lay.altitude_layers) & (75 > lay.altitude_layers), 1, 0)
ice_profile = GCMProfile(lay, iceprof)
rayleigh_info = (rco2.hyperspectral_optical_depths, rco2.hyperspectral_optical_depths, rco2.hyperspectral_layered_phase_function)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# These are all calculations I have to re-do for each time I try a new parameter
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# THIS USES MULTIPROCESSING MODULE
# Make an array in shared memory
'''junkThing = np.zeros((len(szas), len(wavs)), dtype=np.float)
shm = shared_memory.SharedMemory(name='shared_answer', create=True, size=junkThing.nbytes)
answer = np.ndarray(junkThing.shape, dtype=junkThing.dtype, buffer=shm.buf)
answer[:] = junkThing[:]'''

# THIS USES JOBLIB MODULE
#filename = os.path.join(mkdtemp(), 'myNewFile.dat')
#answer = np.memmap(filename, dtype=np.float, shape=(len(szas), len(wavs)), mode='w+')

# Use my code to create an array in shared memory
sa = SharedArray((len(szas), len(wavs)))
answer = sa.array


def myModel(guess, pixel):
    dust_od = guess[0]
    ice_od = guess[1]
    if dust_od <= 0 or ice_od <= 0:
        return 99999
    # The column will change everytime I use a different OD
    dust_column = Column(dust, lay, dust_conrath, np.array([1]), np.array([dust_od]))
    ice_column = Column(ice, lay, ice_profile, np.array([2]), np.array([ice_od]))

    # The phase functions change as columns change
    dust_phase = TabularLegendreCoefficients(dust_column, dust_phase_function.array,
                                             particle_sizes=dust_phase_radii.array, wavelengths=dust_phase_wavs.array)
    ice_phase = TabularLegendreCoefficients(ice_column, ice_coeff.array)

    # Make the model
    model = ModelAtmosphere()
    dust_info = (dust_column.hyperspectral_total_optical_depths, dust_column.hyperspectral_scattering_optical_depths,
                 dust_phase.coefficients)
    ice_info = (ice_column.hyperspectral_total_optical_depths, ice_column.hyperspectral_scattering_optical_depths,
                ice_phase.coefficients)

    # Add everything to the model
    model.add_constituent(dust_info)
    model.add_constituent(ice_info)
    model.add_constituent(rayleigh_info)

    # Once everything is in the model, compute the model
    model.compute_model()
    optical_depths = model.hyperspectral_total_optical_depths
    ssa = model.hyperspectral_total_single_scattering_albedos
    polynomial_moments = model.hyperspectral_legendre_moments

    #existing_shm = shared_memory.SharedMemory(name='shared_answer')
    #shared_array = np.ndarray((len(szas), len(wavs)), dtype=np.float, buffer=existing_shm.buf)
    for w in range(len(wavs)):
        # The 0s here are just me testing this on a single pixel
        junk, junk, junk, junk, junk, uu, junk, junk = disort.disort(usrang, usrtau, ibcnd, onlyfl, prnt, plank,
             lamber, deltamplus, do_pseudo_sphere, optical_depths[:, w], ssa[:, w], polynomial_moments[:, :, w], temperatures,
             low_wavenumber[w], high_wavenumber[w], utau, umu0[pixel], phi0, umu[pixel], phi[pixel], fbeam, fisot, albedo,
             surface_temp, top_temp, top_emissivity, planet_radius, h_lyr, hapke.rhoq, hapke.rhou, hapke.rho_accurate[:, pixel],
             hapke.bemst, hapke.emust, accur, header, direct_beam_flux, diffuse_down_flux,
             diffuse_up_flux, flux_divergence, mean_intensity, intensity[:, :, pixel], albedo_medium, transmissivity_medium)
        # This is what I'd do for multiprocessing
        #shared_array[pixel, w] = uu[0, 0]
        answer[pixel, w] = uu[0, 0]


def calculate_model_difference(guess, pixel):
    # This is just a sample spectrum I took... each pixel should have its own
    target = np.array([0.784799E-01, 0.753058E-01, 0.698393E-01, 0.654867E-01, 0.605224E-01, 0.572824E-01, 0.534369E-01,
                       0.496474E-01, 0.464914E-01, 0.427855E-01, 0.429494E-01, 0.457046E-01, 0.491979E-01, 0.495246E-01,
                       0.457136E-01, 0.433431E-01, 0.483898E-01, 0.504774E-01, 0.457426E-01])
    test_run = myModel(guess, pixel)
    return np.sum((test_run - target)**2)


def do_optimization(guess, pixel):
    return optimize.minimize(calculate_model_difference, np.array(guess), pixel, method='Nelder-Mead').x


# Without this pool won't cooperate
if __name__ == '__main__':
    t0 = time.time()
    n_cpus = 8  # mp.cpu_count()
    pixel_inds = np.linspace(0, len(szas)-1, num=len(szas), dtype='int')
    #iterable = [[[0.8, 0.2], f] for f in pixel_inds]

    #m = joblib.Parallel(n_jobs=-2, prefer='processes')(joblib.delayed(myModel)([0.8, 0.2], f) for f in pixel_inds)
    joblib.Parallel(n_jobs=-2, prefer='processes')(joblib.delayed(myModel)([0.8, 0.2], f) for f in pixel_inds)
    print(np.amax(answer))
    #np.save('/home/kyle/bestArray.npy', answer)
    #sa.delete()
    t1 = time.time()
    print(t1 - t0)
