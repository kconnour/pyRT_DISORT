"""Perform an integrated test under the following conditions:

- The model contains only Rayleigh scattering
- The surface is Lambertian

"""

import os
import numpy as np
import disort
from pyRT_DISORT.controller import ComputationalParameters, ModelBehavior, \
    OutputArrays, UserLevel
from pyRT_DISORT.eos import eos_from_array
from pyRT_DISORT.flux import IncidentFlux, ThermalEmission
from pyRT_DISORT.observation import Angles, Wavelengths
from pyRT_DISORT.surface import Lambertian
from pyRT_DISORT.untested_utils.utilities.external_files import ExternalFile
from pyRT_DISORT.untested.rayleigh import RayleighCo2
from pyRT_DISORT.untested.model_atmosphere import ModelAtmosphere

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Observation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# New: The old Observation class took angles and wavelengths, but they operated
# independently so I made them into 2 classes. This class basically just creates
# wavenumbers from wavelengths
short_wav = np.array([1, 9.3])   # microns
long_wav = short_wav + 1
wavelengths = Wavelengths(short_wav, long_wav)
low_wavenumber = wavelengths.low_wavenumber
high_wavenumber = wavelengths.high_wavenumber

# New: Angles is now responsible for making mu, mu0, etc. For imagers like IUVS
# (which have a 2D array of angles in the .fits file) I no longer ask that the
# input angles are flattened. For for simplicity, I'll just stick to a single
# value
sza = np.array([50])
emission_angle = np.array([40])
phase_angle = np.array([20])
angles = Angles(sza, emission_angle, phase_angle)
mu = angles.mu
mu0 = angles.mu0
phi = angles.phi
phi0 = angles.phi0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read in external files
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# New: I moved the relevant files into the tests directory. This way I can have
# files that change (like the atmosphere equation of state file) elsewhere
# while keeping my tests unchanged.

# Read in the atmosphere file
tests_path = os.path.dirname(os.path.realpath(__file__))
eos_file = ExternalFile(os.path.join(tests_path, 'aux/marsatm.npy'))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the equation of state variables on a custom grid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
z_boundaries = np.linspace(80, 0, num=20)    # Define the boundaries to use
# New: eos_from_array is a helper function to make a ModelEquationOfState class
# See eos.py for more info but this replaces ModelGrid (mostly I think the name
# is better)
model_eos = eos_from_array(eos_file.array, z_boundaries, 3.71, 7.3*10**-26)
temperatures = model_eos.temperature_boundaries
h_lyr = model_eos.scale_height_boundaries

print(model_eos.column_density_layers)
raise SystemExit(9)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Construct aerosol/model properties
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make Rayleigh stuff---the only stuff going into the model
n_moments = 1000
rco2 = RayleighCo2(short_wav, model_eos, n_moments)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
model = ModelAtmosphere()
# Make tuples of (dtauc, ssalb, pmom) for each constituent
rayleigh_info = (rco2.scattering_optical_depths, rco2.scattering_optical_depths,
                 rco2.phase_function)  # This works since scat OD = total OD

# Add dust and Rayleigh scattering to the model
model.add_constituent(rayleigh_info)

# Once everything is in the model, compute the model. Then, slice off the
# wavelength dimension since DISORT can only handle 1 wavelength at a time
model.compute_model()
optical_depths = model.hyperspectral_total_optical_depths[:, 1]
ssa = model.hyperspectral_total_single_scattering_albedos[:, 1]
polynomial_moments = model.hyperspectral_legendre_moments[:, :, 1]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the size of the computational parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Semi-new: this class just holds parameters DISORT wants. It's kinda useless by
# itself but a valuable input into the upcoming classes
n_layers = model_eos.n_layers
n_streams = 16
n_umu = 1
n_phi = len(phi)
n_user_levels = 81
cp = ComputationalParameters(
    n_layers, n_moments, n_streams, n_phi, n_umu, n_user_levels)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make misc variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# New: I split the old "control" classes into additional classes that I think
# are more accurately named and grouped. I think many of these variables in
# DISORT are horribly named. For clarity, I included the "DISORT" name as the
# variable name, and the name I prefer as the property
# (i.e. fisot = isotropic_flux). There's really no reason to define any of these
# variables here---you can just put them directly into the disort call---but I
# thought it might be helpful.

# Semi-new: another note, that many of these variables take a boolean or float
# value. I made them optional, and use default values that disort_mulit uses
incident_flux = IncidentFlux()
fbeam = incident_flux.beam_flux
fisot = incident_flux.isotropic_flux

te = ThermalEmission()
plank = te.thermal_emission
btemp = te.bottom_temperature
ttemp = te.top_temperature
temis = te.top_emissivity

mb = ModelBehavior()
accur = mb.accuracy
deltamplus = mb.delta_m_plus
dopseudosphere = mb.do_pseudo_sphere
header = mb.header
ibcnd = mb.incidence_beam_conditions
onlyfl = mb.only_fluxes
prnt = mb.print_variables
radius = mb.radius
usrang = mb.user_angles
usrtau = mb.user_optical_depths

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the output arrays
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
oa = OutputArrays(cp)
albmed = oa.albedo_medium
flup = oa.diffuse_up_flux
rfldn = oa.diffuse_down_flux
rfldir = oa.direct_beam_flux
dfdt = oa.flux_divergence
uu = oa.intensity
uavg = oa.mean_intensity
trnmed = oa.transmissivity_medium

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Optical depth output structure
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# I made this into its own class to handle utau. This one singular variable is
# an absolute nightmare and DISORT should be tweaked to get rid of it, but
# you're not paying me to discuss the bad decisions that went into making this
utau = UserLevel(cp, mb).optical_depth_output

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Surface treatment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# New: I made an abstract Surface class, and all surfaces you'd use inherit from
# it. The idea is that surface makes arrays of 0s for rhou, rhoq, bemst, etc. In
# the special case of a Lambertian surface, these are fine inputs to DISORT
# since it ignores these arrays when LAMBER=True. Otherwise, these arrays get
# populated when you call disobrdf. For instance, if you instantiate HapkeHG2,
# it calls disobrdf in the constructor which populates the surface arrays. The
# benefit to the user is that all classes derived from Surface will have the
# same properties.
lamb = Lambertian(0.5, cp)   # albedo = 0.5
albedo = lamb.albedo
lamber = lamb.lambertian
rhou = lamb.rhou
rhoq = lamb.rhoq
bemst = lamb.bemst
emust = lamb.emust
rho_accurate = lamb.rho_accurate

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run the model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed = \
    disort.disort(usrang, usrtau, ibcnd, onlyfl, prnt, plank, lamber,
                  deltamplus, dopseudosphere, optical_depths, ssa,
                  polynomial_moments, temperatures, low_wavenumber,
                  high_wavenumber, utau, mu0, phi0, mu, phi, fbeam, fisot,
                  albedo, btemp, ttemp, temis, radius, h_lyr, rhoq, rhou,
                  rho_accurate, bemst, emust, accur, header, rfldir,
                  rfldn, flup, dfdt, uavg, uu, albmed, trnmed)

print(uu[0, 0, 0])   # shape: (1, 81, 1)
# This gives         0.3213986
# disort_multi gives 0.321434200
# I'm running ./disort_multi -dust_conrath 0.5, 10 -dust_phsfn 98 -NSTR 16 < testInput.txt
# testInput.txt is: 1, 0.5, 10, 30, 50, 40, 20, 0, 0, 0
