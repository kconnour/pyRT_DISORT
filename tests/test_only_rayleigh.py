import os
import numpy as np
from pyRT_DISORT.observation import Angles, Wavelengths
from pyRT_DISORT.untested.aerosol import ForwardScatteringProperty, ForwardScatteringPropertyCollection
from pyRT_DISORT.untested_utils.utilities.external_files import ExternalFile
from pyRT_DISORT.eos import ModelEquationOfState, eos_from_array
from pyRT_DISORT.untested.aerosol_column import Column
from pyRT_DISORT.untested.vertical_profiles import Conrath
from pyRT_DISORT.untested.phase_function import TabularLegendreCoefficients
from pyRT_DISORT.untested.rayleigh import RayleighCo2
from pyRT_DISORT.untested.model_atmosphere import ModelAtmosphere
from pyRT_DISORT.controller import ComputationalParameters, ModelBehavior, OutputArrays
from pyRT_DISORT.flux import IncidentFlux, ThermalEmission
from pyRT_DISORT.untested.unsure import Unsure
from pyRT_DISORT.surface import Lambertian
from pyRT_DISORT.untested.surface import HapkeHG2Roughness
import disort

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Observation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# New: The old Observation class took angles and wavelengths, but they operated
# independently so I made them into 2 classes. This class basically just creates
# wavenumbers from wavelengths
short_wav = np.array([1])   # microns
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
ang = Angles(sza, emission_angle, phase_angle)
mu = ang.mu
mu0 = ang.mu0
phi = ang.phi
phi0 = ang.phi0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read in external files
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read in the atmosphere file
data_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')), 'data')  # This hack sucks but I figure we need a quick resolution
#atmFile = ExternalFile(os.path.join(data_path, 'planets/mars/aux/mars_atm.npy'))
project_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
atmFile = ExternalFile(os.path.join(project_path, 'tests/marsatm.npy'))
z_boundaries = np.linspace(80, 0, num=20)    # Define the boundaries I want to use. Note that I'm sticking with DISORT's convention of starting from TOA
# New: eos_from_array is a function that returns a custom class---to help you out
model_eos = eos_from_array(atmFile.array, z_boundaries)
temperatures = model_eos.temperature_boundaries  # Define an oddball variable for use in the disort call

# Read in a 3D dust file
dustFile = ExternalFile(os.path.join(data_path, 'planets/mars/aux/dust_properties.fits'))
wavs = dustFile.array['wavelengths'].data
sizes = dustFile.array['particle_sizes'].data

# Make Rayleigh stuff
n_moments = 1000
rco2 = RayleighCo2(short_wav, model_eos, n_moments)

# Make the model
model = ModelAtmosphere()
rayleigh_info = (rco2.scattering_optical_depths, rco2.scattering_optical_depths, rco2.phase_function)  # This works since scattering OD = total OD

# Add dust and Rayleigh scattering to the model
model.add_constituent(rayleigh_info)

# Once everything is in the model, compute the model. Then, slice off the wavelength dimension
model.compute_model()
optical_depths = model.hyperspectral_total_optical_depths
ssa = model.hyperspectral_total_single_scattering_albedos
polynomial_moments = model.hyperspectral_legendre_moments

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make the size of the computational parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
n_layers = model_eos.n_layers
n_streams = 16
n_umu = 1
n_phi = len(phi)
n_user_levels = 81
cp = ComputationalParameters(n_layers, n_moments, n_streams, n_phi, n_umu, n_user_levels)

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
# Make the arrays I'm unsure about (for now)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
uns = Unsure(cp)
h_lyr = uns.h_lyr

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Surface treatment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# New: I made an abstract Surface class, and plan to make all surfaces inherit
# from it. I haven't gotten to it yet, but I did make a Lambertian class that
# does do that. The idea (unless I stumble across a problem...) is that Surface
# makes arrays of 0s for rhou, rhoq, bemst, etc. In the special case of a
# Lambertian surface, the Lambertian class will just have these arrays. In the
# case Hapke surfaces, these arrays will be input into disobrdf, where that
# routine will populate those arrays with values. So any class that's derived
# from Surface will all have the same properties.

# To use the Lambertian class, do the following:
lamb = Lambertian(0.5, cp)   # albedo = 0.5
albedo = lamb.albedo
lamber = lamb.lambertian
rhou = lamb.rhou
rhoq = lamb.rhoq
bemst = lamb.bemst
emust = lamb.emust
rho_accurate = lamb.rho_accurate


# These are the next things I plan to update, but for now (these are untested)...
# Choose which Hapke surface to use: the default 3 parameter one that comes with DISORT, a 2-lobed HG without roughness,
# or a 2-lobed HG with roughness. The purpose of these classes is to make the rhou, rhoq, bemst, emust, ... arrays
#hapke = Hapke(size, obs, control, boundary, albedo)
#hapke = HapkeHG2(size, obs, control, boundary, albedo, w=0.12, asym=0.75, frac=0.9, b0=1, hh=0.04, n_mug=200)
#hapke = HapkeHG2Roughness(0.5, cp, mb, incident_flux, ang, w=0.12, asym=0.75, frac=0.5, b0=1, hh=0.04, n_mug=200, roughness=0.5)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# I guess I have no idea where to put this still
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
utau = np.zeros(n_user_levels)

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
# This gives         0.32141164
# disort_multi gives 0.321434200
# I'm running ./disort_multi -dust_conrath 0.5, 10 -dust_phsfn 98 -NSTR 16 < testInput.txt
# testInput.txt is: 1, 0.5, 10, 30, 50, 40, 20, 0, 0, 0
