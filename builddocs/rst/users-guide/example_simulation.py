import numpy as np
from pyRT_DISORT.observation import Angles, Spectral

pixel_wavelengths = np.array([1, 2, 3, 4, 5])
dummy_wavelengths = np.broadcast_to(pixel_wavelengths, (15, 20, 5))
width = 0.05

spectral = Spectral(pixel_wavelengths - width, pixel_wavelengths + width)
short_wavelengths = spectral.short_wavelength
long_wavelengths = spectral.long_wavelength
high_wavenumbers = spectral.high_wavenumber
low_wavenumbers = spectral.low_wavenumber

dummy_angles = np.outer(np.linspace(5, 10, num=15), np.linspace(5, 8, num=20))
angles = Angles(dummy_angles, dummy_angles, dummy_angles)
mu = angles.mu
mu0 = angles.mu0
phi = angles.phi
phi0 = angles.phi0

from pyRT_DISORT.eos import Hydrostatic

altitude_grid = np.linspace(100, 0, num=51)
pressure_profile = 500 * np.exp(-altitude_grid / 10)
temperature_profile = np.linspace(150, 250, num=51)
z_grid = np.linspace(100, 0, num=15)
mass = 7.3 * 10**-26
gravity = 3.7

hydro = Hydrostatic(altitude_grid, pressure_profile, temperature_profile,
                    z_grid, mass, gravity)
altitude = hydro.altitude
pressure = hydro.pressure
temperature = hydro.temperature
number_density = hydro.number_density
column_density = hydro.column_density
n_layers = hydro.n_layers
scale_height = hydro.scale_height

from pyRT_DISORT.rayleigh import RayleighCO2

short_rayleigh = RayleighCO2(altitude, high_wavenumbers, column_density)
rayleigh_phase_function = short_rayleigh.phase_function
rayleigh_od = short_rayleigh.scattering_optical_depth

from pyRT_DISORT.controller import ComputationalParameters, ModelBehavior

cp = ComputationalParameters(hydro.n_layers, 64, 16, 1, 1, 80)
mb = ModelBehavior()


from pyRT_DISORT.radiation import IncidentFlux, ThermalEmission

flux = IncidentFlux()
beam_flux = flux.beam_flux
iso_flux = flux.isotropic_flux

te = ThermalEmission()
thermal_emission = te.thermal_emission
bottom_temp = te.bottom_temperature
top_temp = te.top_temperature
top_emissivity = te.top_emissivity

from pyRT_DISORT.output import OutputArrays, OutputBehavior, UserLevel
# After defining CP, make these in the example (outputArrays)

ob = OutputBehavior()
ibcnd = ob.incidence_beam_conditions
only_fluxes = ob.only_fluxes
user_angles = ob.user_angles
user_tau = ob.user_optical_depths

ulv = UserLevel()
od_output = ulv.optical_depth_output
