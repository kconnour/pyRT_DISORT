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
























