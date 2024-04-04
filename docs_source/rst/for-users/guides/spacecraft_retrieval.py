import numpy as np
import pyrt
import disort

dummy_angles = np.outer(np.linspace(5, 10, num=15), np.linspace(5, 8, num=20))

UMU = np.cos(np.radians(dummy_angles))  # since dummy_angles is emission
UMU0 = np.cos(np.radians(dummy_angles))  # since dummy_angles is incidence
PHI = pyrt.azimuth(dummy_angles, dummy_angles, dummy_angles)
PHI0 = np.zeros(PHI.shape)

UMU = UMU[0, 0]
UMU0 = UMU0[0, 0]
PHI = PHI[0, 0]
PHI0 = PHI0[0, 0]

pixel_wavelengths = np.array([1, 2, 3, 4, 5]) / 5
spectral_width = 0.05

WVNMHI = pyrt.wavenumber(pixel_wavelengths - spectral_width)
WVNMLO = pyrt.wavenumber(pixel_wavelengths + spectral_width)

altitude_grid = np.linspace(100, 0, num=15)
pressure_profile = 500 * np.exp(-altitude_grid / 10)
temperature_profile = np.linspace(150, 250, num=15)
mass = 7.3 * 10**-26
gravity = 3.7

column_density = pyrt.column_density(pressure_profile, temperature_profile, altitude_grid)

TEMPER = temperature_profile
H_LYR = altitude_grid

rayleigh_co2 = pyrt.rayleigh_co2(column_density, pixel_wavelengths)
print(np.sum(rayleigh_co2.optical_depth, axis=0))

altitude_midpoint = (altitude_grid[:-1] + altitude_grid[1:]) / 2

q0 = 1
nu = 0.01
dust_profile = pyrt.conrath(altitude_midpoint, q0, 10, nu)

particle_size_grid = np.linspace(0.5, 10, num=50)
wavelength_grid = np.linspace(0.2, 50, num=20)
extinction_cross_section = np.ones((50, 20))
scattering_cross_section = np.ones((50, 20)) * 0.5

particle_size_gradient = np.linspace(1, 1.5, num=len(altitude_midpoint))

ext = pyrt.extinction_ratio(extinction_cross_section, particle_size_grid, wavelength_grid, 9.3)
ext = pyrt.regrid(ext, particle_size_grid, wavelength_grid, particle_size_gradient, pixel_wavelengths)
dust_optical_depth = pyrt.optical_depth(dust_profile, column_density, ext, 1)

dust_single_scattering_albedo = pyrt.regrid(scattering_cross_section / extinction_cross_section, particle_size_grid, wavelength_grid, particle_size_gradient, pixel_wavelengths)

dust_pmom = np.ones((128, 50, 20))
dust_legendre = pyrt.regrid(dust_pmom, particle_size_grid, wavelength_grid, particle_size_gradient, pixel_wavelengths)

dust_column = pyrt.Column(dust_optical_depth, dust_single_scattering_albedo, dust_legendre)

model = rayleigh_co2 + dust_column

DTAUC = model.optical_depth
SSALB = model.single_scattering_albedo
PMOM = model.legendre_coefficients

MAXCLY = len(altitude_midpoint)
MAXMOM = PMOM.shape[0] - 1
MAXCMU = 16      # AKA the number of streams
MAXPHI = 1
MAXUMU = 1
MAXULV = len(altitude_midpoint) + 1

ACCUR = 0.0
DELTAMPLUS = True
DO_PSEUDO_SPHERE = False
HEADER = ''
PRNT = [False, False, False, False, False]
EARTH_RADIUS = 6371

FBEAM = np.pi
FISOT = 0

PLANK = False
BTEMP = temperature_profile[-1]
TTEMP = temperature_profile[0]
TEMIS = 0

ALBMED = pyrt.empty_albedo_medium(MAXUMU)
FLUP = pyrt.empty_diffuse_up_flux(MAXULV)
RFLDN = pyrt.empty_diffuse_down_flux(MAXULV)
RFLDIR = pyrt.empty_direct_beam_flux(MAXULV)
DFDT = pyrt.empty_flux_divergence(MAXULV)
UU = pyrt.empty_intensity(MAXUMU, MAXULV, MAXPHI)
UAVG = pyrt.empty_mean_intensity(MAXULV)
TRNMED = pyrt.empty_transmissivity_medium(MAXUMU)

IBCND = False
ONLYFL = False
USRANG = True
USRTAU = False

ALBEDO = 0  # this seems to only matter if the surface is Lambertian
LAMBER = False

RHOQ, RHOU, EMUST, BEMST, RHO_ACCURATE = \
    pyrt.make_hapke_surface(USRANG, ONLYFL, MAXUMU, MAXPHI, MAXCMU, UMU, UMU0, PHI, PHI0, FBEAM, 200, 1, 0.06, 0.7)

UTAU = np.zeros((MAXULV,))

test_run = np.zeros(pixel_wavelengths.shape)

for ind in range(pixel_wavelengths.size):
    rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed = \
        disort.disort(USRANG, USRTAU, IBCND, ONLYFL, PRNT, PLANK, LAMBER,
                      DELTAMPLUS, DO_PSEUDO_SPHERE, DTAUC[:, ind], SSALB[:, ind],
                      PMOM[:, :, ind], TEMPER, WVNMLO, WVNMHI,
                      UTAU, UMU0, PHI0, UMU, PHI, FBEAM, FISOT,
                      ALBEDO, BTEMP, TTEMP, TEMIS, EARTH_RADIUS, H_LYR, RHOQ, RHOU,
                      RHO_ACCURATE, BEMST, EMUST, ACCUR, HEADER, RFLDIR,
                      RFLDN, FLUP, DFDT, UAVG, UU, ALBMED, TRNMED)

    test_run[ind] = uu[0, 0, 0]

print(test_run)

def simulate_spectra(test_optical_depth):
    dust_optical_depth = pyrt.optical_depth(dust_profile, column_density, ext, test_optical_depth)
    dust_column = pyrt.Column(dust_optical_depth, dust_single_scattering_albedo, dust_legendre)
    model = rayleigh_co2 + dust_column

    od_holder = np.zeros(pixel_wavelengths.shape)
    for wav_index in range(5):
        rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed = \
            disort.disort(USRANG, USRTAU, IBCND, ONLYFL, PRNT, PLANK, LAMBER,
                          DELTAMPLUS, DO_PSEUDO_SPHERE,
                          model.optical_depth[:, wav_index], model.single_scattering_albedo[:, wav_index],
                          model.legendre_coefficients[:, :, wav_index], TEMPER, WVNMLO, WVNMHI,
                          UTAU, UMU0, PHI0, UMU, PHI, FBEAM, FISOT,
                          ALBEDO, BTEMP, TTEMP, TEMIS, EARTH_RADIUS, H_LYR, RHOQ, RHOU,
                          RHO_ACCURATE, BEMST, EMUST, ACCUR, HEADER, RFLDIR,
                          RFLDN, FLUP, DFDT, UAVG, UU, ALBMED, TRNMED)

        od_holder[wav_index] = uu[0, 0, 0]
    return np.sum((od_holder - test_run) ** 2)


from scipy import optimize


def retrieve_od(guess):
    return optimize.minimize(simulate_spectra, np.array([guess]),
                             method='Nelder-Mead', bounds=((0, 2),)).x

import time

t0 = time.time()
print(retrieve_od(1.5))
t1 = time.time()
print(f'The retrieval took {(t1 - t0):.3f} seconds.')
