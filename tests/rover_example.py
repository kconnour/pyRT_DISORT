import numpy as np

incidence = 30
beam_azimuth = 25
emission = np.linspace(130, 160, num=15)
azimuth = np.linspace(30, 40, num=20)

from pyRT_DISORT.observation import sky_image

angles = sky_image(incidence, emission, azimuth, beam_azimuth)

incidence = angles.incidence
emission = angles.emission
mu = angles.mu
mu0 = angles.mu0
phi = angles.phi
phi0 = angles.phi0

UMU = mu[0, :]
UMU0 = mu0[0]
PHI = phi[0, :]
PHI0 = phi0[0]


