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
