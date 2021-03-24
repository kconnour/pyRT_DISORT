import numpy as np

from astropy.io import fits





f = '/home/kyle/repos/pyRT_DISORT/tests/aux/dust_properties.fits'
hdul = fits.open(f)
cext = hdul['primary'].data[:, :, 0]
csca = hdul['primary'].data[:, :, 1]
wavs = hdul['wavelengths'].data
psizes = hdul['particle_sizes'].data

pgrad = np.linspace(1, 1.5, num=10)
wavelengths = np.array([1, 2, 3, 4, 5])
#cext = np.broadcast_to(cext, (65,) + cext.shape)
print(cext.shape)
nni = NearestNeighborInterpolator(cext, psizes, wavs, pgrad, wavelengths)
print(nni.coeff.shape)

#import disort
#print(disort.disort.__doc__)
