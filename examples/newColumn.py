from pyRT_DISORT.preprocessing.model.aerosol_column import AerosolProperties, Column, SporadicParticleSizes
from pyRT_DISORT.preprocessing.utilities.external_files import MultipleExternalFiles, ExternalFile
from pyRT_DISORT.preprocessing.model.atmosphere import ModelGrid
from pyRT_DISORT.preprocessing.model.vertical_profiles import Conrath, Uniform, Layers, ProfileHolder

import numpy as np



f = ExternalFile('/home/kyle/repos/pyRT_DISORT/pyRT_DISORT/data/planets/mars/aux/dust_properties.fits')

atmFile = ExternalFile('/home/kyle/repos/pyRT_DISORT/pyRT_DISORT/data/planets/mars/aux/mars_atm.npy')
z_boundaries = np.linspace(80, 0, num=20)    # Define the boundaries I want to use. Note that I'm sticking with DISORT's convention of starting from TOA
model_grid = ModelGrid(atmFile.array, z_boundaries)

sh = np.array([10, 10, 10])        # each particle size has the same scale height
nu = np.array([0.5, 0.5, 0.5])     # each particle size has the same nu
c = Conrath(model_grid, sh, nu)    # Make a Conrath profile for all 3 particle sizes at the same time
c = c.profile[:, 0]
#particle_sizes = np.ones(19)*1.5   # Note: the old aerosol_dust.dat was for 1.5 microns
particle_sizes = np.linspace(0.1, 1, num=19)
w = np.array([1, 9.3])

#prop = AerosolProperties(f.array['primary'].data, wavelength_grid=f.array['wavelengths'].data, particle_size_grid=f.array['particle_sizes'].data)
prop = AerosolProperties(f.array['primary'].data[:, 0, :], particle_size_grid=f.array['particle_sizes'].data)
#prop = AerosolProperties(f.array['primary'].data[0, :, :], wavelength_grid=f.array['wavelengths'].data)
newcol = Column(prop, model_grid, c, particle_sizes, w, 9.3, 1)
print(newcol.total_optical_depth)

#print(prop.c_extinction_grid[5, 17:113])
#print(newcol.model_grid.column_density_layers.shape)
#print(newcol.mixing_ratio_profile.shape)
#print(newcol.extinction_profile)
#print(np.sum(newcol.total_optical_depth, axis=0))
#print(newcol.total_optical_depth)
#print(np.sum(newcol.scatting_optical_depth, axis=0))

spor = np.array([[50, 1], [30, 1.2], [10, 1.4], [5, 1.6]])
s = SporadicParticleSizes(spor, model_grid)
print(s.regridded_particle_sizes)
