from pyRT_DISORT.preprocessing.model.aerosol import ForwardScatteringProperty, ForwardScatteringProperties
from pyRT_DISORT.preprocessing.utilities.external_files import ExternalFile
from pyRT_DISORT.preprocessing.model.atmosphere import ModelGrid
#from pyRT_DISORT.preprocessing.model.vertical_profiles import Conrath
#from pyRT_DISORT.preprocessing.model.new_phase_function import LegendreCoefficients, HenyeyGreenstein, TabularLegendreCoefficients

import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Preprocessing steps
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step 0.0: Read in the atmosphere file
atmFile = ExternalFile('/home/kyle/repos/pyRT_DISORT/pyRT_DISORT/data/planets/mars/aux/mars_atm.npy')
z_boundaries = np.linspace(80, 0, num=20)    # Define the boundaries I want to use. Note that I'm sticking with DISORT's convention of starting from TOA
model_grid = ModelGrid(atmFile.array, z_boundaries)

# Step 0.1: Read in an aerosol file---here, dust
# Note that I combined all your files in a new .fits file so primary is 3D
dustFile = ExternalFile('/home/kyle/repos/pyRT_DISORT/pyRT_DISORT/data/planets/mars/aux/dust_properties.fits')
#print(dustFile.array['primary'].shape)
wavs = dustFile.array['wavelengths'].data
sizes = dustFile.array['particle_sizes'].data

c_ext = ForwardScatteringProperty(dustFile.array['primary'].data[:, :, 0], wavelength_grid=wavs, particle_size_grid=sizes)
c_sca = ForwardScatteringProperty(dustFile.array['primary'].data[:, :, 1], wavelength_grid=wavs, particle_size_grid=sizes)
dust_properties = ForwardScatteringProperties(c_sca, c_ext)
print(dust_properties.c_scattering.property_values)
print(dust_properties.c_extinction.property_values.shape)
print(dust_properties.g)
