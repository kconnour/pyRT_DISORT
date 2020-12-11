import os
import numpy as np

# Local imports
from pyRT_DISORT.model_atmosphere.aerosol import ForwardScatteringProperty, ForwardScatteringPropertyCollection
from pyRT_DISORT.model_atmosphere.atmosphere_grid import ModelGrid
from pyRT_DISORT.utilities.external_files import ExternalFile
from data.get_data import get_data_path

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Preprocessing steps
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read in the atmosphere file
atmFile = ExternalFile(os.path.join(get_data_path(), 'planets/mars/aux/mars_atm.npy'))
z_boundaries = np.linspace(80, 0, num=20)    # Define the boundaries I want to use. Note that I'm sticking with DISORT's convention of starting from TOA
model_grid = ModelGrid(atmFile.array, z_boundaries)

# Read in an aerosol file---here, dust
# Note that I combined all your files in a new .fits file so primary is 3D
dustFile = ExternalFile(os.path.join(get_data_path(), 'planets/mars/aux/dust_properties.fits'))
wavs = dustFile.array['wavelengths'].data
sizes = dustFile.array['particle_sizes'].data

c_ext = ForwardScatteringProperty(dustFile.array['primary'].data[:, :, 0], particle_size_grid=sizes, wavelength_grid=wavs)
c_sca = ForwardScatteringProperty(dustFile.array['primary'].data[:, :, 1], particle_size_grid=sizes, wavelength_grid=wavs)

fsp = ForwardScatteringPropertyCollection()
fsp.add_property(c_ext, 'c_ext')
fsp.add_property(c_sca, 'c_sca')

print(fsp.c_sca.wavelength_grid)

