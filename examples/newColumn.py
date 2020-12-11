from pyRT_DISORT.preprocessing.model.aerosol_column import AerosolProperties, Column, SporadicParticleSizes
from pyRT_DISORT.preprocessing.utilities.external_files import ExternalFile
from pyRT_DISORT.preprocessing.model.atmosphere import ModelGrid
from pyRT_DISORT.preprocessing.model.vertical_profiles import Conrath
from pyRT_DISORT.preprocessing.model.new_phase_function import LegendreCoefficients, HenyeyGreenstein, TabularLegendreCoefficients

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

# Step 0.2: Make up some fake values
wavs = np.array([1, 9.3])               # Suppose these are the wavelengths you observed at


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example 1: The new column object using a Conrath profile
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Note I haven't changed any VerticalProfiles, so I'm kinda using Conrath clunkily to make 1D q profiles... this can be
# easily changed once we settle on a finalized design for Column
sh = np.array([10])
nu = np.array([0.5])
c = Conrath(model_grid, sh, nu)
q_profile = np.squeeze(c.profile)

# Make a smooth particle size gradient from 0.5 microns to 1.5 microns. It's much higher precision than in reality but
# I think you'll get the idea. I use 19 values since my atmosphere has 19 layers.
particle_sizes = np.linspace(0.5, 1.5, num=19)

# Make an AerosolProperties object to just hold the aerosol's properties
# The "['soemthing'].data" is just cause astropy is kinda clunky imo
# I designed my classes to handle 2D or 3D input, though I need to test the 2D input more. Regardless, the 2 commented
# out lines below show how it handles a 2D array with no spectral info, and a 2D array with no size info
prop = AerosolProperties(dustFile.array['primary'].data, wavelength_grid=dustFile.array['wavelengths'].data, particle_size_grid=dustFile.array['particle_sizes'].data)
#prop = AerosolProperties(dustFile.array['primary'].data[:, 0, :], particle_size_grid=dustFile.array['particle_sizes'].data)
#prop = AerosolProperties(dustFile.array['primary'].data[5, :, :], wavelength_grid=dustFile.array['wavelengths'].data)

# Also of note: instead of checking all inputs, I added a debug flag to run some tests. I don't claim that it's
# comprehensive, but it may be useful when messing with these classes to handle errors before they propagate further

# Phase function stuff
#pf = HenyeyGreenstein(n_moments=200)
idk = ExternalFile('/home/kyle/repos/pyRT_DISORT/pyRT_DISORT/data/planets/mars/aux/dust_phase_function.fits')
#pf = TabularLegendreCoefficients(idk.array['primary'].data, phase_function_wavelengths=idk.array['wavelengths'].data, phase_function_particle_sizes=idk.array['particle_sizes'].data)
pf = TabularLegendreCoefficients(idk.array['primary'].data[:, 0, 0])

# Make a new Column instance
newcol = Column(prop, model_grid, q_profile, particle_sizes, wavs, 9.3, 1, pf)   # 9.3 = wavelength reference; 1 = column integrated OD

# Column contains 2D arrays of the radiative properties. Before, I passed wavelengths and particle sizes to Aerosol,
# but this redesign rendered Aerosol obsolete (at least in my mind). So these properties are now all (n_layers, n_wavelengths)
#print(newcol.c_extinction)
#print(newcol.c_scattering)
#print(newcol.g)

# Assuming I did everything with q correctly (which wasn't much at all...) then you can get the correct ODs in the
# Column attributes. These at least match old versions of my code and disort_multi. Also, I removed
# all attributes involving "multisize" since I don't think they're useful anymore.
#print(newcol.total_optical_depth)
#print(np.sum(newcol.total_optical_depth, axis=0))
#print(newcol.scatting_optical_depth)

print(newcol.phase_function.shape)
#print(np.amax(newcol.phase_function))
raise SystemExit(9)

'''# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example 2: Similar to example 1 but you don't know the particle sizes as well
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make a q_profile again
sh = np.array([10])
nu = np.array([0.5])
c = Conrath(model_grid, sh, nu)
q_profile = np.squeeze(c.profile)

# This time, you don't know the particle sizes in all layers so I made a helper class. This essentially handles your
# request to use an Nx2 array by turning it into input that I expect. You can request a better class name if you have ideas...
spor = np.array([[50, 0.8], [40, 1], [30, 1.2], [20, 1.4]])  # Make the 2D array of particle sizes and their associated altitudes
s = SporadicParticleSizes(spor, model_grid, debug=True)   # Make the object
particle_sizes = s.regridded_particle_sizes  # The class interpolates sizes onto the grid you provided it in model_grid

# Make an AerosolProperties object to just hold the aerosol's properties
# The "['soemthing'].data" is just cause astropy is kinda clunky imo
# I designed my classes to handle 2D or 3D input, though I need to test the 2D input more. Regardless, the 2 commented
# out lines below show how it handles a 2D array with no spectral info, and a 2D array with no size info
prop = AerosolProperties(dustFile.array['primary'].data, wavelength_grid=dustFile.array['wavelengths'].data, particle_size_grid=dustFile.array['particle_sizes'].data)
#prop = AerosolProperties(dustFile.array['primary'].data[:, 0, :], particle_size_grid=dustFile.array['particle_sizes'].data)
#prop = AerosolProperties(dustFile.array['primary'].data[5, :, :], wavelength_grid=dustFile.array['wavelengths'].data)

# Make a new Column instance
newcol = Column(prop, model_grid, q_profile, particle_sizes, wavs, 9.3, 1)   # 9.3 = wavelength reference; 1 = column integrated OD

# Column contains 2D arrays of the radiative properties. Before, I passed wavelengths and particle sizes to Aerosol,
# but this redesign rendered Aerosol obsolete (at least in my mind). So these properties are now all (n_layers, n_wavelengths)
print(newcol.c_extinction)
print(newcol.c_scattering)
print(newcol.g)

# Assuming I did everything with q correctly (which wasn't much at all...) then you can get the correct ODs in the
# Column attributes. These at least match old versions of my code and disort_multi. Also, I removed
# all attributes involving "multisize" since I don't think they're useful anymore.
print(newcol.total_optical_depth)
print(np.sum(newcol.total_optical_depth, axis=0))
print(newcol.scatting_optical_depth)

'''