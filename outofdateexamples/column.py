# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.preprocessing.model.atmosphere import ModelGrid
from pyRT_DISORT.preprocessing.utilities.external_files import ExternalFile
from pyRT_DISORT.preprocessing.model.aerosol_column import Column
from pyRT_DISORT.preprocessing.model.aerosol import Aerosol
from pyRT_DISORT.preprocessing.model.vertical_profiles import Conrath, Uniform, Layers, ProfileHolder

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Preprocessing steps
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step 0.0: Read in the atmosphere file
atmFile = ExternalFile('/home/kyle/repos/pyRT_DISORT/pyRT_DISORT/data/planets/mars/aux/mars_atm.npy')
z_boundaries = np.linspace(80, 0, num=20)    # Define the boundaries I want to use. Note that I'm sticking with DISORT's convention of starting from TOA
model_grid = ModelGrid(atmFile.array, z_boundaries)

# Step 0.1: Read in an aerosol file---here, dust
dustFile = ExternalFile('/home/kyle/repos/pyRT_DISORT/pyRT_DISORT/data/planets/mars/aux/dust_properties.fits')

# Step 0.2: Make up some fake values
radii = np.array([1, 1.2, 1.4])         # Suppose these are the particle sizes you want in the model
wavs = np.array([1, 9.3])               # Suppose these are the wavelengths you observed at
wavref = np.array([12.1, 12.1, 12.1])   # Suppose you want to scale everything to 12.1 microns. Note these can be whatever numbers you want---they don't all have to be the same

# Step 0.3: Make an aerosol
# Note that I haven't put the files you sent me into a 3D array, so I'm using a file with no particle size info
aero = Aerosol(dustFile.array['primary'].data, radii, wavs, wavelength_grid=dustFile.array['wavelengths'].data,
               reference_wavelengths=wavref)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example 1: Recreate profiles from an old example assuming dust has a Conrath distribution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sh = np.array([10, 10, 10])        # each particle size has the same scale height
nu = np.array([0.5, 0.5, 0.5])     # each particle size has the same nu
c = Conrath(model_grid, sh, nu)    # Make a Conrath profile for all 3 particle sizes at the same time
p = ProfileHolder()                # Make a "blank" object to just old all the profiles I give it
p.add_profile(c.profile)           # Add the Nx3 array of vertical profiles to the object
p.stack_profiles()                 # Combine all the arrays into an Mx3 array. Here, M=N but hopefully later outofdateexamples will make this step more clear

# Define optical depths at each of the particle sizes
ods = np.array([0.1, 0.8, 0.2])
dust_column = Column(aero, model_grid, p.profile, ods)
# Some print statements you may or may not want...
#print(dust_column.multisize_total_optical_depth)
#print(dust_column.total_optical_depth)
#print(np.sum(dust_column.total_optical_depth, axis=0))
#print(dust_column.scattering_optical_depth)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example 2: Recreate a GCM, where you "know" the difference between particle sizes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sh = np.array([15, 10, 9.1])       # each particle size may have a different scale height
nu = np.array([0.2, 0.5, 0.1])     # each particle size may have a different nu
c = Conrath(model_grid, sh, nu)    # Make a Conrath profile for all 3 particle sizes at the same time
p = ProfileHolder()                # Make a "blank" object to just old all the profiles I give it
p.add_profile(c.profile)           # Add the Nx3 array of vertical profiles to the object
p.stack_profiles()                 # Combine all the arrays into an Mx3 array. Here, M=N but hopefully later outofdateexamples will make this step more clear

# Define optical depths at each of the particle sizes
ods = np.array([0.1, 0.8, 0.2])
dust_column = Column(aero, model_grid, p.profile, ods)
# Some print statements you may or may not want...
#print(dust_column.multisize_total_optical_depth)
#print(dust_column.total_optical_depth)
#print(np.sum(dust_column.total_optical_depth, axis=0))
#print(dust_column.scattering_optical_depth)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example 3: Recreate global dust storm (also, I think this will recreate the disort_multi water ice vertical profile)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Instead of Conrath, I'll make a uniform profile
z_bottom = np.array([15, 10, 5])   # Set the top and bottom altitudes for each particle size
z_top = np.array([55, 40, 39])     # So here, 1 micron dust has constant mixing ratio between 15--55 km, and so on
uniform = Uniform(model_grid, z_bottom, z_top)    # Make 3 uniform profiles all at once
p = ProfileHolder()
p.add_profile(uniform.profile)
p.stack_profiles()

# Define optical depths at each of the particle sizes
ods = np.array([0.1, 0.8, 0.2])
dust_column = Column(aero, model_grid, p.profile, ods)
# Some print statements you may or may not want...
#print(dust_column.multisize_total_optical_depth)
#print(dust_column.total_optical_depth)
#print(np.sum(dust_column.total_optical_depth, axis=0))
#print(dust_column.scattering_optical_depth)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example 4: Combine multiple profile types (maybe this isn't all that necessary on Mars but I could see someone wanting it)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Suppose someone believes 1 micron dust follows a Conrath profile but 1.2 and 1.4 micron dust has a constant shape
# Make the Conrath distribution first
sh = np.array([10])
nu = np.array([0.5])
c = Conrath(model_grid, sh, nu)    # c.profile is Nx1

# Make the blank object to hold profiles and add the Conrath profile
p = ProfileHolder()
p.add_profile(c.profile)

# Now make the uniform distributions
z_bottom = np.array([15, 10])   # 1.2 and 1.4 micron dust have constant q between these altitudes
z_top = np.array([55, 40])
uniform = Uniform(model_grid, z_bottom, z_top)     # uniform.profile is Nx2

# Now we can add different profiles to the ProfileHolder object
p.add_profile(uniform.profile)
p.stack_profiles()                   # This creates an Nx3 array in the p.profile attribute

# Define optical depths at each of the particle sizes
ods = np.array([0.1, 0.8, 0.2])
dust_column = Column(aero, model_grid, p.profile, ods)
# Some print statements you may or may not want...
#print(dust_column.multisize_total_optical_depth)
#print(dust_column.total_optical_depth)
#print(np.sum(dust_column.total_optical_depth, axis=0))
#print(dust_column.scattering_optical_depth)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example 5: Associate layers with the particle sizes (what I think you're asking for)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Suppose that you think the lowest 2 layer contain 1.4 micron dust, the next 2 layers contain 1.2 micron dust, and the 4 after that contain 1 micron dust
# This is a little clunkier because I want to make sure this is what you want before spending more time on it---but it'd be very easy to make it as slick as the other VerticalProfile classes
l0 = Layers(model_grid, np.array([-1, -2]))    # Negative indices since altitudes start from TOA
l1 = Layers(model_grid, np.array([-3, -4]))
l2 = Layers(model_grid, np.array([-5, -6, -7, -8]))

# Combine these profiles together in the correct order (the column numbers should match the particle sizes, so p.profile[:, 0] should be the profile for particle_sizes[0])
p = ProfileHolder()
p.add_profile(l2.profile)
p.add_profile(l1.profile)
p.add_profile(l0.profile)
p.stack_profiles()
#print(p.profile)

# Define optical depths at each of the particle sizes
ods = np.array([0.1, 0.8, 0.2])
dust_column = Column(aero, model_grid, p.profile, ods)
# Some print statements you may or may not want...
#print(dust_column.multisize_total_optical_depth)
#print(dust_column.total_optical_depth)
#print(np.sum(dust_column.total_optical_depth, axis=0))
#print(dust_column.scattering_optical_depth)