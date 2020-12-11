from pyRT_DISORT.preprocessing.model.aerosol import ForwardScatteringProperty, ForwardScatteringProperties
from pyRT_DISORT.preprocessing.utilities.external_files import ExternalFile
from pyRT_DISORT.preprocessing.model.atmosphere import ModelGrid
from pyRT_DISORT.preprocessing.model.aerosol_column import Column
from pyRT_DISORT.preprocessing.model.vertical_profiles import Conrath
from pyRT_DISORT.preprocessing.model.phase_function import LegendreCoefficients, HenyeyGreenstein, TabularLegendreCoefficients
from pyRT_DISORT.data.get_data import get_data_path
import os
import numpy as np

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example 1: Read in fully 3D files
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c_ext = ForwardScatteringProperty(dustFile.array['primary'].data[:, :, 0], particle_size_grid=sizes, wavelength_grid=wavs)
c_sca = ForwardScatteringProperty(dustFile.array['primary'].data[:, :, 1], particle_size_grid=sizes, wavelength_grid=wavs)
# Note 1: The files you sent me had the properties defined on the same grid, which seems natural. But what I've done so
# far can handle the case of each property having a different grid
dust_properties = ForwardScatteringProperties(c_sca, c_ext)
# Note 2: I decided it's most flexible to put each forward scattering property into its own object, then add all of
# these to a container object. This way, the user doesn't have to ensure the columns are in the correct order. They
# also don't need g if not using a HG phase function. Additionally, if there's another analytic phase function they want
# to use that's not HG, it's extremely simple for FSPs to handle it.

# Make a dust conrath profile. This is still clunky until I'm convinced Column does what you want
sh = np.array([10])
nu = np.array([0.5])
c = Conrath(model_grid, sh, nu)
q_profile = np.squeeze(c.profile)

# Define particle sizes and wavelengths
p_sizes = np.linspace(0.5, 1.5, num=len(q_profile))
wavelengths = np.array([1, 9.3])

# Make a phase function
dust_phsfn_file = ExternalFile(os.path.join(get_data_path(), 'planets/mars/aux/dust_phase_function.fits'))
# The coefficients I have are often negative, so set the negative ones to 10**-6
t = np.where(dust_phsfn_file.array['primary'].data <= 0, 10**-6, dust_phsfn_file.array['primary'].data)
dust_phsfn = TabularLegendreCoefficients(t, dust_phsfn_file.array['particle_sizes'].data, dust_phsfn_file.array['wavelengths'].data)

# Make the new Column
dust_col = Column(dust_properties, model_grid, q_profile, p_sizes, wavelengths, 9.3, 1, dust_phsfn)
# Then you can access total_optical_depth, scattering_optical_depth, and phase_function as attributes

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example 2: Same as before but you only believe the first 16 moments
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c_ext = ForwardScatteringProperty(dustFile.array['primary'].data[:, :, 0], particle_size_grid=sizes, wavelength_grid=wavs)
c_sca = ForwardScatteringProperty(dustFile.array['primary'].data[:, :, 1], particle_size_grid=sizes, wavelength_grid=wavs)
# Note 1: The files you sent me had the properties defined on the same grid, which seems natural. But what I've done so
# far can handle the case of each property having a different grid
dust_properties = ForwardScatteringProperties(c_sca, c_ext)
# Note 2: I decided it's most flexible to put each forward scattering property into its own object, then add all of
# these to a container object. This way, the user doesn't have to ensure the columns are in the correct order. They
# also don't need g if not using a HG phase function. Additionally, if there's another analytic phase function they want
# to use that's not HG, it's extremely simple for FSPs to handle it.

# Make a dust conrath profile. This is still clunky until I'm convinced Column does what you want
sh = np.array([10])
nu = np.array([0.5])
c = Conrath(model_grid, sh, nu)
q_profile = np.squeeze(c.profile)

# Define particle sizes and wavelengths
p_sizes = np.linspace(0.5, 1.5, num=len(q_profile))
wavelengths = np.array([1, 9.3])

# Make a phase function
dust_phsfn_file = ExternalFile(os.path.join(get_data_path(), 'planets/mars/aux/dust_phase_function.fits'))
# The coefficients I have are often negative, so set the negative ones to 10**-6
t = np.where(dust_phsfn_file.array['primary'].data <= 0, 10**-6, dust_phsfn_file.array['primary'].data)
dust_phsfn = TabularLegendreCoefficients(t, dust_phsfn_file.array['particle_sizes'].data, dust_phsfn_file.array['wavelengths'].data, max_moments=16)

# Make the new Column
dust_col_trimmed = Column(dust_properties, model_grid, q_profile, p_sizes, wavelengths, 9.3, 1, dust_phsfn)
print(np.array_equal(dust_col_trimmed.phase_function, dust_col.phase_function[:16, :, :]))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example 3: Use a HG phase function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c_ext = ForwardScatteringProperty(dustFile.array['primary'].data[:, :, 0], particle_size_grid=sizes, wavelength_grid=wavs)
c_sca = ForwardScatteringProperty(dustFile.array['primary'].data[:, :, 1], particle_size_grid=sizes, wavelength_grid=wavs)
g = ForwardScatteringProperty(dustFile.array['primary'].data[:, :, 2], particle_size_grid=sizes, wavelength_grid=wavs)
dust_properties = ForwardScatteringProperties(c_sca, c_ext, g=g)
# Note 2: I decided it's most flexible to put each forward scattering property into its own object, then add all of
# these to a container object. This way, the user doesn't have to ensure the columns are in the correct order. They
# also don't need g if not using a HG phase function. Additionally, if there's another analytic phase function they want
# to use that's not HG, it's extremely simple for FSPs to handle it.

# Make a dust conrath profile. This is still clunky until I'm convinced Column does what you want
sh = np.array([10])
nu = np.array([0.5])
c = Conrath(model_grid, sh, nu)
q_profile = np.squeeze(c.profile)

# Define particle sizes and wavelengths
p_sizes = np.linspace(0.5, 1.5, num=len(q_profile))
wavelengths = np.array([1, 9.3])

# Make a phase function
dust_phsfn = HenyeyGreenstein(n_moments=128)   # Make a HG phase function with 128 moments, default is 200

# Make the new Column
dust_col = Column(dust_properties, model_grid, q_profile, p_sizes, wavelengths, 9.3, 1, dust_phsfn)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example 4: exoplanet where you know basically nothing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c_ext = ForwardScatteringProperty(np.array([0.7]))   # c_ext has no size/spectral info
c_sca = ForwardScatteringProperty(np.array([0.8]))      # c_sca has no size/spectral info
dust_properties = ForwardScatteringProperties(c_sca, c_ext)

# Make a dust conrath profile. This is still clunky until I'm convinced Column does what you want
sh = np.array([10])
nu = np.array([0.5])
c = Conrath(model_grid, sh, nu)
q_profile = np.squeeze(c.profile)

# Define particle sizes and wavelengths
p_sizes = np.linspace(0.5, 1.5, num=len(q_profile))
wavelengths = np.array([1, 9.3])

# Suppose you have 20 moments and no size/spectral info
dust_phsfn = TabularLegendreCoefficients(dust_phsfn_file.array['primary'].data[:20, 0, 0])

# Make the new Column
dust_col = Column(dust_properties, model_grid, q_profile, p_sizes, wavelengths, 9.3, 1, dust_phsfn)
print(dust_col.phase_function.shape)
