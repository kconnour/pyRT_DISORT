import numpy as np
from pyRT_DISORT.preprocessing.utilities.external_files import MultipleExternalFiles, ExternalFile
from pyRT_DISORT.preprocessing.utilities.create_fits import CreateFits

# This example will read in the .phsfn files for ice within the folder /home/kyle/disort_multi/phsfn/
# On my computer that folder contains .dat, .coef, and .phsfn files for dust along with just .phsfn files for ice
# It's mainly just to show you how to use my classes to create multidimensional arrays

f = MultipleExternalFiles('*v2*', '/home/kyle/dustFiles')   # Change this to work with your system
# print(f.files)    # print out the absolute paths of all the files, if desired

# Construct 1D arrays of the wavelengths and radii corresponding to these files. I'll work on a better solution
f0 = ExternalFile('/home/kyle/dustFiles/tmq_mod1_r001v030_forw_v2.dat', text1d=False, header_lines=3)
wavs = f0.array[:, 0]
radii = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1, 1.5, 2, 2.5, 3])

dust = np.zeros((len(radii), len(wavs), 3))            # shape: 14, 331, 3
print(dust.shape)

# Fill in the empty array with all the values from the files
radius_index = 0
for counter, file in enumerate(f.files):
    # Start by reading in file as a numpy array
    ext_file = ExternalFile(file, header_lines=3, text1d=False)

    dust[counter, :, :] = ext_file.array[:, 1:]


asdf = CreateFits(dust)
asdf.add_image_hdu(radii, 'particle_sizes')
asdf.add_image_hdu(wavs, 'wavelengths')
asdf.save_fits('/home/kyle/repos/pyRT_DISORT/pyRT_DISORT/data/planets/mars/aux/dust_properties.fits')
