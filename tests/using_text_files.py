import numpy as np
import time
from pyRT_DISORT.preprocessing.utilities.utilities import MultipleExternalFiles, ExternalFile

# This example will read in the .phsfn files for ice within the folder /home/kyle/disort_multi/phsfn/
# On my computer that folder contains .dat, .coef, and .phsfn files for dust along with just .phsfn files for ice
# It's mainly just to show you how to use my classes to create multidimensional arrays

f = MultipleExternalFiles('ice*.phsfn', '/home/kyle/disort_multi/phsfn')   # Change this to work with your system
# print(f.files)    # print out the absolute paths of all the files, if desired

# Construct 1D arrays of the wavelengths and radii corresponding to these files. I'll work on a better solution
# for this sort of thing later but for now it's what I've got
short_wavs = [0.2, 0.255, 0.3, 0.336, 0.4, 0.5, 0.588, 0.6, 0.631, 0.673, 0.7, 0.763, 0.8, 0.835, 0.9, 0.953]
wavs = np.linspace(1, 50, num=(50-1)*10+1)
wavs = np.concatenate((short_wavs, wavs))
radii = np.array([1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 80])

# Each .phsfn file is 2D and it's dependent on radius and wavelength so I need a 4D array to hold all their values
# I'll keep the 2 dimensions from the file, then add radius and wavelength as the 3rd and 4th dim, respectively
# Assuming each file is the same length, get one for its array shape
junk_file = ExternalFile(f.files[0], header_lines=7, text1d=False)     # False since it's 2D
ice_phsfns = np.zeros((junk_file.array.shape[0], junk_file.array.shape[1], len(radii), len(wavs)))  # shape: (181, 6, 11, 507)

# Fill in the empty array with all the values from the files
#t0 = time.time()
radius_index = 0
for counter, file in enumerate(f.files):
    # Start by reading in file as a numpy array
    ext_file = ExternalFile(file, header_lines=7, text1d=False)

    # Be somewhat clever to put the array into the proper indices
    # If you aren't familiar, % is Python's mod operator
    # This only works because a sorted list of your files increments by wavelength
    ice_phsfns[:, :, radius_index, counter % len(wavs)] = ext_file.array
    if ((counter + 1) % len(wavs)) == 0:
        radius_index += 1

# If you want to save the array you just created, uncomment this line (but change it to your system)
# np.save('/home/kyle/ice_phsfns.npy', ice_phsfns)
#t1 = time.time()
#print(t1 - t0)

''' Comments
--- This takes ~4.5 seconds on my computer to make + populate the numpy array, so I wouldn't recommend doing this
    if you want to use this code for a retrieval where it'd need to be run many times
--- This was easy enough to write because I know how the files are ordered in relation to the array I want. In other
    words, if a sorted list of files incremented by particle size instead of wavelength I'd need to change my indices.
    Also, I knew I wanted particle size and wavelength to be the 3rd and 4th dimension, but I see no reason why the 
    user may hypothetically want them to be any of the indices, which is the main reason I think I cannot make this
    function "standard"---only provide tools so they can do it themselves.

'''