import numpy as np

'''from pyRT_DISORT.preprocessing.utilities.external_files import MultipleExternalFiles, ExternalFile
from pyRT_DISORT.preprocessing.utilities.fit_phase_function import PhaseFunction

# Read in the files
f = MultipleExternalFiles('ice*.phsfn', '/home/kyle/disort_multi/phsfn')   # Change this to work with your system
short_wavs = [0.2, 0.255, 0.3, 0.336, 0.4, 0.5, 0.588, 0.6, 0.631, 0.673, 0.7, 0.763, 0.8, 0.835, 0.9, 0.953]
wavs = np.linspace(1, 50, num=(50-1)*10+1)
wavs = np.concatenate((short_wavs, wavs))
radii = np.array([1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 80])
junk_file = ExternalFile(f.files[0], header_lines=7, text1d=False)     # False since it's 2D
ice_phsfns = np.zeros((junk_file.array.shape[0], junk_file.array.shape[1], len(radii), len(wavs)))  # shape: (181, 6, 11, 507)

radius_index = 0
for counter, file in enumerate(f.files):
    ext_file = ExternalFile(file, header_lines=7, text1d=False)
    ice_phsfns[:, :, radius_index, counter % len(wavs)] = ext_file.array
    if ((counter + 1) % len(wavs)) == 0:
        radius_index += 1

print(ice_phsfns.shape)
# Use the equation to make s12 into f11
degrees = ice_phsfns[:, 0, 0, 0]
f11 = ice_phsfns[:, 1, :, :]

a = np.zeros((128, 11, 507))
for i in range(11):
    print(i)
    for j in range(507):
        print(j)
        ice_phase_function = np.column_stack((degrees, np.squeeze(f11[:, i, j])))
        pf = PhaseFunction(ice_phase_function)
        a[:, i, j] = pf.create_legendre_coefficients(n_moments=128, n_samples=361)

np.save('/home/kyle/ice_phase_function.npy', a)'''


dustfile = '/home/kyle/repos/pyRT_DISORT/pyRT_DISORT/data/planets/mars/aux/dust.npy'
a = np.load(dustfile)
print(a[:, 0])

