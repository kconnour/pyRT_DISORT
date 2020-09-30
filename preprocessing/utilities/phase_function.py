import numpy as np
import os
import fnmatch as fnm


def turn_tmqphsfn_to_npy(file_path):
    """ Turn a tmq_mod1_r###v030_#####.dat.coef to a numpy array

    Parameters
    ----------
    file_path: str
        The Unix-like path to the file

    Returns
    -------
    np.ndarray
        The coefficients stored in a numpy array
    """
    coeff = []
    f = open(file_path)
    lines = f.readlines()[1:]
    for line in lines:
        a = np.fromstring(line.strip(), sep=' ')
        coeff.append(a)

    # This unravels a list and stores it as an array
    return np.array([co for all_coeff in coeff for co in all_coeff])


def nearest_index(value, array):
    return np.where(np.amin(np.abs(array - value)) == np.abs(array - value))[0][0]


def find_all(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnm.fnmatch(name, pattern):
                result.append(os.path.join(root, name))

    return sorted(result)


effective_radius = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1, 1.5, 2, 2.5, 3])
wavelengths = np.load('/planets/mars/aux/dust.npy')[:, 0]

phsfns = np.zeros((len(effective_radius), len(wavelengths), 65))

#files = find_all('*.coef', '/home/kyle/disort_multi/phsfn/')
#print(files)

for counter, r in enumerate(effective_radius):
    print(r)
    files = find_all('*{}v030*.coef'.format(str(int(r*10)).zfill(3)), '/home/kyle/disort_multi/phsfn/')
    for fcounter, file in enumerate(files):
        phsfns[counter, fcounter, :] = turn_tmqphsfn_to_npy(file)

np.save('/planets/mars/aux/phase_functions.npy', phsfns)


