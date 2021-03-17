import numpy as np


def get_nearest_indices(array, values):
    #diff = (values.reshape(1, -1) - array.reshape(-1, 1))
    #indices = np.abs(diff).argmin(axis=0)
    #return indices
    indices = np.abs(np.subtract.outer(array, values)).argmin(0)
    return indices

wavgrid = np.linspace(1, 10, num=11)
#print(wavgrid)
wavs = np.array([[3, 4], [7, 8]])

inds = get_nearest_indices(wavgrid, wavs)
#print(wavgrid[inds])

import disort
print(disort.disort.__doc__)
