import numpy as np
from scipy.constants import Boltzmann

'''atmfile = '/home/kyle/disort_multi/marsatm.inp'

f = np.genfromtxt(atmfile, skip_header=3)
newfile = np.zeros((f.shape[0], 4))
newfile[:, :-1] = f
newfile[:, 1] *= 100


for i in range(f.shape[0]):
    newfile[i, -1] = newfile[i, 1] / Boltzmann / newfile[i, 2]

newfile = np.flipud(newfile)
print(newfile[:, 0])
print(newfile[:, 1])
print(newfile[:, 2])
print(newfile[:, 3])
np.save('/home/kyle/repos/pyRT_DISORT/pyRT_DISORT/data/planets/mars/aux/mars_atm_copy.npy', newfile)'''


ice = np.load('/home/kyle/ice_phsfns.npy')
print(ice.shape)
