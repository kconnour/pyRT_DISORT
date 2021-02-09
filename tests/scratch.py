#import disort

#print(disort.disobrdf.__doc__)

import numpy as np
from scipy.constants import Boltzmann
a = np.genfromtxt('/home/kyle/disort_multi/marsatm.inp', skip_header=3)
print(a.shape)

atm = np.zeros((41, 4))
atm[:, 0] = a[:, 0]
atm[:, 1] = a[:, 1] * 100
atm[:, 2] = a[:, 2]
atm[:, 3] = a[:, 1] * 100 / Boltzmann / a[:, 2]

np.save('/home/kyle/repos/pyRT_DISORT/tests/marsatm.npy', atm)
