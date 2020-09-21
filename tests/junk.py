import numpy as np

effective_radius = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1, 1.5, 2, 2.5, 3])
wavelengths = np.load('/home/kyle/repos/pyRT_DISORT/preprocessing/planets/mars/aux/dust.npy')[:, 0]
phs = np.load('/home/kyle/repos/pyRT_DISORT/preprocessing/planets/mars/aux/phase_functions.npy')

p = np.swapaxes(phs, -1, 0)
np.save('/home/kyle/repos/pyRT_DISORT/preprocessing/planets/mars/aux/phase_functions.npy', p)