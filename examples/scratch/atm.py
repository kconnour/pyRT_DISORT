# 3rd-party imports
import numpy as np
from scipy.constants import Boltzmann

from pyRT_DISORT.preprocessing.model.atmosphere import ModelAtmosphere, Layers
from pyRT_DISORT.preprocessing.utilities.external_files import ExternalFile

f = ExternalFile('/home/kyle/repos/pyRT_DISORT/pyRT_DISORT/data/planets/mars/aux/mars_atm.npy')
lay = Layers(f.array)
#print(lay.pressure_layers)

print(lay.column_density_layers)

#print(f.array)

ma = ModelAtmosphere(f.array, f.array[:, 0])
#print(ma.pressure_grid)
print(ma.column_density_layers)
