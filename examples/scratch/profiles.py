import numpy as np
from pyRT_DISORT.preprocessing.model.atmosphere import ModelAtmosphere
from pyRT_DISORT.preprocessing.utilities.external_files import ExternalFile
from pyRT_DISORT.preprocessing.model.vertical_profiles import Conrath, Uniform, Layers, ProfileHolder

f = ExternalFile('/home/kyle/repos/pyRT_DISORT/pyRT_DISORT/data/planets/mars/aux/mars_atm.npy')

ma = ModelAtmosphere(f.array, f.array[:, 0])

sh = np.array([9, 10, 11])
nu = np.array([0.5, 0.5, 0.7])

c = Conrath(ma, sh, nu)
cold = Conrath(ma, np.array([10]), np.array([0.5]))
print(cold.profile)
#print(ma.model_altitudes)

zb = np.array([10, 21, 0])
zt = np.array([30, 51, 109])
u = Uniform(ma, zb, zt)
print(u.profile)

l = Layers(ma, np.array([1, -8]))
print(l.profile)

p = ProfileHolder()
p.add_profile(cold.profile)
p.add_profile(u.profile)
p.stack_profiles()
print(p.profile.shape)
p.add_profile(l.profile)
p.stack_profiles()
print(p.profile.shape)


#print(cold.profile)
#print(u.profile.shape)
#print(l.profile.shape)
#print(p.profile[:, 0])
