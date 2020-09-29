from disort import disort
from disort import disobrdf

print(disort.__doc__)
print(disobrdf.__doc__)

#phsfn = np.load('/home/kyle/repos/pyRT_DISORT/preprocessing/planets/mars/aux/phase_functions.npy')
#p = np.moveaxis(phsfn, -1, 0)
#print(p.shape)
#np.save('/home/kyle/repos/pyRT_DISORT/preprocessing/planets/mars/aux/phase_functions.npy', p)

'''
# This allows multiplication of oddly sized arrays
a = np.zeros((14, 5, 2))
b = np.ones((128, 5, 2))

c = np.broadcast_to(a, (128, 14, 5, 2))
d = np.broadcast_to(b[:, None, :, :], (128, 14, 5, 2))

e = c*d'''