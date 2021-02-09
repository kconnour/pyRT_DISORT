# 3rd-party imports
import numpy as np

# Local imports
#from pyRT_DISORT.preprocessing.model.atmosphere import Layers
from pyRT_DISORT.preprocessing.model.atmosphere import ModelAtmosphere
from pyRT_DISORT.preprocessing.utilities.external_files import ExternalFile
from pyRT_DISORT.preprocessing.model.aerosol_column import Column as NewColumn
from pyRT_DISORT.preprocessing.model.aerosol import Aerosol
from pyRT_DISORT.preprocessing.model.vertical_profiles import Conrath, Uniform, Layers, ProfileHolder


atmFile = ExternalFile('/home/kyle/repos/pyRT_DISORT/pyRT_DISORT/data/planets/mars/aux/mars_atm.npy')
ma = ModelAtmosphere(atmFile.array, atmFile.array[:, 0])

dustFile = ExternalFile('/home/kyle/repos/pyRT_DISORT/pyRT_DISORT/data/planets/mars/aux/dust_properties.fits')
radii = np.array([1, 1.2, 1.4])
wavs = np.array([1, 9.3])
wavref = np.array([12.1, 12.1, 12.1])
aero = Aerosol(dustFile.array['primary'].data, radii, wavs, wavelength_grid=dustFile.array['wavelengths'].data, reference_wavelengths=wavref)

sh = np.array([10, 10, 10])
nu = np.array([0.5, 0.5, 0.5])
c = Conrath(ma, sh, nu)
p = ProfileHolder()
p.add_profile(c.profile)
p.stack_profiles()


#raise SystemExit(9)

ods = np.array([0.1, 0.8, 0.2])
newCol = NewColumn(aero, ma, p.profile, ods)
print(newCol.total_optical_depth)
