import os
from pyRT_DISORT.data.get_data import get_data_path
from pyRT_DISORT.preprocessing.utilities.external_files import ExternalFile
from pyRT_DISORT.preprocessing.utilities.fit_phase_function import PhaseFunction

ice_phase_function = ExternalFile(os.path.join(get_data_path(), 'planets/mars/aux/ice_shape001_r030_00321.dat'),
                                  header_lines=3, text1d=False)
pf = PhaseFunction(ice_phase_function.array)
created_coefficients = pf.create_legendre_coefficients(n_moments=128, n_samples=361)
print(created_coefficients)
