# 3rd-party imports
import numpy as np

# Local imports
from generic.aerosol import Aerosol


def test_asymmetry():
    dust_file = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/dust.npy'
    # Make wavelengths outside the range in the file
    unsupported_wavelengths = np.array([0.19, 4, 6, 250])
    unsupported_dust = Aerosol(dust_file, '', unsupported_wavelengths, 9.3)

    # Make wavelengths inside the range in the file
    supported_wavelengths = np.array([0.2, 4, 6, 50])
    supported_dust = Aerosol(dust_file, '', supported_wavelengths, 9.3)
    return np.all(supported_dust.calculate_asymmetry_parameter() == unsupported_dust.calculate_asymmetry_parameter())


print(test_asymmetry())
