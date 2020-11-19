# Built-in imports
import os
from unittest import TestCase
import unittest

# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.preprocessing.model import AerosolProperties, Aerosol
from pyRT_DISORT.data.get_data import get_data_path
from pyRT_DISORT.preprocessing.utilities.external_files import ExternalFile


'''class AerosolPropertiesTest(TestCase):
    def setUp(self):
        self.dust_file = ExternalFile(os.path.join(get_data_path(), 'planets/mars/aux/dust_properties.fits'))
        self.properties ='''


class AerosolTest(TestCase):
    def setUp(self):
        self.dustfile = ExternalFile(os.path.join(get_data_path(), 'planets/mars/aux/dust.npy'))
        self.wavs = np.array([1, 9.3])
        self.wave_ref = 9.3
        self.aerosol = Aerosol(self.dustfile.array, self.wavs, self.wave_ref)

    def test1DInput(self):
        input = np.zeros(10)
        try:
            Aerosol(input, self.wavs, self.wave_ref)
        except AssertionError:
            return True

    def testNDInput(self):
        rng = np.random.default_rng()
        random_int = rng.integers(3, high=10, size=1)[0]




    def testWavs(self):
        return np.array_equal(self.aerosol.wavelengths, self.wavs)

    def testWaveRef(self):
        return self.aerosol.reference_wavelength == self.wave_ref

    def testWavInput(self):
        Aerosol(self.dustfile.array, 10, self.wave_ref)


if __name__ == '__main__':
    dustfile = ExternalFile(os.path.join(get_data_path(), 'planets/mars/aux/dust.npy'))
    wavs = np.array([1, 9.3])
    wref = 9.3
    aero = Aerosol(dustfile.array, wavs, wref)


    #unittest.main()
