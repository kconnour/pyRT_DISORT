# Built-in imports
import os
import unittest

# 3rd-party imports
import numpy as np

# Local imports
from pyRT_DISORT.preprocessing.model import AerosolProperties, Aerosol
from pyRT_DISORT.data.get_data import get_data_path
from pyRT_DISORT.preprocessing.utilities.external_files import ExternalFile


class Test2DMarsDust(unittest.TestCase):
    def setUp(self):
        self.dust_file = ExternalFile(os.path.join(get_data_path(), 'planets/mars/aux/dust_properties.fits'))
        self.wavelength_grid = self.dust_file.array['wavelengths'].data
        self.primary = self.dust_file.array['primary'].data
        self.properties = AerosolProperties(self.primary, wavelength_grid=self.wavelength_grid)


class Test2DSpectralOperations(Test2DMarsDust):
    # All tests must start with 'test'
    def test_types(self):
        assert (self.properties.particle_size_grid is None) and (type(self.properties.wavelength_grid) == np.ndarray)

    def test_numeric(self):
        assert np.issubdtype(self.properties.wavelength_grid.dtype, np.number)

    def test_1d(self):
        assert np.ndim(self.properties.wavelength_grid) == 1

    def test_positive(self):
        assert np.all(self.properties.wavelength_grid > 0)

    def test_finite(self):
        assert np.all(np.isfinite(self.properties.wavelength_grid))


class Test2DBadInput(Test2DMarsDust):
    @unittest.expectedFailure
    def test_no_input(self):
        AerosolProperties(self.primary)

    @unittest.expectedFailure
    def test_too_much_input(self):
        AerosolProperties(self.primary, wavelength_grid=self.wavelength_grid, particle_size_grid=self.wavelength_grid)

    @unittest.expectedFailure
    def test_grid_cannot_be_none(self):
        AerosolProperties(self.primary, wavelength_grid=None)

    @unittest.expectedFailure
    def test_grid_cannot_be_str(self):
        AerosolProperties(self.primary, wavelength_grid='myGrid')

    @unittest.expectedFailure
    def test_grid_cannot_be_nd(self):
        a = np.zeros(4, 5)
        AerosolProperties(self.primary, wavelength_grid=a)

    @unittest.expectedFailure
    def test_grid_cannot_contain_infinity(self):
        a = np.zeros(10)
        a[0] = np.inf
        AerosolProperties(self.primary, wavelength_grid=a)

    def test_unrelaistic_grids_can_work(self):
        a = np.linspace(10000, 100000, num=len(self.wavelength_grid))
        AerosolProperties(self.primary, wavelength_grid=a)


if __name__ == '__main__':
    unittest.main()
