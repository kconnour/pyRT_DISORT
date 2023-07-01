import numpy as np

from pyrt.grid import regrid


class TestRegrid:
    def test_function_works_for_2d_array_with_smaller_gridding(self):
        array = np.ones((50, 40))
        reff_grid = np.ones((50,))
        wav_grid = np.ones((40,))
        reff = np.ones((20,))
        wav = np.ones((10,))
        expected_shape = (20, 10)

        grid = regrid(array, reff_grid, wav_grid, reff, wav)

        assert grid.shape == expected_shape

    def test_function_works_for_2d_array_with_larger_gridding(self):
        array = np.ones((50, 40))
        reff_grid = np.ones((50,))
        wav_grid = np.ones((40,))
        reff = np.ones((200,))
        wav = np.ones((100,))
        expected_shape = (200, 100)

        grid = regrid(array, reff_grid, wav_grid, reff, wav)

        assert grid.shape == expected_shape

    def test_function_works_for_3d_array(self):
        array = np.ones((256, 50, 40))
        reff_grid = np.ones((50,))
        wav_grid = np.ones((40,))
        reff = np.ones((20,))
        wav = np.ones((10,))
        expected_shape = (256, 20, 10)

        grid = regrid(array, reff_grid, wav_grid, reff, wav)

        assert grid.shape == expected_shape
