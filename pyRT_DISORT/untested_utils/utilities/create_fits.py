# 3rd-party imports
from astropy.io import fits
import numpy as np

# Local imports
from pyRT_DISORT.untested_utils.utilities import ArrayChecker


class CreateFits:
    """A CreateFits object allows users to make .fits files"""
    def __init__(self, primary_hdu):
        """
        Parameters
        ----------
        primary_hdu: np.ndarray
            The data to go into the primary structure
        """
        self.primary_hdu = primary_hdu
        self.columns = []
        self.__add_primary_hdu()

    def add_image_hdu(self, data, name):
        """Add an ImageHDU to this object

        Parameters
        ----------
        data: np.ndarray
            The data to add to this structure
        name: str
            The name of this ImageHDU

        Returns
        -------
            None
        """
        self.__check_input_is_str(name, 'name')
        self.__check_addition_is_numpy_array(data, name)
        image = fits.ImageHDU(name=name)
        image.data = data
        self.columns.append(image)

    def save_fits(self, save_location, overwrite=True):
        """Save this object as a .fits file

        Parameters
        ----------
        save_location: str
            The location where to save this .fits file
        overwrite: bool
            Denote if this object should overwrite a file with the same name as save_location. Default is True

        Returns
        -------
            None
        """
        self.__check_input_is_str(save_location, 'save_location')
        combined_fits = fits.HDUList(self.columns)
        combined_fits.writeto(save_location, overwrite=overwrite)

    def __add_primary_hdu(self):
        self.__check_addition_is_numpy_array(self.primary_hdu, 'primary')
        hdu = fits.PrimaryHDU()
        hdu.data = self.primary_hdu
        self.columns.append(hdu)

    @staticmethod
    def __check_addition_is_numpy_array(array, name):
        hdu_checker = ArrayChecker(array, name)
        hdu_checker.check_object_is_array()

    @staticmethod
    def __check_input_is_str(test_name, input_name):
        if not isinstance(test_name, str):
            raise TypeError(f'{input_name} must be a string.')
