from astropy.io import fits
import numpy as np


class Fits:
    """Create a .fits file from numpy arrays"""
    def __init__(self, primary_hdu):
        """

        Parameters
        ----------
        primary_hdu: np.ndarray
            The data to go into the primary structure
        """
        self.primary_hdu = primary_hdu
        self.columns = []
        assert isinstance(primary_hdu, np.ndarray), 'primary_hdu must be a numpy array'
        self.__add_primary_hdu()

    def __add_primary_hdu(self):
        hdu = fits.PrimaryHDU()
        hdu.data = self.primary_hdu
        self.columns.append(hdu)

    def add_image_hdu(self, data, name):
        """ Add an ImageHDU object

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
        assert isinstance(data, np.ndarray), 'data must be a numpy array'
        assert isinstance(name, str), 'name must be a string'
        image = fits.ImageHDU(name=name)
        image.data = data
        self.columns.append(image)

    def save_fits(self, save_location, overwrite=True):
        """ Save this object as a .fits file

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
        assert isinstance(save_location, str), 'save location must be a string'
        assert isinstance(overwrite, bool), 'overwrite must be a boolean'

        combined_fits = fits.HDUList(self.columns)
        combined_fits.writeto(save_location, overwrite=overwrite)
