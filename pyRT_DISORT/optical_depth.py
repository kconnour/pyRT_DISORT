import numpy as np
from pyRT_DISORT.forward_scattering import GriddedForwardScatteringProperties, ForwardScatteringProperties


class OpticalDepth:
    def __init__(self, forward_scattering: GriddedForwardScatteringProperties,
                 mixing_ratio_profile: np.ndarray,
                 column_density_layers: np.ndarray,
                 column_integrated_optical_depth: float) -> None:
        self.__gfsp = forward_scattering
        self.__OD = column_integrated_optical_depth
        self.__optical_depth = self.__calculate_total_optical_depth(mixing_ratio_profile, column_density_layers)

    def __calculate_total_optical_depth(self, mixing_ratio_profile: np.ndarray,
                                        column_density_layers: np.ndarray) -> np.ndarray:
        normalization = np.sum(mixing_ratio_profile * column_density_layers)
        profile = mixing_ratio_profile * column_density_layers * self.__OD / normalization
        return (profile * self.__gfsp.extinction.T).T

    @property
    def total(self) -> np.ndarray:
        return self.__optical_depth


if __name__ == '__main__':
    from astropy.io import fits
    f = '/Users/kyco2464/repos/pyRT_DISORT/tests/aux/dust_properties.fits'
    hdul = fits.open(f)
    cext = hdul['primary'].data[:, :, 0]
    csca = hdul['primary'].data[:, :, 1]
    wavs = hdul['wavelengths'].data
    psizes = hdul['particle_sizes'].data
    wavelengths = np.ones((10, 50)) * 9.3
    z = np.linspace(1, 1.5, num=20)
    fsp = ForwardScatteringProperties(csca, cext, psizes, wavs)
    g = GriddedForwardScatteringProperties(fsp, z, wavelengths, 1)
    q = np.ones(20)
    od = OpticalDepth(g, q, q, 2)
    print(np.sum(od.total[:, 0, 0]))
