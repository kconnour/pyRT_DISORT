import numpy as np
from generic.aerosol import Aerosol


class Column:
    def __init__(self, aerosol, aerosol_scale_height, conrath_nu, column_optical_depth):
        """ Initialize the class

        Parameters
        ----------
        aerosol: Aerosol
            An aerosol object
        aerosol_scale_height: float
            The scale height of the aerosol
        conrath_nu: float
            The Conrath nu parameter
        column_optical_depth: float
            The column-integrated optical depth
        """
        self.aerosol = aerosol
        self.H = aerosol_scale_height
        self.nu = conrath_nu
        self.column_optical_depth = column_optical_depth
        self.check_conrath_parameters()

    def check_conrath_parameters(self):
        """ Check that the Conrath parameters don't suck

        Returns
        -------
        None
        """
        if np.isnan(self.H):
            print('You wily bastard, you input a nan as the Conrath dust scale height. Fix that.')
        elif np.isnan(self.nu):
            print('You wily bastard, you input a nan as the Conrath nu parameter. Fix that.')
        if self.H < 0:
            raise SystemExit('Bad Conrath aerosol scale height: it cannot be negative. Fix...')
        elif self.nu < 0:
            raise SystemExit('Bad Conrath nu parameter: it cannot be negative. Fix...')

    def make_conrath_profile(self, z_layer):
        """Calculate the vertical dust distribution assuming a Conrath profile, i.e.
        q(z) / q(0) = exp( nu * (1 - exp(z / H)))
        where q is the mass mixing ratio

        Parameters
        ----------
        z_layer: np.ndarray
            The altitudes of each layer's midpoint

        Returns
        -------
        fractional_mixing_ratio: np.ndarray
            The fraction of the mass mixing ratio at the midpoint altitudes
        """

        fractional_mixing_ratio = np.exp(self.nu * (1 - np.exp(z_layer / self.H)))
        return fractional_mixing_ratio

    def calculate_aerosol_optical_depths(self, z_layer, N):
        """ Make the optical depths within each layer

        Returns
        -------
        tau_aerosol: np.ndarray
            The optical depths in each layer of shape (n_layers, n_wavelengths)
        """
        vertical_mixing_ratio = self.make_conrath_profile(z_layer)
        dust_scaling = np.sum(N * vertical_mixing_ratio)
        tau_aerosol = np.outer(N * vertical_mixing_ratio, self.aerosol.extinction_ratio) * \
            self.column_optical_depth / dust_scaling
        return tau_aerosol


dust = Aerosol('/home/kyle/repos/pyRT_DISORT/planets/mars/aux/dust.npy', '', np.array([1, 2, 8, 9]), 9.3)
c = Column(dust, 10, 0.5, 1)
z = np.linspace(0, 100, num=14)
n = np.linspace(10**23, 10**26, num=14)

asdf = c.calculate_aerosol_optical_depths(z, n)
print(asdf.shape)
