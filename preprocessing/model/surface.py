import numpy as np
'''from disort import disobrdf

# Local imports
from preprocessing.controller.size import Size
from preprocessing.observation import Observation
from preprocessing.controller.unsure import Unsure
from preprocessing.controller.control import Control
from preprocessing.model.boundary_conditions import BoundaryConditions'''


# This calls disobrdf... which seems broken
class Hapke:
    def __init__(self, size, observation, unsure, control, boundary_conditions, albedo, b0=1, hh=0.06, w=0.6,
                 n_mug=200, debug=False):
        assert isinstance(size, Size), 'size needs to be an instance of Size.'
        assert isinstance(observation, Observation)
        assert isinstance(unsure, Unsure)
        assert isinstance(control, Control)
        assert isinstance(boundary_conditions, BoundaryConditions)

        self.brdf_type = 1    # The "Hapke" integer used in DISORT
        self.size = size
        self.observation = observation
        self.unsure = unsure
        self.control = control
        self.boundary_conditions = boundary_conditions
        self.albedo = albedo
        self.b0 = b0          # Hapke's B0 factor for finite particle size
        self.hh = hh          # Angular width parameter of oppposition effect
        self.w = w            # Single scattering albedo in Hapke model
        self.n_mug = n_mug    # No idea...
        self.brdf_argument = np.array([self.b0, self.hh, self.w, 0])
        self.debug = debug

    def call_disobrdf(self):
        rhou = np.zeros((16, 9, 16))
        disobrdf(self.control.user_angles, self.observation.mu, self.boundary_conditions.beam_flux,
                 self.observation.mu0, self.boundary_conditions.lambertian, self.albedo, self.control.only_fluxes,
                 self.unsure.make_rhoq(), rhou, self.unsure.make_emust(), self.unsure.make_bemst(),
                 self.debug, self.observation.phi, self.observation.phi0, self.unsure.make_rho_accurate(),
                 self.brdf_type, self.brdf_argument, self.n_mug, nstr=self.size.n_streams, numu=self.size.n_umu,
                 nphi=self.size.n_phi)


class MyHapke:
    def __init__(self, b0, w, h):
        self.b0 = b0
        self.w = w
        self.h = h
        self.gamma = np.sqrt(1 - w)
        self.r0 = (1 - self.gamma) / (1 + self.gamma)

    def B_function(self, g):
        # This is equation 8.90 in Theory of Reflectance and Emittance Spectroscopy
        return (self.b0 * self.h) / (self.h + np.tan(g/2))

    def H_function(self, x):
        # This is the second line of equation 8.57 in Theory of Reflectance and Emittance Spectroscopy
        par = 1 - 0.5*self.r0 - self.r0*x
        log = np.log((1+x)/x)
        return 1 / (1 - ((1 - self.gamma) * x * (self.r0 + par * log)))
