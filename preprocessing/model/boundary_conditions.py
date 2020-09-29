# 3rd-party imports
import numpy as np
from disort import disobrdf

# Local imports
from preprocessing.controller.size import Size
from preprocessing.observation import Observation
from preprocessing.controller.unsure import Unsure
from preprocessing.controller.control import Control


class BoundaryConditions:
    def __init__(self, ibcnd=0, beam_flux=np.pi, fisot=0, lambertian_bottom_boundary=True, albedo=0,
                 plank=False, bottom_temperature=0, top_temperature=0, top_emissivity=1):

        # Check for errors first
        assert isinstance(lambertian_bottom_boundary, bool), 'lambertian_bottom_boundary must be a boolean'
        assert isinstance(plank, bool), 'plank must be a boolean'

        self.ibcnd = ibcnd
        self.beam_flux = beam_flux
        self.fisot = fisot
        self.lambertian = lambertian_bottom_boundary

        if self.lambertian:
            self.albedo = albedo
        else:
            print('Using a bottom boundary described in BDREF.')

        self.plank = plank
        if self.plank:
            self.bottom_temperature = bottom_temperature
            self.top_temperature = top_temperature
            self.top_emissivity = top_emissivity
        else:
            # If plank == False, these values are not used
            self.bottom_temperature = 0
            self.top_temperature = 0
            self.top_emissivity = 0


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
