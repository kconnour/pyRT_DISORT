# 3rd-party imports
import numpy as np
from disort import disobrdf

# Local imports
from pyRT_DISORT.preprocessing.controller.size import Size
from pyRT_DISORT.preprocessing.observation import Observation
from pyRT_DISORT.preprocessing.controller.control import Control
from pyRT_DISORT.preprocessing.model.boundary_conditions import BoundaryConditions


class Hapke:
    def __init__(self, size, observation, control, boundary_conditions, albedo, b0=1, hh=0.06, w=0.6,
                 n_mug=200, debug=False):
        assert isinstance(size, Size), 'size needs to be an instance of Size.'
        assert isinstance(observation, Observation)
        assert isinstance(control, Control)
        assert isinstance(boundary_conditions, BoundaryConditions)

        self.brdf_type = 1    # The "Hapke" integer used in DISORT
        self.size = size
        self.observation = observation
        self.control = control
        self.boundary_conditions = boundary_conditions
        self.albedo = albedo
        self.b0 = b0          # Hapke's B0 factor for finite particle size
        self.hh = hh          # Angular width parameter of oppposition effect
        self.w = w            # Single scattering albedo in Hapke model
        self.n_mug = n_mug    # No idea...
        self.brdf_argument = np.array([self.b0, self.hh, self.w, 0])
        self.debug = debug
        self.rhoq = self.__make_rhoq()
        self.rhou = self.__make_rhou()
        self.bemst = self.__make_bemst()
        self.emust = self.__make_emust()
        self.rho_accurate = self.__make_rho_accurate()
        self.__call_disobrdf()

    def __make_rhoq(self):
        # Make the variable "rhoq"
        return np.zeros((int(0.5*self.size.n_streams), int(0.5*self.size.n_streams+1), self.size.n_streams))

    def __make_rhou(self):
        # Make the variable "rhou"
        return np.zeros((self.size.n_streams, int(0.5*self.size.n_streams+1), self.size.n_streams))

    def __make_bemst(self):
        # Make the variable "bemst"
        # In DISOBRDF.f it's called the directional emissivity at quadrature angles
        return np.zeros(int(0.5*self.size.n_streams))

    def __make_emust(self):
        # Make the variable "emust"
        # In DISOBRDF.f it's called the directional emissivity at user angles
        return np.zeros(self.size.n_umu)

    def __make_rho_accurate(self):
        # Make the variable "rho_accurate"
        return np.zeros((self.size.n_umu, self.size.n_phi))

    def __call_disobrdf(self):
        rhoq, rhou, emust, bemst, rho_accurate = disobrdf(self.control.user_angles, self.observation.mu,
                                                          self.boundary_conditions.beam_flux,
                 self.observation.mu0, self.boundary_conditions.lambertian, self.albedo, self.control.only_fluxes,
                 self.rhoq, self.rhou, self.emust, self.bemst,
                 self.debug, self.observation.phi, self.observation.phi0, self.rho_accurate,
                 self.brdf_type, self.brdf_argument, self.n_mug, nstr=self.size.n_streams, numu=self.size.n_umu,
                 nphi=self.size.n_phi)
        self.rhoq = rhoq
        self.rhou = rhou
        self.emust = emust
        self.bemst = bemst
        self.rho_accurate = rho_accurate
