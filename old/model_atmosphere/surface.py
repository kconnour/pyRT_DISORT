# 3rd-party imports
import numpy as np
from disort import disobrdf

# Local imports
from old.model_controller import Size
from pyRT_DISORT.observation_old.observation import Observation
from old.model_controller import Control
from old.model_atmosphere.boundary_conditions import BoundaryConditions


class Hapke:
    """A Hapke object makes a Hapke surface phase function as defined in default DISORT"""
    def __init__(self, size, observation, control, boundary_conditions, albedo, b0=1, hh=0.06, w=0.6, n_mug=200):
        """
        Parameters
        ----------
        size: Size
        observation: Observation
        control: Control
        boundary_conditions: BoundaryConditions
        albedo: float
        b0: int or float
        hh: float
        w: float
        n_mug: int
        """
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
        self.brdf_argument = np.array([self.b0, self.hh, self.w, 0, 0, 0])
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


class HapkeHG2:
    """ Construct the arrays to make a Hapke surface with 2-lobed Henyey-Greenstein phase function"""
    def __init__(self, size, observation, control, boundary_conditions, albedo, b0=1, hh=0.06, w=0.6, asym=0.5,
                 frac=0.5, n_mug=200, debug=False):
        assert isinstance(size, Size), 'size needs to be an instance of Size.'
        assert isinstance(observation, Observation), 'observation_old needs to be an instance of Observation.'
        assert isinstance(control, Control), 'control needs to be an instance of Control.'
        assert isinstance(boundary_conditions, BoundaryConditions), \
            'boundary_conditions nees to be an instance of BoundaryConditions'

        assert 0 <= asym <= 1, 'The asymmetry parameter needs to be in [0, 1]'
        assert 0 <= frac <= 1, 'The forward scattering fraction needs to  be in [0, 1]'

        self.brdf_type = 5    # The "Hapke" integer used in DISORT
        self.size = size
        self.observation = observation
        self.control = control
        self.boundary_conditions = boundary_conditions
        self.albedo = albedo
        self.b0 = b0          # Hapke's B0 factor for finite particle size
        self.hh = hh          # Angular width parameter of oppposition effect
        self.w = w            # Single scattering albedo in Hapke model
        self.asym = asym
        self.frac = frac
        self.n_mug = n_mug    # My guess: the number of expansion coefficients
        self.brdf_argument = np.array([self.b0, self.hh, self.w, self.asym, self.frac, 0])
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


class HapkeHG2Roughness:
    """ Construct the arrays to make a Hapke surface with 2-lobed Henyey-Greenstein phase function and roughness"""
    def __init__(self, size, observation, control, boundary_conditions, albedo, b0=1, hh=0.06, w=0.6, asym=0.5,
                 frac=0.5, roughness=0.5, n_mug=200, debug=False):
        """

        Parameters
        ----------
        size
        observation
        control
        boundary_conditions
        albedo
        b0
        hh
        w
        asym
        frac
        roughness: float
            The roughness parameter in radians
        n_mug
        debug
        """
        assert isinstance(size, Size), 'size needs to be an instance of Size.'
        assert isinstance(observation, Observation), 'observation_old needs to be an instance of Observation.'
        assert isinstance(control, Control), 'control needs to be an instance of Control.'
        assert isinstance(boundary_conditions, BoundaryConditions), \
            'boundary_conditions nees to be an instance of BoundaryConditions'

        assert 0 <= asym <= 1, 'The asymmetry parameter needs to be in [0, 1]'
        assert 0 <= frac <= 1, 'The forward scattering fraction needs to  be in [0, 1]'

        self.brdf_type = 6    # The "Hapke" integer used in DISORT
        self.size = size
        self.observation = observation
        self.control = control
        self.boundary_conditions = boundary_conditions
        self.albedo = albedo
        self.b0 = b0          # Hapke's B0 factor for finite particle size
        self.hh = hh          # Angular width parameter of oppposition effect
        self.w = w            # Single scattering albedo in Hapke model
        self.asym = asym
        self.frac = frac
        self.roughness = roughness
        self.n_mug = n_mug    # My guess: the number of expansion coefficients
        self.brdf_argument = np.array([self.b0, self.hh, self.w, self.asym, self.frac, self.roughness])
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
