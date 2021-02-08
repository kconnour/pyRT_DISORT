import numpy as np
from pyRT_DISORT.controller import ComputationalParameters, ModelBehavior
from pyRT_DISORT.surface import Surface
from pyRT_DISORT.flux import IncidentFlux
from pyRT_DISORT.observation import Angles
from disort import disobrdf


# TODO: Become sure of what this does and remove it
class Unsure:
    """This class makes the variable "h_lyr". I don't really know what this
    variable does since it can be all 0s.

    """
    def __init__(self, cp: ComputationalParameters):
        self.cp = cp
        self.__h_lyr = np.zeros(self.cp.n_layers+1)

    @property
    def h_lyr(self):
        return self.__h_lyr


class Hapke(Surface):
    """A Hapke object makes a Hapke surface phase function as defined in default DISORT"""
    def __init__(self, albedo, cp, mb, flux, ang, b0=1, hh=0.06, w=0.6, n_mug=200, debug=False):
        """
        Parameters
        ----------
        albedo: float
        cp: ComputationalParameters
        mb: ModelBehavior
        flux: IncidentFlux
        ang: Angles
        b0: int or float
        hh: float
        w: float
        n_mug: int
        """
        super().__init__(albedo, cp)

        self.brdf_type = 1    # The "Hapke" integer used in DISORT
        self.cp = cp
        self.mb = mb
        self.flux = flux
        self.ang = ang
        self.b0 = b0          # Hapke's B0 factor for finite particle size
        self.hh = hh          # Angular width parameter of oppposition effect
        self.w = w            # Single scattering albedo in Hapke model
        self.n_mug = n_mug    # No idea...
        self.debug = debug
        self.brdf_argument = np.array([self.b0, self.hh, self.w, 0, 0, 0])
        self.__call_disobrdf()

    def __call_disobrdf(self):
        rhoq, rhou, emust, bemst, rho_accurate = \
            disobrdf(self.mb.user_angles, self.ang.mu,self.flux.beam_flux,
                 self.ang.mu0, False, self.albedo, self.mb.only_fluxes,
                 self._rhoq, self._rhou, self._emust, self._bemst,
                 self.debug, self.ang.phi, self.ang.phi0, self._rho_accurate,
                 self.brdf_type, self.brdf_argument, self.n_mug, nstr=self.cp.n_streams, numu=self.cp.n_polar,
                 nphi=self.cp.n_azimuth)
        self.rhoq = rhoq
        self.rhou = rhou
        self.emust = emust
        self.bemst = bemst
        self.rho_accurate = rho_accurate


class HapkeHG2(Surface):
    """ Construct the arrays to make a Hapke surface with 2-lobed Henyey-Greenstein phase function"""
    def __init__(self, albedo, cp, mb, flux, ang, b0=1, hh=0.06, w=0.6, asym=0.5,
                 frac=0.5, n_mug=200, debug=False):
        super().__init__(albedo, cp)
        self.brdf_type = 5    # The "Hapke" integer used in DISORT
        self.cp = cp
        self.mb = mb
        self.flux = flux
        self.ang = ang
        self.b0 = b0          # Hapke's B0 factor for finite particle size
        self.hh = hh          # Angular width parameter of oppposition effect
        self.w = w            # Single scattering albedo in Hapke model
        self.asym = asym
        self.frac = frac
        self.n_mug = n_mug    # My guess: the number of expansion coefficients
        self.brdf_argument = np.array([self.b0, self.hh, self.w, self.asym, self.frac, 0])
        self.debug = debug
        self.__call_disobrdf()

    def __call_disobrdf(self):
        rhoq, rhou, emust, bemst, rho_accurate = \
            disobrdf(self.mb.user_angles, self.ang.mu,self.flux.beam_flux,
                 self.ang.mu0, False, self.albedo, self.mb.only_fluxes,
                 self._rhoq, self._rhou, self._emust, self._bemst,
                 self.debug, self.ang.phi, self.ang.phi0, self._rho_accurate,
                 self.brdf_type, self.brdf_argument, self.n_mug, nstr=self.cp.n_streams, numu=self.cp.n_polar,
                 nphi=self.cp.n_azimuth)
        self.rhoq = rhoq
        self.rhou = rhou
        self.emust = emust
        self.bemst = bemst
        self.rho_accurate = rho_accurate


class HapkeHG2Roughness(Surface):
    """ Construct the arrays to make a Hapke surface with 2-lobed Henyey-Greenstein phase function and roughness"""
    def __init__(self, albedo, cp, mb, flux, ang, b0=1, hh=0.06, w=0.6, asym=0.5,
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
        super().__init__(albedo, cp)
        self.brdf_type = 6    # The "Hapke" integer used in DISORT
        self.cp = cp
        self.mb = mb
        self.flux = flux
        self.ang = ang
        self.b0 = b0          # Hapke's B0 factor for finite particle size
        self.hh = hh          # Angular width parameter of oppposition effect
        self.w = w            # Single scattering albedo in Hapke model
        self.asym = asym
        self.frac = frac
        self.roughness = roughness
        self.n_mug = n_mug    # My guess: the number of expansion coefficients
        self.brdf_argument = np.array([self.b0, self.hh, self.w, self.asym, self.frac, self.roughness])
        self.debug = debug
        self.__call_disobrdf()

    def __call_disobrdf(self):
        rhoq, rhou, emust, bemst, rho_accurate = \
            disobrdf(self.mb.user_angles, self.ang.mu,self.flux.beam_flux,
                 self.ang.mu0, False, self.albedo, self.mb.only_fluxes,
                 self._rhoq, self._rhou, self._emust, self._bemst,
                 self.debug, self.ang.phi, self.ang.phi0, self._rho_accurate,
                 self.brdf_type, self.brdf_argument, self.n_mug, nstr=self.cp.n_streams, numu=self.cp.n_polar,
                 nphi=self.cp.n_azimuth)
        self.rhoq = rhoq
        self.rhou = rhou
        self.emust = emust
        self.bemst = bemst
        self.rho_accurate = rho_accurate
