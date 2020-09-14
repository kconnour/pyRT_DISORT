# 3rd-party imports
import numpy as np


class BoundaryConditions:
    def __init__(self, ibcnd=0, beam_flux=np.pi, fisot=0, lambertian_bottom_boundary=True, albedo=0,
                 plank=True, bottom_temperature=0, top_temperature=0, top_emissivity=1):

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
