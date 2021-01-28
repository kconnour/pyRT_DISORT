# 3rd-party imports
import numpy as np


class BoundaryConditions:
    """ A BoundaryConditions object holds the quantities DISORT needs for boundary conditions."""
    def __init__(self, ibcnd=0, beam_flux=np.pi, fisot=0, lambertian_bottom_boundary=True, albedo=0,
                 plank=False, bottom_temperature=0, top_temperature=0, top_emissivity=1):
        """
        Parameters
        ----------
        ibcnd: ??
            No idea what this is but it's 0
        beam_flux: float
            Not really sure why this would be anything but pi
        fisot: ??
            No idea what this is but it's 0
        lambertian_bottom_boundary: bool
            Denote if the bottom boundary should be a Lambertian boundary. Default is True
        albedo: float
            Set the bottom boundary albedo. Default is 0
        plank: bool
            Denote if the model should consider thermal emission. Default is False
        bottom_temperature: float
            Denote the temperature of the bottom boundary. Only used if plank=False. Default is 0
        top_temperature: float
            Denote the temperature of the top boundary. Only used if plank=False. Default is 0
        top_emissivity: float
            Denote the emissivity at TOA. Only used if plank=False. Default is 1
        """

        #TODO: Once I figure out what these variables are useful for, I need to write checks that the input is good
        #TODO: and add attributes to the docstring

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
