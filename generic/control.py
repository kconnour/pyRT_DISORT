# 3rd-party imports
import numpy as np


class Control:
    def __init__(self, user_optical_depths=False, user_angles=True, only_fluxes=True, accuracy=0,
                 print_variables=np.array([True, False, False, False, True]), header='', do_pseudo_sphere=False,
                 planetary_radius=6371, delta_m_plus=False):
        self.user_optical_depths = user_optical_depths
        self.user_angles = user_angles
        self.only_fluxes = only_fluxes
        self.accuracy = accuracy
        self.print_variables = print_variables
        if not header:
            self.header = 'Atmospheric pre-processing done in pyRT_DISORT'
        else:
            self.header = header
        self.do_pseudo_sphere = do_pseudo_sphere
        self.radius = planetary_radius
        self.delta_m_plus = delta_m_plus
