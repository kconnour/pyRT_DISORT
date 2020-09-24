# 3rd-party imports
import numpy as np


class ModelAtmosphere:
    def __init__(self):
        self.ods = []
        self.ssas = []
        self.pmoms = []
        self.total_optical_depths = np.nan
        self.total_single_scattering_albedos = np.nan
        self.total_polynomial_moments = np.nan

    def add_constituent(self, properties):
        assert len(properties) == 3
        self.ods.append(properties[0])
        self.ssas.append(properties[1])
        self.pmoms.append(properties[2])

    def compute_model(self):
        self.calculate_optical_depths()
        self.calculate_single_scattering_albedos()
        self.calculate_pmom()

    def calculate_optical_depths(self):
        self.total_optical_depths = sum(self.ods)

    def calculate_single_scattering_albedos(self):
        self.total_single_scattering_albedos = sum(self.ssas) / self.total_optical_depths

    def calculate_pmom(self):
        self.total_polynomial_moments = sum(self.pmoms) / (self.total_optical_depths * self.total_single_scattering_albedos)
