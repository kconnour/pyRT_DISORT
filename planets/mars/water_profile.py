# 3rd-party imports
from scipy.constants import Boltzmann
import numpy as np

# Local imports
from layer import Atmosphere, Layers


class WaterVaporProfile(object):
    def __init__(self, altitudes, temperatures, pressures, saturation_scale, h2o_column, convergence=10**(-3)):
        self.altitudes = altitudes
        self.temperatures = temperatures
        self.pressures = pressures
        self.sat_scale = saturation_scale
        self.h2o_column = h2o_column
        self.convergence = convergence

    def calculate_saturation_mixing_ratio(self):
        """ Calculate the saturation mixing ratio by some crazy function

        Returns
        -------
        f: np.ndarray
            The saturation mixing ratio at the input temperatures and pressures
        """
        f = 10 ** (-6.757169 - 2445.5624 / self.temperatures + 8.2312 * np.log10(
            self.temperatures) - 0.01677006 * self.temperatures + 1.20514 * 10 ** (-5) * self.temperatures ** 2) / (
                        self.pressures * 760 / 1013)
        return f

    def calculate_pressure_midpoints(self):
        """Calculate the pressure midpoints using Mike's method

        Returns
        -------

        """
        midpoints = np.sqrt(self.pressures[:-1] * self.pressures[1:])
        return midpoints

    def calculate_temperature_midpoints(self):
        """Calculate the temperature midpoints by taking their average

        Returns
        -------

        """
        midpoints = (self.temperatures[:-1] + self.temperatures[1:]) / 2
        return midpoints

    def calculate_saturation_midpoints(self):
        saturation_mixing_ratios = self.calculate_saturation_mixing_ratio()
        midpoints = np.sqrt(saturation_mixing_ratios[:-1] * saturation_mixing_ratios[1:]) * self.sat_scale
        return midpoints

    def make_water_profile(self):
        # Make arrays
        temp_midpoints = self.calculate_temperature_midpoints()
        pressure_midpoints = self.calculate_pressure_midpoints()
        sat_midpoints = self.calculate_saturation_midpoints()
        delta_mol_h2o = np.zeros(len(self.temperatures))

        # Make looping varaibles
        h2o_total = self.h2o_column * 3.34 * 10 ** 16
        initial_sat_mixing_ratio = 10**(-6)
        sat_mixing_ratio_guess = 10**(-6)
        relative_error = 10
        iteration = 0
        sat_scale_increase = 0
        sat_scale = self.sat_scale

        while np.abs(relative_error) > self.convergence:
            for i in range(len(self.temperatures) - 1):
                f_mix = sat_mixing_ratio_guess
                if f_mix > sat_midpoints[i]:
                    f_mix = sat_midpoints[i]
                f_water = f_mix * pressure_midpoints[i] / Boltzmann / temp_midpoints[i]
                delta_mol_h2o[i] = f_water * (self.altitudes[i+1] - self.altitudes[i])

            total_mols_water = np.sum(delta_mol_h2o)

            # calculate relative error and adjust the initial guess
            print(iteration, sat_scale_increase)
            relative_error = (total_mols_water - h2o_total) / h2o_total
            sat_mixing_ratio_guess *= (1 - relative_error)
            iteration += 1

            # If the water column isn't converging, change the saturation scaling
            if iteration == 100:
                sat_scale *= 2
                sat_midpoints = self.calculate_saturation_midpoints()

                # reset looping values
                sat_mixing_ratio_guess = initial_sat_mixing_ratio
                relative_error = 10
                iteration = 0

                # remember that this iteration didn't really work
                sat_scale_increase += 1

            if sat_scale_increase == 5:
                print('You want too much water. Seek help...')
                return

        return f_water


marsatm = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/mars_atm.npy'
lay = Layers(15, 10, 10, 10, marsatm)
z, P, T = lay.read_atmosphere()
prof = WaterVaporProfile(z, T, P, 1, 10**6)
print(prof.make_water_profile())
