import numpy as np
import scipy.interpolate.interpolate as interpolate

# Make the super class
class Map():
    def __init__(self, map_file, latitude, longitude):
        self.map_file = map_file
        self.latitude = latitude
        self.longitude = longitude

    def read_map(self):
        return np.load(self.map_file, allow_pickle=True)

# This is its own class for now because we may extend this to include hyperspectral albedo
class Albedo(Map):
    def __init__(self, map_file, latitude, longitude):
        super().__init__(map_file, latitude, longitude)

    def interpolate_albedo(self):
        map_array = self.read_map()
        latitudes = np.linspace(-90, 90, num=180, endpoint=True)
        longitudes = np.linspace(0, 360, num=360, endpoint=True)
        interp = interpolate.RectBivariateSpline(latitudes, longitudes, map_array)
        return interp(self.latitude, self.longitude)[0]


class Altitude(Map):
    def __init__(self, latitude, longitude, map_file):
        super().__init__(latitude, longitude, map_file)

    def interpolate_altitude(self):
        map_array = self.read_map()
        latitudes = np.linspace(-90, 90, num=180, endpoint=True)
        longitudes = np.linspace(0, 360, num=360, endpoint=True)
        interp = interpolate.RectBivariateSpline(latitudes, longitudes, map_array)
        return interp(self.latitude, self.longitude)[0]


''' #Examples:
albmap = '/Users/kyco2464/pyRT_DISORT/aux/albedo_map.npy'
altmap = '/Users/kyco2464/pyRT_DISORT/aux/altitude_map.npy'

altitude = Altitude(altmap, 0, 0)
print(altitude.interpolate_altitude())

albedo = Albedo(albmap, 0, 0)
print(albedo.interpolate_albedo())'''
