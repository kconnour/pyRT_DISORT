import numpy as np
import scipy.interpolate.interpolate as interpolate


class Observation:
    def __init__(self, short_wavelength, long_wavelength, solar_zenith_angle, emission_angle, phase_angle, latitude,
                 longitude, altitude_map_path, solar_flux_file_path):
        """ Initialize the class

        Parameters
        ----------
        short_wavelength: float
            The shorter wavelength for this pixel (in nm)
        long_wavelength: float
            The longer wavelength for this pixel (in nm)
        solar_zenith_angle: float
            The pixel solar zenith angle (in degrees)
        emission_angle: float
            The pixel emission angle (in degrees)
        phase_angle: float
            The pixel phase angle (in degrees)
        latitude: float
            The pixel latitude (in degrees)
        longitude: float
            The pixel longitude (in degrees east). Note The convention is 0--360, not -180 -- +180
        altitude_map_path: str
            The Unix-like path to the .npy file containing the MOLA map of altitudes
        solar_flux_file_path: str
            The Unix-like path the the .npy file containing the solar flux
        """
        self.short_wavelength = short_wavelength
        self.long_wavelength = long_wavelength
        self.sza = solar_zenith_angle
        self.emission = emission_angle
        self.phase = phase_angle
        self.latitude = latitude
        self.longitude = longitude
        self.phi0 = 0
        self.map_path = altitude_map_path
        self.solar_flux_file = solar_flux_file_path

        # Ensure the object knows everything it ought to know
        self.low_wavenumber = self.wavelength_to_wavenumber(self.long_wavelength)
        self.high_wavenumber = self.wavelength_to_wavenumber(self.short_wavelength)
        self.mu = self.calculate_mu()
        self.mu0 = self.calculate_mu0()
        self.phi = self.calculate_phi()
        self.altitude = self.get_altitude()
        self.solar_flux = self.calculate_solar_flux()

    def get_altitude(self):
        map_array = np.load(self.map_path)
        latitudes = np.linspace(-90, 90, num=180, endpoint=True)
        longitudes = np.linspace(0, 360, num=360, endpoint=True)
        interp = interpolate.RectBivariateSpline(latitudes, longitudes, map_array)
        return interp(self.latitude, self.longitude)[0]

    def calculate_mu0(self):
        """ Calculate the cosine of the solar zenith angle

        Returns
        -------
        mu0 = cos(theta_0)
        """
        return np.cos(np.deg2rad(self.sza))

    def calculate_mu(self):
        """ Calculate the cosine of the emission angle

        Returns
        -------
        mu = cos(theta)
        """
        return np.cos(np.deg2rad(self.emission))

    def calculate_phi(self):
        """ Calculate the azimuthal angle for the given geometry.

        Returns
        -------
        azimuthal_angle: float
            The angle in degrees
        """

        # I used a different calculation than Mike's.. but they seem to give the same answer and this is faster
        sin_emission_angle = np.cos(np.deg2rad(90-self.emission))
        sin_solar_zenith_angle = np.cos(np.deg2rad(90-self.sza))

        # Trap the case of the emission angle or solar zenith angle = 0
        if sin_emission_angle == 0 or sin_solar_zenith_angle == 0:
            d_phi = np.pi

        else:
            temp_var = (np.cos(np.deg2rad(self.phase)) - self.calculate_mu0()*self.calculate_mu0()) / \
                       (sin_emission_angle * sin_solar_zenith_angle)
            # Trap the round-off case for arc cosine
            if np.abs(temp_var) > 1:
                temp_var = np.sign(temp_var)
            d_phi = np.arccos(temp_var)

        d_phi = np.rad2deg(d_phi)
        azimuthal_angle = self.phi0 + 180 - d_phi
        return azimuthal_angle

    @staticmethod
    def wavelength_to_wavenumber(wavelength):
        """ Convert wavelength (in nm) to wavenumber (in inverse cm)

        Returns
        -------
        wavenumber: float
            The wavenumber for this observation
        """
        cm_wavelength = wavelength * 10**-7
        return 1 / cm_wavelength

    def calculate_solar_flux(self):
        """Calculate the incident solar flux between the input wavelengths. Note this is NOT corrected
        for solar zenith angle; in other words, flux = calculate_solar_flux * cos(SZA)

        Returns
        -------
        integrated_flux: float
            The integrated flux
        """
        solar_spec = np.load(self.solar_flux_file, allow_pickle=True)
        wavelengths = solar_spec[:, 0]
        fluxes = solar_spec[:, 1]
        interp_fluxes = np.interp(np.array([self.short_wavelength, self.long_wavelength]), wavelengths, fluxes)
        integrated_flux = np.mean(interp_fluxes) * (self.long_wavelength - self.short_wavelength)
        return integrated_flux
