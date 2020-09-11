




'''    def add_rayleigh_co2_optical_depth(self, wavelength):
        """ Add the optical depth from Rayleigh scattering of CO2 to the total Rayleigh optical depth

        Parameters
        ----------
        wavelength:
            The wavelength of the observation

        Returns
        -------
        None
        """
        self.tau_rayleigh += self.calculate_rayleigh_co2_optical_depth(wavelength)

    def calculate_rayleigh_co2_optical_depth(self, wavelength):
        """ Calculate the Rayleigh CO2 optical depth at a given wavelength

        Parameters
        ----------
        wavelength:
            The wavelength of the observation

        Returns
        -------

        """
        tau_rayleigh_co2 = np.outer(self.N, rayleigh_co2(wavelength))
        return tau_rayleigh_co2

    def add_column(self, column):
        """ Add a column of an aerosol or gas to the atmosphere

        Parameters
        ----------
        column: Column
            A column instance of an aerosol or gas

        Returns
        -------
        None
        """
        self.columns.append(column)


def calculate_column_optical_depth(self, optical_depth_minimum=10**-7):
    """ Calculate the optical depth of each layer in a column

    Returns
    -------
    column_optical_depth: np.ndarray
        The optical depths in each layer
    """
    # Add in Rayleigh scattering
    column_optical_depth = np.copy(self.tau_rayleigh)

    # Add in the optical depths of each column
    for i in range(len(self.columns)):
        column_optical_depth += self.columns[i].calculate_aerosol_optical_depths(self.z_midpoints, self.N)

    # Make sure ODs cannot be 0 to avoid dividing by 0 later on
    column_optical_depth = np.where(column_optical_depth < optical_depth_minimum, optical_depth_minimum, column_optical_depth)
    return column_optical_depth

def calculate_single_scattering_albedo(self):
    """ Calculate the single scattering albedo of each layer in a column

    Returns
    -------
    single_scattering_albedo: np.ndarray
        The SSAs in each layer
    """

    # Add in Rayleigh scattering
    single_scattering_albedo = np.copy(self.tau_rayleigh)

    # Add in the single scattering albedo of each aerosol
    for i in range(len(self.columns)):
        scattering_ratio = self.columns[i].aerosol.scattering_coeff
        optical_depths = self.columns[i].calculate_aerosol_optical_depths(self.z_midpoints, self.N)
        single_scattering_albedo += scattering_ratio * optical_depths

    column_optical_depth = self.calculate_column_optical_depth()
    return single_scattering_albedo / column_optical_depth

def calculate_polynomial_moments(self):
    """ Calculate the polynomial moments for the atmosphere

    Returns
    -------
    polynomial_moments: np.ndarray
        An array of the polynomial moments
    """
    # Get info I'll need
    n_moments = self.columns[0].aerosol.n_moments
    n_layers = len(self.z_midpoints)
    n_wavelengths = len(self.columns[0].aerosol.wavelengths)
    rayleigh_moments = make_rayleigh_phase_function(n_moments, n_layers, n_wavelengths)
    tau_rayleigh = np.copy(self.tau_rayleigh)

    # Start by populating PMOM with Rayleigh scattering
    polynomial_moments = tau_rayleigh * rayleigh_moments

    # Add in the moments for each column
    for i in range(len(self.columns)):
        scattering = self.columns[i].aerosol.scattering_coeff
        column_optical_depths = self.columns[i].calculate_aerosol_optical_depths(self.z_midpoints, self.N)
        moments = self.columns[i].aerosol.phase_function
        moments_holder = np.zeros(polynomial_moments.shape)
        for j in range(len(self.z_midpoints)):
            moments_holder[:, j, :] = moments
        polynomial_moments += scattering * column_optical_depths * moments_holder

    column_optical_depth = self.calculate_column_optical_depth()
    single_scattering_albedo = self.calculate_single_scattering_albedo()
    scaling = column_optical_depth * single_scattering_albedo
    scaling = np.where(scaling == 0, 10**10, scaling)
    return polynomial_moments / scaling'''