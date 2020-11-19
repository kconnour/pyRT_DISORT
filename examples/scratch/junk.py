import numpy as np
from astropy.io import fits
from pyRT_DISORT.preprocessing.utilities.external_files import MultipleExternalFiles, ExternalFile
from pyRT_DISORT.preprocessing.utilities.create_fits import CreateFits
'''from pyRT_DISORT.preprocessing.utilities.fit_phase_function import PhaseFunction

# Read in the files
f = MultipleExternalFiles('ice*.phsfn', '/home/kyle/disort_multi/phsfn')   # Change this to work with your system
short_wavs = [0.2, 0.255, 0.3, 0.336, 0.4, 0.5, 0.588, 0.6, 0.631, 0.673, 0.7, 0.763, 0.8, 0.835, 0.9, 0.953]
wavs = np.linspace(1, 50, num=(50-1)*10+1)
wavs = np.concatenate((short_wavs, wavs))
radii = np.array([1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 80])
junk_file = ExternalFile(f.files[0], header_lines=7, text1d=False)     # False since it's 2D
ice_phsfns = np.zeros((junk_file.array.shape[0], junk_file.array.shape[1], len(radii), len(wavs)))  # shape: (181, 6, 11, 507)

radius_index = 0
for counter, file in enumerate(f.files):
    ext_file = ExternalFile(file, header_lines=7, text1d=False)
    ice_phsfns[:, :, radius_index, counter % len(wavs)] = ext_file.array
    if ((counter + 1) % len(wavs)) == 0:
        radius_index += 1

print(ice_phsfns.shape)
# Use the equation to make s12 into f11
degrees = ice_phsfns[:, 0, 0, 0]
f11 = ice_phsfns[:, 1, :, :]

a = np.zeros((128, 11, 507))
for i in range(11):
    print(i)
    for j in range(507):
        #print(j)
        ice_phase_function = np.column_stack((degrees, np.squeeze(f11[:, i, j])))
        pf = PhaseFunction(ice_phase_function)
        a[:, i, j] = pf.create_legendre_coefficients(n_moments=128, n_samples=361)

np.save('/home/kyle/ice_phase_function.npy', a)


#dustfile = '/home/kyle/repos/pyRT_DISORT/pyRT_DISORT/data/planets/mars/aux/dust.npy'
#a = np.load(dustfile)
#print(a[:, 0])'''

class OldAerosol:
    """ Create a class to hold all of the information about an aerosol"""

    def __init__(self, aerosol_file, wavelengths, reference_wavelength):
        """ Initialize the class to hold all the aerosol's properties
        Parameters
        ----------
        aerosol_file: np.ndarray
            An array containing the aerosol's properties
        wavelengths: np.ndarray
            The wavelengths at which this aerosol was observed
        reference_wavelength: float
            The wavelength at which to scale the wavelengths
        """

        self.aerosol_file = aerosol_file
        self.wavelengths = wavelengths
        self.reference_wavelength = reference_wavelength

        assert isinstance(self.aerosol_file, np.ndarray), 'aerosol_file must be a numpy array.'
        assert isinstance(self.wavelengths, np.ndarray), 'wavelengths needs to be a numpy array.'
        assert isinstance(self.reference_wavelength, float), 'reference_wavelength needs to be a float.'

        # Make sure the aerosol knows its properties
        self.wavelengths_quantities, self.c_extinction, self.c_scattering, self.kappa, self.g, \
            self.p_max, self.theta_max = self.__read_aerosol_file()
        self.__inform_if_outside_wavelength_range()
        self.extinction_ratios = self.__calculate_wavelength_extinction_ratios()
        self.hyperspectral_single_scattering_albedos = self.__calculate_hyperspectral_single_scattering_albedos()
        self.hyperspectral_asymmetry_parameters = self.__calculate_hyperspectral_asymmetry_parameters()

    def __read_aerosol_file(self):
        wavelengths = self.aerosol_file[:, 0]
        c_extinction = self.aerosol_file[:, 1]
        c_scattering = self.aerosol_file[:, 2]
        kappa = self.aerosol_file[:, 3]
        g = self.aerosol_file[:, 4]

        if self.aerosol_file.shape[1] == 5:
            p_max = np.array([])
            theta_max = np.array([])
        else:
            p_max = self.aerosol_file[:, 5]
            theta_max = self.aerosol_file[:, 6]

        return wavelengths, c_extinction, c_scattering, kappa, g, p_max, theta_max

    def __inform_if_outside_wavelength_range(self):
        if np.size((too_short := self.wavelengths[self.wavelengths < self.wavelengths_quantities[0]]) != 0):
            print('The following input wavelengths: {} microns are shorter than {:.3f} microns---the shortest '
                  'wavelength in the file. Using properties from that wavelength.'
                  .format(too_short, self.wavelengths_quantities[0]))
        if np.size((too_long := self.wavelengths[self.wavelengths > self.wavelengths_quantities[-1]]) != 0):
            print('The following input wavelengths: {} microns are longer than {:.1f} microns---the shortest '
                  'wavelength in the file. Using properties from that wavelength.'
                  .format(too_long, self.wavelengths_quantities[-1]))

    def __calculate_wavelength_extinction_ratios(self):
        reference_c_ext = np.interp(self.reference_wavelength, self.wavelengths_quantities, self.c_extinction)
        wavelengths_c_ext = np.interp(self.wavelengths, self.wavelengths_quantities, self.c_extinction)
        return wavelengths_c_ext / reference_c_ext

    def __calculate_hyperspectral_single_scattering_albedos(self):
        interpolated_extinction = np.interp(self.wavelengths, self.wavelengths_quantities, self.c_extinction)
        interpolated_scattering = np.interp(self.wavelengths, self.wavelengths_quantities, self.c_scattering)
        return interpolated_scattering / interpolated_extinction

    def __calculate_hyperspectral_asymmetry_parameters(self):
        return np.interp(self.wavelengths, self.wavelengths_quantities, self.g)

'''from pyRT_DISORT.preprocessing.model import AerosolProperties, Aerosol
file = ExternalFile('/home/kyle/repos/pyRT_DISORT/pyRT_DISORT/data/planets/mars/aux/dust.npy')
wavs = np.array([1, 9.3])
sizes = np.array([1, 1.2, 1.4])
dwaveref = np.array([9.1, 9.2, 9.3])
aero = Aerosol(file.array, wavs, sizes, dwaveref)
print(aero.extinction)
print(type(aero))
print(type(aero).__bases__)
#oldaero = OldAerosol(file.array, wavs, 9.2)
#print(oldaero.extinction_ratios)'''
from pyRT_DISORT.preprocessing.model.aerosol import AerosolProperties, Aerosol
from pyRT_DISORT.preprocessing.utilities.create_fits import CreateFits

file = ExternalFile('/home/kyle/repos/pyRT_DISORT/pyRT_DISORT/data/planets/mars/aux/dust_properties.fits')

data = file.array['primary'].data
wavs = file.array['wavelengths'].data
sizes = np.linspace(1, 2, num=10)

w = np.array([1, 9.3])
r = np.array([1, 1.2, 1.4])
ref = np.array([9.1, 9.2, 9.3])

# Make fake 3D data for testing
fakeData = np.zeros((10, len(wavs), 3))
for i in range(10):
    fakeData[i, :, :] = data * ((i+1)/10)

#asdf = CreateFits(fakeData)
#asdf.add_image_hdu(wavs, 'wavelengths')
#asdf.add_image_hdu(sizes, 'sizes')
#asdf.save_fits('/home/kyle/junk3D.fits')

#aero = Aerosol(fakeData, r, w, ref, wavelength_grid=wavs, particle_size_grid=sizes)
aero = Aerosol(data, r, w, ref, wavelength_grid=wavs)
print(aero.single_scattering_albedo)
