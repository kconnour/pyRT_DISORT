import pandas as pd
import numpy as np


def turn_csv_to_npy(file):
    df = pd.read_csv(file).to_numpy()
    path = file.split('.')[0]
    np.save('{}.npy'.format(path), df)


def turn_npy_to_csv(file, headers):
    path = file.split('.')[0]
    array = np.load(file, allow_pickle=True)
    pd.DataFrame(array).to_csv('{}.csv'.format(path), header=headers, index=False)


def turn_tmqphsfn_to_npy(file_path):
    """ Turn a tmq_mod1_r###v030_#####.dat.coef to a numpy array

    Parameters
    ----------
    file_path: str
        The Unix-like path to the file

    Returns
    -------
    np.ndarray
        The coefficients stored in a numpy array
    """
    coeff = []
    f = open(file_path)
    lines = f.readlines()[1:]
    for line in lines:
        a = np.fromstring(line.strip(), sep=' ')
        coeff.append(a)

    # This unravels a list and stores it as an array
    return np.array([co for all_coeff in coeff for co in all_coeff])




# An example
#file = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/mars_atm.csv'
#turn_csv_to_npy(file)

# Another example: modify a .npy file and save it
'''ice = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/ice.npy'
ice = np.load(ice, allow_pickle=True)
ice[:, 0] = ice[:, 0] / 1000
np.save('/home/kyle/repos/pyRT_DISORT/planets/mars/aux/ice.npy', ice)
print(ice[:, 0])'''

# Take a .npy file and turn it into a .csv
#dust = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/dust.npy'
#turn_npy_to_csv(dust, ['wavelength (microns)', 'c_extinction', 'c_scattering', 'kappa', 'g'])

#file = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/marsatm.inp'
#atm = np.genfromtxt(file, skip_header=3)
#atm[:, 1] = atm[:, 1] / 100

#atm = np.load('/home/kyle/repos/pyRT_DISORT/planets/mars/aux/marsatmNew.npy')
#p = np.pad(atm, ((0, 0), (0, 1)), mode='constant', constant_values=0)
#np.save('/home/kyle/repos/pyRT_DISORT/planets/mars/aux/marsatmNew.npy', p)

#file = '/home/kyle/disort_multi/aerosol_ice.dat'
#a = np.genfromtxt(file, skip_header=3)
#np.save('/home/kyle/repos/pyRT_DISORT/preprocessing/planets/mars/aux/ice.npy', a)
#turn_npy_to_csv('/home/kyle/repos/pyRT_DISORT/preprocessing/planets/mars/aux/ice.npy', ['Wavelengths (microns)', 'C_extinction', 'C_scattering', 'kappa', 'g', 'Pmax', 'Thetmax'])