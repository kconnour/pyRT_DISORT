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


# An example
file = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/mars_atm.csv'
turn_csv_to_npy(file)

# Another example: modify a .npy file and save it
'''ice = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/ice.npy'
ice = np.load(ice, allow_pickle=True)
ice[:, 0] = ice[:, 0] / 1000
np.save('/home/kyle/repos/pyRT_DISORT/planets/mars/aux/ice.npy', ice)
print(ice[:, 0])'''

# Take a .npy file and turn it into a .csv
#dust = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/dust.npy'
#turn_npy_to_csv(dust, ['wavelength (microns)', 'c_extinction', 'c_scattering', 'kappa', 'g'])
