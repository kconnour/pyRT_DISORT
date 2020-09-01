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


'''# An example
file = '/home/kyle/repos/pyRT_DISORT/planets/mars/aux/mars_atm.npy'
headers = ['Altitude (m)', 'Pressure (Pascal)', 'Temperature (K)']
turn_npy_to_csv(file, headers)'''
