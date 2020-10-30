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


short_wavs = [0.2, 0.255, 0.3, 0.336, 0.4, 0.5, 0.588, 0.6, 0.631, 0.7, 0.763, 0.8, 0.835, 0.9, 0.953]
wavs = np.linspace(1, 50, num=(50-1)*10+1)
wavs = np.concatenate((short_wavs, wavs))

radii = np.array([1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 80])



