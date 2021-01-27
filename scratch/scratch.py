import numpy as np
import matplotlib.pyplot as plt


a = np.load('/home/kyle/Downloads/data_increase.npy')
inc = a[:, 2] / a[:, 1]

plt.plot(a[:, 0], inc)
plt.xlabel('Oribt block')
plt.ylabel('Fractional increase in data volume')
plt.savefig('/home/kyle/increase.png')
per_inc = np.sum(a[:, 2]) / np.sum(a[:, 1])
print(per_inc)
bytes_inc = np.sum(a[:, 1]) * (1 + per_inc)
print((bytes_inc - np.sum(a[:, 1])) / 1000/ 1000/1000)
