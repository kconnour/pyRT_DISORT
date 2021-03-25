import numpy as np
from pyRT_DISORT.observation import Angles, Spectral

w = np.array([1, 2, 3])
s = Spectral(w, w+47.0001)

'''class Foo:
    def __init__(self, a):
        self.__a = a

    def __getattr__(self, method):
        return getattr(self.a, method)

    @property
    def a(self):
        return self.__a


f = Foo(np.linspace(0, 50, num=51))
print(np.amax(f))'''

