import numpy as np
from pyRT_DISORT.eos import Hydrostatic

z = np.linspace(100, -1, num=50)

import disort
print(disort.disort.__doc__)

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

