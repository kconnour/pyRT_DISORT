# Built-in imports
import os
from tempfile import mkdtemp

# 3rd-party imports
import numpy as np


class SharedArray:
    def __init__(self, shape, dtype=np.float, mode='w+'):
        self.filename = 'tmp_mmap'
        self.tmp_dir = mkdtemp()
        self.path = os.path.join(self.tmp_dir, 'myNewFile.dat')
        self.array = np.memmap(self.path, shape=shape, dtype=dtype, mode=mode)

    def delete(self):
        os.remove(self.path)
        os.rmdir(self.tmp_dir)
