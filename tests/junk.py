import numpy as np


class Junk:
    def __init__(self):
        self.junk_array = np.ones((5, 13))

    def dont_do_this(self):
        bad_thing_to_do = self.junk_array
        for i in range(8):
            bad_thing_to_do += 1


j = Junk()
print(j.junk_array)
j.dont_do_this()
print(j.junk_array)
