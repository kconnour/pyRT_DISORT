Installation
============

To install pyRT_DISORT, you must have gfortran installed on your computer.
Assuming you have that, simply clone the repo and once you cd into that
directory simply install it with :code:`pip install .`. By default, this will
create two importable packages:

1. :code:`disort`
   This is a binarized version of the standard DISORT FORTRAN code (with very
   minor modifications for this project). This can take quite some time to
   install and since DISORT is not frequently updated, you can skip this step
   by setting :code:`install_disort=False` within :code:`setup.py`.
2. :code:`pyRT_DISORT`
   This is the pure Python project that will do the atmospheric pre-processing.

You can now import the libraries with :code:`import disort` and/or
:code:`import pyRT_DISORT as pyrt`.
