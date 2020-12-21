pyRT_DISORT makes atmospheric preprocessing much simpler when using the DISORT radiative
transfer code. 

Dependencies
============
- numpy
- pdoc3

Install
=======
Note that I've only tested this on Ubuntu.
- Install gfortran. On Ubunutu it's simple: sudo apt-get install gfortran
- Clone this repo from Github.
- cd into the cloned repo and run `pip install .`  This will build the disort.so file 
  and make the rest of the module importable.

Conventions
===========
The physical units follow the following conventions:
- Altitudes are in km
- Wavelengths are in microns
- Wavenumbers are in 1/cm
- Everything else is in MKS units

Miscellaneous
=============
I got DISORT at: http://www.rtatmocn.com/disort/
If you mess with the solar spectrum, its values are at 1AU
