## pyRT_DISORT
This project aims to improve the lives of anyone performing retrievals with 
DISORT by combining the readability of Python with the speed of FORTRAN. It
provides a numpy-like toolset for creating all the inputs required by DISORT
from a set of aerosol property files. It allows the user to easily swap
what surface parameterization, aerosol vertical profile, etc. they want to use
in the model and then computes all of the input arrays.

## Documentation
Check out the documentation [here](https://kconnour.github.io/pyRT_DISORT/)

### Cautionary note
This project is under active development and is _unstable_ so many aspects of
the implementation are subject to change. 

### Installation
After cloning this repo, simply install it with pip: `pip install .` This will
build an importable `disort.so` file so that DISORT can be run directly from
Python. Note that this can be switched off within `setup.py`. The rest of the 
Python code will be importable via `pyRT_DISORT`. 

### Conventions
The physical units follow the following conventions:
- Altitudes are in km
- Wavelengths are in microns
- Wavenumbers are in 1/cm
- Everything else is in MKS units

### Miscellaneous
The current DISORT distribution site [here](http://www.rtatmocn.com/disort/),
should you want to look at the source code yourself.
