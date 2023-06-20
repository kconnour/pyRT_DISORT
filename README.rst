.. image:: https://github.com/kconnour/pyRT_DISORT/workflows/CI/badge.svg?branch=main
     :target: https://github.com/kconnour/pyRT_DISORT/actions?workflow=CI
     :alt: CI Status

## pyRT_DISORT
pyRT_DISORT is a package for computing the arrays needed by DISORT. It 
provides:
* swappable vertical profiles, allowing streamlined construction of the
optical depth, single scattering albedo, and phase function arrays
* swappable surface profiles, including Lambertian and Hapke surface treatments
* generic radiative transfer utilities, such as decomposing empirical phase
functions into Legendre coefficients
* objects to create the flags required by DISORT
  
## Getting started
The best way to get started with pyRT_DISORT is to check out the documentation 
[here](https://kconnour.github.io/pyRT_DISORT/). It provides installation
instructions, example use cases, and documentation on everything included in
this project.

### Cautionary note
This project is under active development and is _unstable_ so many aspects of
the implementation are subject to change.

## Acknowledgement
This work was performed for the Jet Propulsion Laboratory, California Institute 
of Technology, sponsored by the United States Government under Prime Contract 
NNN13D496T between Caltech and NASA under subcontract number 1511125.