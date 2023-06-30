API Reference
=============
.. currentmodule:: pyrt

This page describes the pyRT_DISORT API, where the API is grouped by
functionality. Many docstrings contain practical example code.

Columns
-------
Perhaps the most important set of DISORT inputs is the optical depth, single
scattering albedo, and phase function in each layer of the model. You can
create these on a per-aerosol basis and store them in a Column object. These
can easily help you create the optical properties of the composite atmospheric
model.

.. autosummary::
   :toctree: generated/

   Column

Equation of state
-----------------
These describe utilities for working with equation of state variables.

.. autosummary::
   :toctree: generated/

   column_density

Phase function
--------------
These describe utilities for working with phase functions.

.. autosummary::
   :toctree: generated/

   decompose
   fit_asymmetry_parameter
   set_negative_coefficients_to_0
   construct_henyey_greenstein
   henyey_greenstein_legendre_coefficients

Rayleigh scattering
-------------------
These describe utilities for working with Rayleigh scattering.

.. autosummary::
   :toctree: generated/

   rayleigh_legendre
   rayleigh_co2

Angles
------
These describe utilities for working with angles.

.. autosummary::
   :toctree: generated/

   azimuth

Wavelengths
-----------
These describe utilities for working with wavelengths.

.. autosummary::
   :toctree: generated/

   wavenumber
