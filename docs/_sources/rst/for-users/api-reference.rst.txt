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

Forward scattering
------------------
These describe utilities for working with forward scattering properties.

.. autosummary::
   :toctree: generated/

   extinction_ratio
   optical_depth
   regrid

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

Vertical profiles
-----------------
These describe utilties for working with vertical profiles.

.. autosummary::
   :toctree: generated/

   conrath

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

Output arrays
-------------
These describe utilities for constructing output arrays.

.. autosummary::
   :toctree: generated/

   empty_albedo_medium
   empty_diffuse_up_flux
   empty_diffuse_down_flux
   empty_direct_beam_flux
   empty_flux_divergence
   empty_intensity
   empty_mean_intensity
   empty_transmissivity_medium

Surface arrays
--------------
These describe utilities for constructing arrays of the surface.

.. autosummary::
   :toctree: generated/

   make_hapke_surface
   make_hapkeHG2_surface
   make_hapkeHG2roughness_surface
