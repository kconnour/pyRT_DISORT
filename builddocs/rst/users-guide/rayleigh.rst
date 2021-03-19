Rayleigh
========
Let's start building our model atmosphere. The first thing we'll want to do
is include Rayleigh scattering in the model. To do that, we'll first need to
construct arrays of the optical depth and Legendre decomposition of the phase
function for Rayleigh scattering (since DISORT only wants to deal with the
Legendre coefficients---not phase functions themselves).

The generic Rayleigh class
--------------------------
The phase function array is actually the easy part of this process. All
Rayleigh scattering has the same Legendre coefficient decomposition. To make
the array(s) required by DISORT, we just need to know the number of
layers to use in the model.

.. note:: DISORT wants a phase function array with shape
   (n_moments, n_layers)---even if the phase function is the same within all of
   the layers. It's important that we include the number of desired layers in
   this step.

While the phase function is the same between multiple atmospheric species,
their optical depths will not be... and I can't possibly hope to create
classes for all the possible Rayleigh scattering species. Instead, I have
an abstract :class:`~rayleigh.Rayleigh` class that simply constructs the
phase function. You can extend this class to a species of your liking.
pyRT_DISORT comes with one example of this, as we'll see next.

Rayleigh CO2
------------
Mars has a CO :math:`_2` atmosphere, so let's instantiate a
:class:`~rayleigh.RayleighCO2` object---a class that extends :code:`Rayleigh`
whose whole purpose is to hold the optical depth and phase function arrays for
Rayleigh scattering from CO2. This will inherit the generic Rayleigh
phase function and it will also create the optical depths in each of the model
layers. We just need to provide it the wavelengths at which to make the
optical depths and the column density in each of the layers. We've already created
these inputs, so we have everything we need to make the relevant arrays.

.. code-block:: python
   :emphasize-lines: 3

   from pyRT_DISORT.rayleigh import RayleighCO2

   rco2 = RayleighCO2(pixel_wavelengths, column_density)
   rayleigh_phase_function = rco2.phase_function
   rayleigh_od = rco2.scattering_optical_depth

Since Rayleigh optical depth is completely due to scattering, there's no
"total_optical_depth" property. These arrays aren't particularly valuable by
themselves but soon we'll add these to similar arrays created from dust to
create the arrays DISORT wants.

.. note:: I previously mentioned that pyRT_DISORT arrays generally have shape
   of (n_moments, n_layers, n_wavelengths, (n_pixels)). Our hyperspectral
   imager will produce 5D arrays here, although DISORT only wants 2D arrays.
   Be sure to keep the first two dimensions but select a specific dimension
   for all later dimensions; more on this to come. Since Rayleigh scattering
   is usually known (or fixed) beforehand, this allows you to compute all the
   Rayleigh scattering for an image at once.
