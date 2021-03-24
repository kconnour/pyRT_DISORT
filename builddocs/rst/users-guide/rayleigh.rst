The rayleigh module
===================
Now that we know the boundaries of our model, let's start building it. What
we'll do is essentially create atmospheric arrays for Rayleigh scattering, then
do the same thing with dust, and then combine them to get the total model
arrays.

These arrays, if you're curious, are

1. The optical depth in each layer (known as :code:`DTAUC`).
2. The single scattering albedo in each layer (known as :code:`SSALB`).
3. The Legendre decomposition of the phase function in each layer
   (known as :code:`PMOM`).

Rayleigh
--------
The single scattering albedo and phase function arrays are actually the easy
part of this process. All Rayleigh scattering has a single scattering albedo of
1, and the same it all has the same Legendre coefficient decomposition. To make
the arrays required by DISORT, we just need to know the number of layers to use
in the model.

.. note:: pyRT_DISORT also asks for the spectral shape so it can construct a
   phase function for each pixel. The value of this array is the same across
   all layers and pixels so you don't *need* to include a pixel shape, but
   it makes matrix math much simpler. This goes back to pyRT_DISORT accepting
   ND inputs.

While these quantities are same between multiple atmospheric species, their
optical depths will not be... and I can't possibly hope to create classes for
all the possible Rayleigh scattering species. Instead, I have an abstract
:class:`~rayleigh.Rayleigh` class that simply constructs the single scattering
albedo and phase function. You can extend this class to a species of your
liking. pyRT_DISORT comes with one example of this, as we'll see next.

RayleighCO2
-------------------
Mars has a CO :sub:`2` atmosphere, so let's instantiate a
:class:`~rayleigh.RayleighCO2` object. This will inherit the generic Rayleigh
single scattering albedo and phase function arrays, and will create the optical
depths in each of the model layers. We just need to provide it the wavelengths
at which to make the optical depths, and the column density in each of the
layers (the shapes that :code:`Rayleigh` accepts are inferred from these
arrays. We've already created these inputs, so we have everything we need to
make the relevant arrays.

.. code-block:: python

   from pyRT_DISORT.rayleigh import RayleighCO2

   rco2 = RayleighCO2(pixel_wavelengths[:, 0, 0], column_density)

   rayleigh_od = rco2.optical_depth
   rayleigh_ssa = rco2.single_scattering_albedo
   rayleigh_pf = rco2.phase_function

These arrays aren't particularly valuable by themselves but next we'll add
these to similar arrays created from dust to create the arrays DISORT wants.

.. note::
   If you're investigating and want to see the total optical depth due to
   Rayleigh scattering at your input wavelengths, you can use the line

   .. code-block:: python

      print(np.sum(rayleigh_od, axis=0))

   to see the column integrated optical depth. For this class it gives
   :code:`[1.62444356e-04 1.00391950e-05 1.97891739e-06 6.25591479e-07
   2.56207684e-07]`
