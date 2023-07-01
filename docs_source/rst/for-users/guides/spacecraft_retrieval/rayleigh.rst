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

RayleighCO2
-------------------
Mars has a CO :sub:`2` atmosphere, so let's instantiate a
:class:`~rayleigh.RayleighCO2` object. This will create the aforementioned
arrays. We just need to provide it the wavelengths
at which to make the optical depths, and the column density in each of the
layers. Let's do that here.

.. code-block:: python

   from pyRT_DISORT.rayleigh import RayleighCO2

   rco2 = RayleighCO2(pixel_wavelengths, column_density)

   rayleigh_od = rco2.optical_depth
   rayleigh_ssa = rco2.single_scattering_albedo
   rayleigh_pf = rco2.phase_function

.. caution::
   These arrays have shapes (14, 5), (14, 5), and (3, 14, 5)---the same shapes
   DISORT expects for ``DTAUC``, ``SSALB``, and ``PMOM`` but with an extra
   wavelength dimension tacked on to the end. This class computed the arrays
   at all wavelengths at once, so don't get tripped up when computing these
   composite arrays.

We've now computed all of the quantities relevant to Rayleigh scattering.

.. tip::
   If you want to see the total optical depth due to Rayleigh scattering at
   the input wavelengths, you can execute the line

   .. code-block:: python

      print(np.sum(rayleigh_od, axis=0))

   to see the column integrated optical depth. For this example it gives
   ``[1.62444356e-04 1.00391950e-05 1.97891739e-06 6.25591479e-07 2.56207684e-07]``
