The atmosphere module
=====================
We've done the hard work of creating all the atmospheric arrays for the
individual constituents. Now we just need to put everything together.

Atmosphere
----------
We can construct the arrays of the optical depth, single scattering
albedo, and phase function with :class:`~atmosphere.Atmosphere`. It requires
tuples of each of the 3 arrays for each atmospheric
constituent. I'll go ahead and make these tuples for Rayleigh scattering and
dust

.. code-block:: python

   rayleigh_info = (rayleigh_od, rayleigh_ssa, rayleigh_pf)
   dust_info = (dust_od, dust_ssa, dust_pf)

We can now add these to ``Atmosphere``, which will go ahead and construct the
composite arrays.

.. code-block:: python

   from pyRT_DISORT.atmosphere import Atmosphere

   model = Atmosphere(rayleigh_info, dust_info)

   DTAUC = model.optical_depth
   SSALB = model.single_scattering_albedo
   PMOM = model.legendre_moments

That's all there is to it. We now have our atmospheric arrays. The remaning
modules are generally small and simply help to construct some of the switches
required by DISORT.
