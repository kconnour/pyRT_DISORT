The model_atmosphere module
===========================
We've done the hard work of creating all the atmospheric arrays for the
individual constituents. Now we just need to put everything together.

ModelAtmosphere
---------------
We can construct the "big 3" arrays of the optical depth, single scattering
albedo, and phase function with :class:`~model_atmosphere.ModelAtmosphere`.
Let's make one of these objects (which takes no inputs to construct)

.. code-block:: python

   from pyRT_DISORT.model_atmosphere import ModelAtmosphere

   model = ModelAtmosphere()

With its :py:meth:`~model_atmosphere.ModelAtmosphere.add_constituent` method,
we can give it tuples of each of the 3
arrays for each atmospheric constituent. This object will just hold on to these
arrays. Let's make the inputs we made for Rayleigh scattering and dust.

.. code-block:: python

   rayleigh_info = (rco2.scattering_optical_depth, rco2.ssa, rco2.phase_function)
   dust_info = (od.total, nnssa.single_scattering_albedo, pf.phase_function)

   model.add_constituent(rayleigh_info)
   model.add_constituent(dust_info)

Now :code:`model` knows about everything it needs to know about. You can access
the total atmospheric properties via the class properties.

.. code-block:: python

   dtauc = model.optical_depth
   ssalb = model.single_scattering_albedo
   pmom = model.legendre_moments

That's all there is to it. And with that, we've done the hard part of
constructing our DISORT run. The upcoming modules will help us create some of
the remaining variables requested by DISORT.
