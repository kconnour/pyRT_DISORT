The optical_depth module
========================
We now have all the pieces we need to compute the optical depth in each of the
layers. For reference, this will compute an aerosol's contribution to
:code:`DTAUC`. Let's import :class:`~optical_depth.OpticalDepth` and give it
the correct inputs

.. code-block:: python

   from pyRT_DISORT.optical_depth import OpticalDepth

   od = OpticalDepth(np.squeeze(conrath.profile),
        hydro.column_density, nnssa.make_extinction_grid(9.3), 1)

This will create an object where to the column integrated optical depth is 1 at
9.3 microns. It will scale this value to the input wavelengths. We can verify
this value with the following:

.. code-block:: python

   print(np.sum(od.total, axis=0))

which outputs :code:`[1.87258736 1.93294559 1.55474492 1.16180818 0.80305085]`,
which is simply the ratio of the extinction coefficients at 9.3 microns to the
extinction coefficient at each of our input wavelengths *at the input particle
sizes*.

