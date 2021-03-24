Output
======
We're nearly done with our simulation. The last thing we need to do before we
can run DISORT is create some output parameters and arrays.

OutputArrays
-------------
Let's start by making the output arrays. We only need a few parameters to
create the arrays (they define the shape of the arrays), and we've already made
the parameters we'll need. Let's add them to :class:`~output.OutputArrays`.

.. code-block:: python

   from pyRT_DISORT.output import OutputArrays, OutputBehavior, UserLevel

   oa = OutputArrays(cp.n_polar, cp.n_user_levels, cp.n_azimuth)
   ALBMED = oa.albedo_medium
   FLUP = oa.diffuse_up_flux
   RFLDN = oa.diffuse_down_flux
   RFLDIR = oa.direct_beam_flux
   DRDT = oa.flux_divergence
   UU = oa.intensity
   UAVG = oa.mean_intensity
   TRNMED = oa.transmissivity_medium

All the arrays are set to 0 and will be populated as DISORT runs.

OutputBehavior
---------------
Let's now set some switches to define how DISORT will
run. This is done with :class:`~output.OutputBehavior`, which sets some default
values for how DISORT should run, but as usual you're free to override them.


.. code-block:: python

   ob = OutputBehavior()
   IBCND = ob.incidence_beam_conditions
   ONLYFL = ob.only_fluxes
   USRANG = ob.user_angles
   USRTAU = ob.user_optical_depths

UserLevel
---------
We need an oddball variable to make the user levels. We can do that here.

.. code-block:: python

   ulv = UserLevel(cp.n_user_levels)
   UTAU = ulv.optical_depth_output

This is a variable you need to completely create yourself if your run will
use it; otherwise pyRT_DISORT will make an array of 0s of the correct shape
to appease DISORT.
