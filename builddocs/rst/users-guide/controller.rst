The controller module
=====================
We've created nearly all of the arrays that DISORT wants, so we now just need
to set some of the controlling parameters.

.. attention::
   To me, everything from this point onwards (with the exception of Surface)
   could be combined into one module. But I can't think of a name. If you can
   I'll happily make these less disjoined.

ComputationalParameters
------------------------
We need to set a number of computational parameters. Let's do that with
:class:`~controller.ComputationalParameters`. We can just plug the number of
layers inferred from the number of altitudes to use from the equation of state.
Let's then use 64 moments, 16 streams, 1 polar and azimuthal angle, and 80
user levels.

.. code-block:: python

   from pyRT_DISORT.controller import ComputationalParameters

   cp = ComputationalParameters(hydro.n_layers, model.legendre_moments.shape[0],
                                16, 1, 1, 80)

   MAXCLY = cp.n_layers
   MAXMOM = cp.n_moments
   MAXCMU = cp.n_streams
   MAXPHI = cp.n_azimuth
   MAXUMU = cp.n_polar
   MAXULV = cp.n_user_levels

.. note::
   All of the variables created in :code:`ComputationalParameters` are optional
   when using the :code:`disort` module, since it infers these values from
   array shapes. This class is completely optional, but I find it convenient to
   bundle all of these variables together.

Model Behavior
--------------
Let's also define how we want our model to run. We can do that with
:class:`~controller.ModelBehavior`, which has some preset values---namely, not
to do any pseudo spherical correction or delta-M correction. Of course, you
can change these to your liking.

.. code-block:: python

   from pyRT_DISORT.controller import ModelBehavior

   mb = ModelBehavior()
   ACCUR = mb.accuracy
   DELTAMPLUS = mb.delta_m_plus
   DO_PSEUDO_SPHERE = mb.do_pseudo_sphere
   HEADER = mb.header
   PRNT = mb.print_variables
   EARTH_RADIUS = mb.radius
