Controller
==========

Now's the time we should should set some of the controlling parameters of the
model. If you come up with a better module name, I'd be happy to change this.
These variables will be necessary for defining the last part of our model.

Computational Parameters
------------------------
We need to set a number of computational parameters. Let's do that with
:class:`controller.ComputationalParameters`. We can just plug the number of
layers inferred from the number of altitudes to use from the equation of state.
Let's then use 64 moments, 16 streams, 1 polar and azimuthal angle, and 80
user levels.

.. literalinclude:: example_simulation.py
   :language: python
   :lines: 40-44

This class just holds all of these parameters and checks they're allowable.

Model Behavior
--------------
Let's also define how we want our model to run. We can do that we
:class:`controller.ModelBehavior`, which has some preset values.

.. literalinclude:: example_simulation.py
   :language: python
   :lines: 40-44
