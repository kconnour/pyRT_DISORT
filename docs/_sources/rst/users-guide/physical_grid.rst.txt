Physical Grid
=============

Now that we created the angular and spectral information, we can turn our
attention to creating the model. Perhaps the most natural place to start is
by defining the boundaries of the model. At each of the boundaries, we'll also
want to know the equation of state variables. We'll also want to know the
column density within each of the layers for use later on.

Hydrostatic
-----------
Suppose you have a temperature and pressure profile, or a set of profiles,
and the altitude where those quantities are defined. If you think the
atmosphere is in hydrostatic equilibrium, you can use :class:`eos.Hydrostatic`
to compute the number density, column density,
and scale height---just about all the quantities you'd care about when doing
a retrieval. I'll go ahead and import :code:`Hydrostatic` and make some
profiles here, along with some miscellaneous variables.

.. literalinclude:: example_simulation.py
   :language: python
   :lines: 21-28

Even though you might have great resolution for the pressure and temperature
(here, 2 km), that doesn't mean that want to run a model with 50 layers in it.
That's where :code:`z_grid` comes in handy---it allows you to specify the
altitude grid you want the model to have. :code:`Hydrostatic` will regrid
these quantities onto :code:`z_grid`.

.. note::
   To keep with DISORT's convention that altitudes start from the top of the
   atmosphere, the input altitudes and grid must be *decreasing*.

Now we can add these to :code:`Hydrostatic` and access atmospheric properties
via the class properties:

.. literalinclude:: example_simulation.py
   :language: python
   :lines: 30-38

Most of these aren't required by DISORT (:code:`temperature` and
:code:`scale_height` are required under certain conditions) but several of
these variables will be needed in a few steps. Regardless, you may find a
number of these "unnecessary" variables to be handy when playing with your
retrievals.

.. note::
   As with the observational quantities, this accepts ND input so in theory
   if you have an image with :code:`MxN` pixels and happen to know the
   pressures and temperatures at 50 points above each of the pixels, you can
   input :code:`50xMxN` arrays and get the corresponding values. In this
   scenario, :code:`z_grid` should be :code:`ZxMxN` where Z is the number
   of desired altitudes.

As you'd expect, the equation of state variables have the same shape as
:code:`z_grid`. The one exception is :code:`column_density` which is one
element shorter than the rest since it's only defined within each of the
*layers*. That's all there is to it.

.. note::
   If you're lucky enough to already have the values for all of these
   quantities (like from a GCM) you can skip making this object and directly
   input these values later on.
