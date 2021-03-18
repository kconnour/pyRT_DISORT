Physical Grid
=============

Now that we created the angular and spectral information, we can turn our
attention to creating the model. Perhaps the most natural place to start is
by defining the boundaries of the model. At each of the boundaries, we'll also
want to know the equation of state variables.

Hydrostatic
-----------
Suppose we have a temperature and pressure profile, or a set of profiles,
and the altitude where those quantities are defined. If we think the
atmosphere is in hydrostatic equilibrium, we can use :class:`~eos.Hydrostatic`
to compute the number density, column density, and scale height---just about
all the equation of state quantities we'd care about when doing a retrieval.

Let's start by making some profiles and defining some properties of the
atmosphere.

.. code-block:: python

   altitude_grid = np.linspace(100, 0, num=51)
   pressure_profile = 500 * np.exp(-altitude_grid / 10)
   temperature_profile = np.linspace(150, 250, num=51)
   mass = 7.3 * 10**-26
   gravity = 3.7

Here, we have great resolution of our profiles (2 km) but that doesn't mean
we necessarily want to run a retrieval with 50 layers in it. Let's specify
an altitude grid that defines the boundaries we actually want to use in the
retrieval.

.. code-block:: python

   z_grid = np.linspace(100, 0, num=15)

.. note::
   To keep with DISORT's convention that altitudes start from the top of the
   atmosphere, the input altitudes and grid must be *decreasing*.

We can now add these to :code:`Hydrostatic`. It will start by linearly
interpolating the input temperature and pressure onto the desired grid. Then,
it will compute number density, column density, and scale height at or within
the new boundaries. As before, we can access these arrays via the class
properties.

.. code-block:: python
   :emphasize-lines: 1

   hydro = Hydrostatic(altitude_grid, pressure_profile, temperature_profile, z_grid, mass, gravity)
   altitude = hydro.altitude
   pressure = hydro.pressure
   temperature = hydro.temperature
   number_density = hydro.number_density
   column_density = hydro.column_density
   n_layers = hydro.n_layers
   scale_height = hydro.scale_height


Most of these properties aren't required by DISORT (:code:`temperature` and
:code:`scale_height` are required under certain conditions) but several of
these variables will be needed in a few steps. Regardless, you may find a
number of these "unnecessary" variables to be handy when playing with your
retrievals.

.. note::
   As with the observational quantities, this accepts ND input so in theory
   if you have an image with MxN pixels and happen to know the
   pressures and temperatures at 50 points above each of the pixels, you can
   input 50xMxN arrays for :code:`altitude_grid`, :code:`pressure_grid`, and
   :code:`temperature_grid`and get the corresponding values. In this
   scenario, :code:`z_grid` should be ZxMxN where Z is the number
   of desired altitudes.

As you'd expect, the equation of state variables have the same shape as
:code:`z_grid`. The one exception is :code:`column_density` which is one
element shorter than the rest since it's only defined within each of the
*layers*. With that, we have our boundaries all good to go.

.. note::
   If you're lucky enough to already have the values for all of these
   quantities (like from a GCM) you can skip making this object and directly
   input these values later on.
