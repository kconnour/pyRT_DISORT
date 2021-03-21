The vertical_profile module
===========================
We added Rayleigh scattering so let's now start to make the optical depth
and phase function for dust. To do this, we'll first have to define the
vertical profile of dust. pyRT_DISORT provides a few tools to construct
special vertical profiles.

.. note:: If you have vertical profiles, say, from a GCM, you can just directly
   input these profiles in the later steps. This module will only help
   constructing the profiles.

Conrath
-------
Suppose you want to use a Conrath profile. :class:`~vertical_profile.Conrath`
provides the ability to construct a Conrath profile. For our retrieval, this
profile will be used to define the aerosol weighting within the *layers*. Let's
assume the midpoint altitudes are a good representation of the layer altitudes
and construct them here.

.. code-block:: python

   z_midpoint = ((z_grid[:-1] + z_grid[1:]) / 2)[:, np.newaxis]

As you're probably tired of hearing about for now, :code:`Conrath` can also
handle ND input. If we just want to a single profile, we need to make an array
of shape (n_altitude, n_pixels) (the :code:`[:, np.newaxis]` adds a dimension).
We should also define the relevant Conrath inputs

.. code-block:: python

   q0 = np.array([1])
   H = np.array([10])
   nu = np.array([0.01])

Let's now add these to :code:`Conrath`, and we can access the profile via its
profile property.

.. code-block:: python
   :emphasize-lines: 3

   from pyRT_DISORT.vertical_profile import Conrath

    conrath = Conrath(z_midpoint, q0, H, nu)
    profile = conrath.profile

This may feel clunky to only create one profile, but it allows you to
simultaneously create as many profiles as you'd like all at once. But that's
all there is to it

.. note:: The :code:`vertical_profile` module also comes with
   :class:`~vertical_profile.Uniform` to make constant mixing ratio profiles.
   This may be more applicable to water-ice clouds so we won't use it here,
   but it's worth mentioning its existence.
