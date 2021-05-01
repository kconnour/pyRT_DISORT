The surface module
==================
With the number of computational parameters defined, we can now make the
arrays of the surface reflectance. We only need a handful of values to define
the shape of these arrays, so let's do that using :class:`~surface.Surface`.

.. code-block:: python

   from pyRT_DISORT.surface import Surface

   sfc = Surface(0.1, cp.n_streams, cp.n_polar, cp.n_azimuth, ob.user_angles,
                 ob.only_fluxes)

:code:`sfc` doesn't know what *kind* of surface it is. We can then set the type
of surface using the methods in :code:`Surface`. For simplicity, let's use a
Lambertian surface here. Once that's set, this class computes all the arrays it
needs.

.. code-block:: python

   sfc.make_lambertian()

   ALBEDO = sfc.albedo
   LAMBER = sfc.lambertian
   RHOU = sfc.rhou
   RHOQ = sfc.rhoq
   BEMST = sfc.bemst
   EMUST = sfc.emust
   RHO_ACCURATE = sfc.rho_accurate

With these defined, we've now created all the variables that DISORT needs!

.. tip::
   :code:`Surface` also comes with :py:meth:`~surface.Surface.make_hapke`,
   :py:meth:`~surface.Surface.make_hapkeHG2`, and
   :py:meth:`~surface.Surface.make_hapkeHG2_roughness` if you want to use more
   complicated phase functions. The 5 surface arrays are initialized with 0s
   when the class is instantiated. When these methods are called, those arrays
   are overridden.

   For a brief example, suppose you want to use a Hapke surface (without the
   surface HG phase function) and you know the Hapke parameters. You can do
   that with the following code:

   .. code-block:: python

      b0 = 1
      h = 0.5
      w = 0.5

      sfc.make_hapke(b0, h, w, UMU, UMU0, PHI, PHI0, FBEAM)

   and then all the arrays will be populated with values from a Hapke surface.

.. warning::
   Making the surface phase functions were the one place where I modified the
   DISORT source code. The shape of :code:`RHOU` seems wrong and inconsistent
   throughout the DISORT documentation. When I make it what I think it should
   be, my code then runs without error. However, it seems an error like this
   would've gone unnoticed, so be aware of this!