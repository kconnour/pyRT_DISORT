The surface module
==================
With the number of computational parameters defined, we can now make the
arrays of the surface reflectance. For this example, I'll stick to a Lambertian
surface.

.. code-block:: python

   from pyRT_DISORT.surface import Lambertian

   lamb = Lambertian(0.5, cp.n_streams, cp.n_polar, cp.n_azimuth, ob.user_angles,
                     ob.only_fluxes)
   ALBEDO = lamb.albedo
   LAMBER = lamb.lambertian
   RHOU = lamb.rhou
   RHOQ = lamb.rhoq
   BEMST = lamb.bemst
   EMUST = lamb.emust
   RHO_ACCURATE = lamb.rho_accurate
