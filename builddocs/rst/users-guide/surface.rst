Surface
=======
With the number of computational parameters defined, we can now make the
arrays of the surface reflectance. For this example,


from pyRT_DISORT.surface import Lambertian

lamb = Lambertian(0.1, cp)
ALBEDO = lamb.albedo
LAMBER = lamb.lambertian
RHOU = lamb.rhou
RHOQ = lamb.rhoq
BEMST = lamb.bemst
EMUST = lamb.emust
RHO_ACCURATE = lamb.rho_accurate

