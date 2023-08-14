Spacecraft Retrieval
====================
This tutorial will walk you through how to simulate reflectance spectra of an
atmosphere containing Martian dust as observed from an orbiter. Then, we'll use
these simulations to perform a retrieval.

For all of this, I assume pyRT_DISORT is imported

.. code-block:: python

   import pyrt
   import disort

.. note::
   All code used in this example is collected in a script located at
   ~/pyRT_DISORT/docs_source/rst/for-users/guides/spacecraft_retrieval.py

.. note::
   Variables defined in all caps will be the ones that we ultimately plug into
   the DISORT call, and they adhere to the same naming convention that DISORT
   uses (for the benefit of those who have worked with DISORT before).

Angles
------
Let's begin by assuming we have a hyperspectral imager on an orbiter that takes
2D images such that the data have shape (15, 20) and each pixel in this grid
contains the same 5 wavelengths. Real data might have more pixels and more
wavelengths, but the scenario seems plausible enough.

Each pixel will have its own unique combination of incidence, emission, and phase
angles---angles that don't depend on wavelength. Let's create a set of angles
defined in each of these 300 pixels to use in this example (for simplicity of
the example, let's assume that all three of these angles are the same).

.. code-block:: python

   import numpy as np

   dummy_angles = np.outer(np.linspace(5, 10, num=15),
                           np.linspace(5, 8, num=20))

.. attention::
   The angles must be in degrees.

In this scenario, we need to make the azimuth angles of each pixel. We can do
that with :func:`~pyrt.azimuth`. Let's go ahead and do that, and create the
other angular variables needed.

.. code-block:: python

   UMU = np.cos(np.radians(dummy_angles))   # since dummy_angles is emission
   UMU0 = np.cos(np.radians(dummy_angles))   # since dummy_angles is incidence
   PHI = pyrt.azimuth(dummy_angles, dummy_angles, dummy_angles)
   PHI0 = np.zeros(PHI.shape)

This computes all of the angular quantities for the same time at each pixel.
DISORT expects the input of :code:`UMU0` and :code:`PHI0` to be floats, which
we can obtain by choosing the pixel's indices. It expects :code:`UMU` and
:code:`PHI` to both be 1D arrays (here, both are length 1 since, again, each
pixel has only 1 set of emission and azimuth angles) which we got the same way.

.. code-block:: python

   UMU = UMU[0, 0]
   UMU0 = UMU0[0, 0]
   PHI = PHI[0, 0]
   PHI0 = PHI0[0, 0]

.. note::
   In the case where DISORT wants a 1D array, you can give it a scalar value
   and f2py will convert it to an array when DISORT is called.

Wavelengths
-----------
Let's now turn our attention to the spectral information provided by the
instrument. I'll define some wavelengths so we have some values to work with.

.. code-block:: python

   pixel_wavelengths = np.array([1, 2, 3, 4, 5]) / 5
   spectral_width = 0.05

.. attention::
   The wavelengths must be in microns.

If we want the wavenumbers (if, for instance, we want to include thermal
emission) we can compute them here.

.. code-block:: python

   WVNMHI = pyrt.wavenumber(pixel_wavelengths - spectral_width)
   WVNMLO = pyrt.wavenumber(pixel_wavelengths + spectral_width)

These spectral quantities have shape (5,)---the same as the input wavelengths.
For now, I'll keep the spectral dimension but be aware that we'll cut off the
spectral dimension closer to when we do the simulation because DISORT requires
a separate call for each wavelength.

Equation of state
-----------------
Let's now start creating the atmospheric model. We'll start by creating the
model boundaries and equation of state variables.

Suppose we have a pressure and temperature profile for a given pixel, along with
the altitudes where these are defined.

.. code-block:: python

   altitude_grid = np.linspace(100, 0, num=15)
   pressure_profile = 500 * np.exp(-altitude_grid / 10)
   temperature_profile = np.linspace(150, 250, num=15)
   mass = 7.3 * 10**-26
   gravity = 3.7

.. attention::
   To keep with DISORT's convention that altitudes start from the top of the
   atmosphere, the altitude and altitude grid must be *decreasing*.

If the hydrostatic approximation is adequate, we can create the column density
in each model layer, which we'll happen to need to compute the Rayleigh
scattering optical depth in each model layer.

.. code-block:: python

   column_density = pyrt.column_density(pressure_profile, temperature_profile, altitude_grid)

We can also make some variables that DISORT needs in special cases.

.. code-block:: python

   TEMPER = temperature_profile
   H_LYR = pyrt.scale_height(temperature_profile, mass, gravity)

Rayleigh scattering
-------------------
Now that we know the boundaries of our model, let's start building it. What
we'll do is essentially create atmospheric arrays for Rayleigh scattering, then
do the same thing with dust, and then combine them to get the total model
arrays.

.. code-block:: python

   rayleigh_co2 = pyrt.rayleigh_co2(column_density, pixel_wavelengths)

This creates a Column, which is pyRT_DISORT's fundamental object. It collects
the optical depth, single scattering albedo, and phase function of each
atmospheric constituent. We can access these arrays via the object's properties.

.. code-block:: python

   rayleigh_co2.optical_depth
   rayleigh_co2.single_scattering_albedo
   rayleigh_co2.legendre_coefficients

These arrays have shapes (14, 5), (14, 5), and (3, 14, 5)---the same shapes
DISORT expects for ``DTAUC``, ``SSALB``, and ``PMOM`` but with an extra
wavelength dimension tacked on to the end. This class computed the arrays
at all wavelengths at once, so don't get tripped up when computing these
composite arrays.

.. tip::
   If you want to see the total optical depth due to Rayleigh scattering at
   the input wavelengths, you can execute the line

   .. code-block:: python

      np.sum(rayleigh_co2.optical_depth, axis=0)

   to see the column integrated optical depth. For this example it gives
   ``[1.62009710e-04 1.00123336e-05 1.97362248e-06 6.23917610e-07 2.55522160e-07]``

Aerosols
--------
We just created 3 arrays for Rayleigh scattering; now, we need to make the same
arrays for dust.

Vertical profile
****************
First, we need to define a vertical volumetric mixing ratio profile for dust.
Let's use a Conrath pfoiel. For our retrieval, this
profile will be used to define the aerosol weighting within the *layers*. Let's
assume the midpoint altitudes are a good representation of the layer altitudes
and construct them here.

.. code-block:: python

   altitude_midpoint = (altitude_grid[:-1] + altitude_grid[1:]) / 2

We can then set the Conrath parameters and construct a profile.

.. code-block:: python

   q0 = 1
   nu = 0.01

   dust_profile = pyrt.conrath(altitude_midpoint, q0, 10, nu)

With this profile, we can start to construct the dust's forward scattering
properties.

Forward scattering properties
*****************************
Next, we need the dust's forward scattering properties. I don't include any
forward scattering properties with pyRT_DISORT; instead I presume you've
computed these with some T-matrix computations. Normally, you'd read these in
but here I'll define some dummy properties.

.. code-block:: python

   particle_size_grid = np.linspace(0.5, 10, num=50)
   wavelength_grid = np.linspace(0.2, 50, num=20)
   extinction_cross_section = np.ones((50, 20))
   scattering_cross_section = np.ones((50, 20)) * 0.5

I leave it up to you to use scipy's `interpolation routines
<https://docs.scipy.org/doc/scipy/tutorial/interpolate.html>`_ if you want to
interpolate the properties onto another grid, but I'll assume what I defined
above is the result of the interpolation (or good enough).

We then need to define the particle size gradient for each model layer.

.. code-block:: python

   particle_size_gradient = np.linspace(1, 1.5, num=len(altitude_midpoint))

From this, we can now compute the dust's optical depth. Let's suppose
(presumably because someone else told us so) that the
column-integrated optical depth is 1 at 9.3 microns. We can compute the
extinction ratio between 9.3 microns and the wavelengths of our T-matrix grid,
regrid this array onto the altitude and wavelength grid of our model, and then
use atmospheric properties to get the optical depth in each model layer at each
wavelength.

.. code-block:: python

   ext = pyrt.extinction_ratio(extinction_cross_section, particle_size_grid, wavelength_grid, 9.3)
   ext = pyrt.regrid(ext, particle_size_grid, wavelength_grid, particle_size_gradient, pixel_wavelengths)
   dust_optical_depth = pyrt.optical_depth(dust_profile, column_density, ext, 1)

The variable ``dust_optical_depth`` has a shape of (14, 5), meaning it's the
optical depth of each model layer, computed at all model wavelengths.

The single scattering albedo is a bit simpler to compute.

.. code-block:: python

   dust_single_scattering_albedo = pyrt.regrid(scattering_cross_section / extinction_cross_section, particle_size_grid, wavelength_grid, particle_size_gradient, pixel_wavelengths)

Legendre coefficients
*********************
The Legendre coefficients essentially work the same way as above. I presume you
have these from the T-matrix computations, though I provide functions to
decompose phase functions into its Legendre moments. I also provide functions
for working with a Henyey-Greenstein phase function.

.. code-block:: python

   dust_pmom = np.ones((128, 50, 20))

   dust_legendre = pyrt.regrid(dust_pmom, particle_size_grid, wavelength_grid, particle_size_gradient, pixel_wavelengths)

Column
******
As a last step for dust, let's bundle all these properties together in a Column.

.. code-block:: python

   dust_column = pyrt.Column(dust_optical_depth, dust_single_scattering_albedo, dust_legendre)

Atmospheric model
-----------------
We've done the hard work of creating all the atmospheric arrays for the
individual constituents. Now we just need to put everything together, which we
can do by adding the columns together. We've constructed the columns for each
of the atmospheric constituents, so we just need to construct a composite
atmospheric model. All the composite arrays are stored in the object's
properties.

.. code-block:: python

   model = rayleigh_co2 + dust_column

   DTAUC = model.optical_depth
   SSALB = model.single_scattering_albedo
   PMOM = model.legendre_coefficients

Computational parameters
------------------------
We can now set a number of computational parameters. These aren't strictly
necessary, but are useful in the off chance we made an error when constructing
an input.

.. note::
   These are optional to pyRT_DISORT because it can infer them from array shapes
   when DISORT is called. They are not optional in the original FORTRAN
   implementation.

.. code-block:: python

   MAXCLY = len(altitude_midpoint)
   MAXMOM = PMOM.shape[0]
   MAXCMU = 16      # AKA the number of streams
   MAXPHI = 1
   MAXUMU = 1
   MAXULV = len(altitude_midpoint) + 1

The geometry of this situation dictates that the number of azimuth and polar
angles are both 1. The number of Legendre coefficients from the T-matrix
computations set the number of moments. The vertical grid set the number of
computational layers and user levels. The number of streams can be changed.

Model behavior
--------------
We can also set how we want the model to behave. These are some example values
but you can change these to your liking

.. code-block:: python

   ACCUR = 0.0
   DELTAMPLUS = True
   DO_PSEUDO_SPHERE = False
   HEADER = ''
   PRNT = None
   EARTH_RADIUS = 6371

Radiation
---------
Let's now specify the flux and thermal quantities used in the model.

Incident flux
*************
We can define the beam flux and the isotropic flux at the top of the atmosphere.

.. code-block:: python

   FBEAM = np.pi
   FISOT = 0

At least for our Martian simulation, there's no real need to worry about the
isotropic flux from space.

Thermal emission
****************
We can also define whether thermal emission is used in the model. For this
example, we'll ignore thermal emission but this code snippet shows how you
may define some variables. If no thermal emission is used, the other variables
can be any floats.

.. code-block:: python

   PLANK = False
   BTEMP = temperature_profile[-1]
   TTEMP = temperature_profile[0]
   TEMIS = 1

Output
------

Arrays
******
Next, we'll create some output arrays. DISORT evidently needs these empty arrays
to be initialized and it'll fill them as it runs.

.. code-block:: python

   ALBMED = pyrt.empty_albedo_medium(MAXUMU)
   FLUP = pyrt.empty_diffuse_up_flux(MAXULV)
   RFLDN = pyrt.empty_diffuse_down_flux(MAXULV)
   RFLDIR = pyrt.empty_direct_beam_flux(MAXULV)
   DFDT = pyrt.empty_flux_divergence(MAXULV)
   UU = pyrt.empty_intensity(MAXUMU, MAXULV, MAXPHI)
   UAVG = pyrt.empty_mean_intensity(MAXULV)
   TRNMED = pyrt.empty_transmissivity_medium(MAXUMU)

Behavior
********
We have yet more switches to tell DISORT how to run.

.. code-block:: python

   IBCND = False
   ONLYFL = False
   USRANG = True
   USRTAU = False

Surface
-------
With the number of computational parameters defined, we can now make the
arrays of the surface reflectance. Let's say we want to use a Hapke surface with
known values for its parameterization. We can use the variables we've previously
defined and these known parameters to compute the surface phase function arrays.

.. code-block:: python

    ALBEDO = 0  # this seems to only matter if the surface is Lambertian
    LAMBER = False

    b0 = 1
    h = 0.06
    w = 0.7

    RHOQ, RHOU, EMUST, BEMST, RHO_ACCURATE = pyrt.make_hapke_surface(USRANG,
    ONLYFL, MAXUMU, MAXPHI, MAXCMU, UMU, UMU0, PHI, PHI0, FBEAM, 200, b0, h, w)

Oddball
-------
This one doesn't fit in with the others in my mind. If we want the radiant
quantities to be returned at user-specified boundaries, we need to tell DISORT
what those boundaries are. If we don't specify these, it'll return them at the
layers of our model, which is perfectly fine in this case.

.. code-block:: python

   UTAU = np.zeros((MAXULV,))

Running the model
-----------------
We have all the variables we need to simulate some reflectance curves. Recall
that we've been making properties (where applicable) at all 5 wavelengths at
once. Unfortunately DISORT can't natively handle this, so we need to loop over
wavelength. I'll import the necessary module and create an array that'll be
filled as DISORT runs.

.. code-block:: python

   test_run = np.zeros(pixel_wavelengths.shape)

Once we carefully put all ~50 variables in the correct order and remember to
only select parts of the arrays we need (slicing off the wavelength dimension),
we can do simulations.

.. code-block:: python

   for ind in range(pixel_wavelengths.size):
       rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed = \
           disort.disort(USRANG, USRTAU, IBCND, ONLYFL, PRNT, PLANK, LAMBER,
                         DELTAMPLUS, DO_PSEUDO_SPHERE, DTAUC[:, ind], SSALB[:, ind],
                         PMOM[:, :, ind], TEMPER, WVNMLO, WVNMHI,
                         UTAU, UMU0, PHI0, UMU, PHI, FBEAM, FISOT,
                         ALBEDO, BTEMP, TTEMP, TEMIS, EARTH_RADIUS, H_LYR, RHOQ, RHOU,
                         RHO_ACCURATE, BEMST, EMUST, ACCUR, HEADER, RFLDIR,
                         RFLDN, FLUP, DFDT, UAVG, UU, ALBMED, TRNMED)

       test_run[ind] = uu[0, 0, 0]

   print(test_run)

This prints :code:`[0.11399675 0.06681997 0.06493594 0.06464735 0.06458137]`.

Retrieval
---------
We were able to make simulations in the previous section, but now let's suppose
we want to retrieve the dust optical depth (instead of treating it as a fixed
quantity). All that we really need to do is run simulations with a bunch of
different optical depths and see which one matches an observation the best
... though the implementation details aren't quite so crude. To demonstrate how
to do this, let's suppose the result of our simulation above where the dust
optical depth was 1 was a measured quantity and let's also pretend we have no
idea what the dust optical depth is.

The simulated spectrum acts as our measured I/F.

.. code-block:: python

   rfl = np.array([0.11399675, 0.06681997, 0.06493594, 0.06464735, 0.06458137])

If we want to retrieve a scalar optical depth over this spectral range
we want a function that accepts a test optical depth and finds the optical
depth that best fits this spectrum. The following function does that

.. code-block:: python

   def simulate_spectra(test_optical_depth):
       dust_optical_depth = pyrt.optical_depth(dust_profile, column_density, ext, test_optical_depth)
       dust_column = pyrt.Column(dust_optical_depth, dust_single_scattering_albedo, dust_legendre)
       model = rayleigh_co2 + dust_column

       od_holder = np.zeros(n_wavelengths)
       for wav_index in range(n_wavelengths):
           rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed = \
                disort.disort(USRANG, USRTAU, IBCND, ONLYFL, PRNT, PLANK, LAMBER,
                         DELTAMPLUS, DO_PSEUDO_SPHERE, DTAUC[:, wav_index], SSALB[:, wav_index],
                         PMOM[:, :, wav_index], TEMPER, WVNMLO, WVNMHI,
                         UTAU, UMU0, PHI0, UMU, PHI, FBEAM, FISOT,
                         ALBEDO, BTEMP, TTEMP, TEMIS, EARTH_RADIUS, H_LYR, RHOQ, RHOU,
                         RHO_ACCURATE, BEMST, EMUST, ACCUR, HEADER, RFLDIR,
                         RFLDN, FLUP, DFDT, UAVG, UU, ALBMED, TRNMED)

           od_holder[wav_index] = uu[0, 0, 0]
       return np.sum((od_holder - rfl)**2)

If I only want to retrieve the dust optical depth, nothing in this guide before the place where
I define the optical depth is affected; that is, the equation of state,
input angles, input wavelengths, etc. don't need modification. In fact, only
the atmospheric model cares about the different optical depths.
I create an empty array that will hold the values
as DISORT is run at each of the wavelengths (just like before) and populate
it along the way, but I only return the square of the distance from the target
spectrum. This essentially tells us how close the input optical depth is to the
target. If we minimize the output of this function, we found the best fit dust
optical depth.

Fortunately, scipy has some minimization routines for just this sort of thing.
I'll use their Nelder-Mead algorithm (a slower but robust algorithm) but feel
free to use another one if you've got more familiarity with them. All we need
to do is provide it a function to minimize (which we just defined) along with
an initial guess. This particular algorithm also has some bounds so I can tell
it not to try negative optical depths.

.. code-block:: python

   from scipy import optimize


   def retrieve_od(guess):
       return optimize.minimize(simulate_spectra, np.array([guess]),
                                method='Nelder-Mead', bounds=(0, 2)).x

Now let's do a retrieval where we guess that the optical depth is 1.5 and see
how it performs, along with how long it takes to run.

.. code-block:: python

   import time

   t0 = time.time()
   print(retrieve_od(1.5))
   t1 = time.time()
   print(f'The retrieval took {(t1 - t0):.3f} seconds.')

On my computer this outputs

.. code-block:: python

   [0.99997559]
   The retrieval took 0.351 seconds.

Seems decent to me.