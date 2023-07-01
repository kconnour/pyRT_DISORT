Retrieval
=========
We were able to make simulations in the previous section, but now let's suppose
we want to retrieve the dust optical depth (instead of treating it as a fixed
quantity). All that we really need to do is run simulations with a bunch of
different optical depths and see which one matches an observation the best
... though the implementation details aren't quite so crude. To demonstrate how
to do this, I ran this code with a a dust optical depth known only to me and
I'll show you how to get this value from what we've written.

Let's suppose we measured the following reflectance (I/F) spectrum at the input
set of wavelengths (this was the result of my simulation).

.. code-block:: python

   rfl = np.array([0.116, 0.108, 0.084, 0.094, 0.092])

If we want to retrieve a scalar optical depth over this spectral range
we want a function that accepts a test optical depth and finds the optical
depth that best fits this spectrum. The following function does that

.. code-block:: python

   def test_optical_depth(test_od):
       # Trap the guess
       if not 0 <= test_od <= 2:
           return 9999999

       od = OpticalDepth(dust_profile, hydro.column_density, fs.extinction,
                          test_od)

       dust_info = (od.total, dust_ssa, dust_pf)
       model = Atmosphere(rayleigh_info, dust_info)

       od_holder = np.zeros(n_wavelengths)
       for wav_index in range(n_wavelengths):
           rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed = \
                disort.disort(USRANG, USRTAU, IBCND, ONLYFL, PRNT, PLANK, LAMBER,
                              DELTAMPLUS, DO_PSEUDO_SPHERE,
                              model.optical_depth[:, wav_index],
                              model.single_scattering_albedo[:, wav_index],
                              model.legendre_moments[:, :, wav_index],
                              TEMPER, WVNMLO, WVNMHI,
                              UTAU, UMU0, PHI0, UMU, PHI, FBEAM, FISOT,
                              ALBEDO, BTEMP, TTEMP, TEMIS, EARTH_RADIUS, H_LYR, RHOQ, RHOU,
                              RHO_ACCURATE, BEMST, EMUST, ACCUR, HEADER, RFLDIR,
                              RFLDN, FLUP, DFDT, UAVG, UU, ALBMED, TRNMED)

           od_holder[wav_index] = uu[0, 0, 0]
       return np.sum((od_holder - rfl)**2)

This function is worth explaining. The :code:`if` statement allows me to
specify some physical boundaries---if the input optical depth is outside an
acceptable range, I return a huge value. Next, I realize that if I only want to
retrieve the dust optical depth, nothing in this guide before the place where
I define the optical depth is affected; that is, the equation of state,
input angles, input wavelengths, etc. don't need modification. In fact, only
the atmospheric model cares about the different optical depths.
Finally, I create an empty array that will hold the values
as DISORT is run at each of the wavelengths (just like before) and populate
it along the way, but I only return the square of the distance from the target
spectrum. This essentially tells us how close the input optical depth is to the
target. If we minimize the output of this function, we found the best fit dust
optical depth.

Fortunately, scipy has some minimization routines for just this sort of thing.
I'll use their Nelder-Mead algorithm (a slower but robust algorithm) but feel
free to use another one if you've got more familiarity with them. All we need
to do is provide it a function to minimize (which we just defined) along with
an initial guess.

.. code-block:: python

   from scipy import optimize


   def retrieve_od(guess):
       return optimize.minimize(test_optical_depth, np.array([guess]),
                                method='Nelder-Mead').x

Now let's do a retrieval where we guess that the optical depth is 1 and see how
long it takes to run.

.. code-block:: python

   import time

   t0 = time.time()
   print(retrieve_od(1))
   t1 = time.time()
   print(f'The retrieval took {(t1 - t0):.3f} seconds.')

On my aging computer this outputs

.. code-block:: python

   [0.2359375]
   The retrieval took 0.699 seconds.

I originally ran the simulation with an optical depth of 0.234 so our solver
worked pretty accurately. It was also pretty fast! Be aware that the time will
increase significantly if you're fitting multiple parameters, such as multiple
aerosol optical depths.

Hopefully you see the benefit of building a model piece by piece---you only
need to repeat the steps that change as you iterate over the parameter(s).
Radiative transfer has a reputation of being computationally intensive. This
won't change that, but it's an attempt to make your model as efficient as
possible. Next, we'll look at how to parallelize the code we just wrote (once
I'm feeling up for it).
