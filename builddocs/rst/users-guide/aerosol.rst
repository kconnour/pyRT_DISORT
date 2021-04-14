The aerosol module
==================
We just created 3 arrays for Rayleigh scattering; now, we need to make the same
arrays for dust.

Conrath
-------
First, we need to define a vertical volumetric mixing ratio profile for dust.
A Conrath profile was invented specifically for Martian dust, so let's use
:class:`~aerosol.Conrath` to make the profile. For our retrieval, this
profile will be used to define the aerosol weighting within the *layers*. Let's
assume the midpoint altitudes are a good representation of the layer altitudes
and construct them here.

.. code-block:: python

   z_midpoint = (z_grid[:-1] + z_grid[1:]) / 2

We can then set the Conrath parameters, and access the profile via the class
property

.. code-block:: python

   from pyRT_DISORT.aerosol import Conrath

   q0 = 1
   H = 10
   nu = 0.01

   conrath = Conrath(z_midpoint, q0, H, nu)

   dust_profile = conrath.profile

.. note::
   This module also comes with :class:`~aerosol.Uniform` to make constant
   mixing ratio profiles. This may be more applicable to water-ice clouds so we
   won't use it here, but it's worth mentioning its existence.

ForwardScattering
-----------------
The last preparation step we need to do is define the aerosol's forward
scattering properties. I assume that all forward scattering properties are a
function of both particle size and wavelength.

.. tip::
   If you have properties that aren't a function of both particle size and
   wavelength, you can use functions like
   `np.broadcast_to() <https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html>`_
   to make the arrays of the proper shape.

I include some dust scattering properties in the tests directory, so let's get
them, along with the grid over which they're defined.

.. code-block:: python

   from astropy.io import fits

   f = '~/pyRT_DISORT/tests/aux/dust_properties.fits'
   hdul = fits.open(f)
   cext = hdul['primary'].data[:, :, 0]
   csca = hdul['primary'].data[:, :, 1]
   wavs = hdul['wavelengths'].data
   psizes = hdul['particle_sizes'].data

Now, we simply need to define a grid of particle sizes where to get these
properties (and the wavelengths too, but we already defined these). We should
also define the wavelength reference---the wavelength to scale the extinction cross section.
This value will come in handy when creating the optical depth profile.

Finally, the particle sizes and wavelengths at which the properties are defined
aren't necessarily the ones we want to use, so we also need to include a grid
of particle sizes and wavelengths to regrid the forward scattering properties
onto. We can do this with :class:`~aerosol.ForwardScattering`.

.. caution::
   The particle size grid should be the same shape as the number of layers in
   the model. That's to say, each layer should have an associated particle
   size. 

.. code-block:: python

   particle_size_grad = np.linspace(1, 1.5, num=len(z_grid)-1)
   wave_ref = 9.3

   from pyRT_DISORT.aerosol import ForwardScattering

   fs = ForwardScattering(csca, cext, psizes, wavs, particle_size_grad,
                          pixel_wavelengths, wave_ref)

Before calling any methods, ``fs`` simply holds on to the inputs. It's now our
job to tell it *how* to grid the forward scattering properties onto this new
grid. Perhaps you want nearest neighbor interpolation, perhaps you want linear
interpolation, or perhaps you want something fancier. Just call the method that
tells it how to do the interpolation and then you can access the computed
properties. Here, I'll use nearest neighbor.

.. code-block:: python

   fs.make_nn_properties()

   nn_sca_cs = fs.scattering_cross_section
   nn_ext_cs = fs.extinction_cross_section
   dust_ssa = fs.single_scattering_albedo
   dust_ext = fs.extinction

.. caution::
   If you don't call a method, all of the properties will be empty arrays.

We've now defined the single scattering albedo at the nearest neighbor grid
points. Since we have the extinction (``dust_ext``) we can create the optical
depth array.

OpticalDepth
------------
Now that we have the extinction profile from :code:`fs`, we can make the array
of optical depth with :class:`~aerosol.OpticalDepth`. I'll plug in everything
and let it calculate. Essentially, all it needs to know to compute the optical
depth is the the vertical mixing ratio profile along with the column density in
each layer. It'll allocate the optical such that the total optical depth sums
up to the column integrated optical depth (which I set to 1 here), and is then
scaled to the reference wavelength that extinction was computed for (here,
9.3 microns when making ``dust_ext``).

.. code-block:: python

   from pyRT_DISORT.aerosol import OpticalDepth

   od = OpticalDepth(dust_profile, hydro.column_density, fs.extinction, 1)
   dust_od = od.total

With that, we computed the optical depth.

.. tip::
   As before, if you want to see the total optical depth due to dust at the
   input wavelengths, you can execute the line

   .. code-block:: python

      print(np.sum(dust_od, axis=0))

   to see the column integrated optical depth. For this example it gives
   ``[1.89162754 1.93270736 1.55633803 1.16197183 0.76995305]``. This is
   just the ratio of the extinction coefficient at the wavelength divided by
   the extinction coefficient at the reference wavelength, summed over all the
   layers.

TabularLegendreCoefficients
---------------------------
Now, we just need to make the phase function, which we can do with
:class:`~aerosol.TabularLegendreCoefficients`. This class will essentially work
the same way that ``ForwardScattering`` did---it will simply hang on to an
array of Legendre coefficients, the grid over which they're defined, and the
grid to regrid them onto. Let's go ahead and do that here.

.. code-block:: python

   from pyRT_DISORT.aerosol import NearestNeighborTabularLegendreCoefficients

   dust_phsfn_file = fits.open('~/pyRT_DISORT/tests/aux/dust_phase_function.fits')
   coeff = dust_phsfn_file['primary'].data
   pf_wavs = dust_phsfn_file['wavelengths'].data
   pf_psizes = dust_phsfn_file['particle_sizes'].data

   pf = NearestNeighborTabularLegendreCoefficients(coeff, pf_psizes, pf_wavs,
                                                   pgrad, pixel_wavelengths)

Like before, there are multiple ways to do the regridding. I'll again go with
nearest neighbor, then access the phase function via the property.

.. code-block:: python

   pf.make_nn_phase_function()

   dust_pf = pf.phase_function

To recap, we regridded the forward scattering properties to a grid via
nearest neighbor interpolation, which gave us the dust single scattering
albedo. We made vertical profile for dust, and with the nearest neighbor
extinction profile we computed the optical depth. Finally, we used nearest
neighbor interpolation to get the Legendre coefficients on our grid. We now
computed all of the arrays for dust!

.. note::
   In this example I used the tabulated Legendre coefficients from an empirical
   phase function, but suppose you want to use an analytic phase function like
   Henyey-Greenstein. In this case the Legendre coefficients are determined
   by the asymmetry parameter at particle sizes and wavelengths. You'd want to
   turn the asymmetry parameter into Legendre coefficients (see
   :class:`~aerosol.HenyeyGreenstein` to do this). After doing this we have an
   array that's functionally identical to the empirical coefficients we defined
   above---Legendre coefficients and the particle size and wavelength grid over
   which they're defined, so you can put the newly created Legendre coefficient
   array into ``TabularLegendreCoefficients`` and be good to go.
