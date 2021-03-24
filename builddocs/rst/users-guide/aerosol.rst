The aerosol module
==================
We just created 3 arrays for Rayleigh scattering; now, we need to make the same
arrays for dust.

Conrath
-------
Suppose you want to use a Conrath profile. :class:`~aerosol.Conrath`
provides the ability to construct a Conrath profile. For our retrieval, this
profile will be used to define the aerosol weighting within the *layers*. Let's
assume the midpoint altitudes are a good representation of the layer altitudes
and construct them here.

.. code-block:: python

   z_midpoint = ((z_grid[:-1] + z_grid[1:]) / 2)

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

NearestNeighborForwardScattering
--------------------------------
The last preparation step we need to do is define the aerosol's forward
scattering properties.

.. note::
   I assume that an aerosol's properties are a function of particle size and
   wavelength. If you have properties that aren't a function of both of those,
   you can use functions like `np.broadcast_to()
   <https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html>`_
   to make 3D arrays.

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
also define the wavelength reference---the wavelength to scale all values to.
This value will come in handy when creating the optical depth profile.

Finally, the particle sizes and wavelengths at which the properties are defined
aren't necessarily the ones we want to use, so we need to select how to how to
get values at undefined grid locations. For this example I'll use nearest
neighbor.

.. note::
   I made a generic :class:`!aerosol.Interpolator` class to do various different
   kinds of interpolation. This class is almost entirely a wrapper around that
   class.

.. code-block:: python

   particle_size_grad = np.linspace(1, 1.5, num=14)
   wave_ref = 9.3

   from pyRT_DISORT.aerosol import NearestNeighborForwardScattering

   fs = NearestNeighborForwardScattering(csca, cext, psizes, wavs, particle_size_grad,
                                         pixel_wavelengths, wave_ref)
   dust_ssa = fs.single_scattering_albedo

I won't list all of the properties, but this object allows you to see the
the input coefficients on this new grid. It also defines the single scattering
albedo, and extinction profile of this new grid. We've actually already defined
the single scattering albedo already, so let's now create the optical depth
array.

OpticalDepth
------------
Now that we have the extinction profile from :code:`nnfs`, we can make the array
of optical depth with :class:`~aerosol.OpticalDepth`. I'll plug in everything
and let it calculate. Essentially, all it needs to know to compute the optical
depth is the the vertical mixing ratio profile along with the column density in
each layer. It'll allocate the optical such that the total optical depth sums
up to the column integrated optical depth (which I set to 1 here), and is then
scaled to the wavelength reference that defined extinction.

.. code-block:: python

   from pyRT_DISORT.aerosol import OpticalDepth

   od = OpticalDepth(dust_profile, hydro.column_density, fs.extinction, 1)
   dust_od = od.total

With that, we have the optical depth computed.

NearestNeighborTabularLegendreCoefficients
------------------------------------------
Now, we just need to make the phase function. The general idea is that any
phase function will be defined on an input grid of particle sizes and
wavelengths. pyRT_DISORT comes with an abstract
:class:`~aerosol.TabularLegendreCoefficients` that houses some methods to help
create a phase function. But something more practical might be
:class:`~aerosol.NearestNeighborTabularLegendreCoefficients`. As before, we
just need to give this object the phase function, the particle size and
wavelength grid over which it's defined, and the particle size and wavelength
grid over which we want the phase function. Let's go ahead and do that here.

.. note::
   Suppose you have an analytic phase function like Henyey-Greenstein. Even in
   this case it depends on the asymmetry parameter, which is itself only
   empirically determined at certain sizes and wavelengths. In this case, we
   just need to convert the asymmetry parameter into Legendre coefficients,
   then we have an array that's functionally identical to empirical, tabular
   Legendre coefficients.

.. code-block:: python

   from pyRT_DISORT.aerosol import NearestNeighborTabularLegendreCoefficients

   dust_phsfn_file = fits.open('~/pyRT_DISORT/tests/aux/dust_phase_function.fits')
   coeff = dust_phsfn_file['primary'].data
   pf_wavs = dust_phsfn_file['wavelengths'].data
   pf_psizes = dust_phsfn_file['particle_sizes'].data

   pf = NearestNeighborTabularLegendreCoefficients(coeff, pf_psizes, pf_wavs,
                                                   pgrad, pixel_wavelengths)
   dust_pf = pf.phase_function

And before you know it, we've created all the things we need for dust!
