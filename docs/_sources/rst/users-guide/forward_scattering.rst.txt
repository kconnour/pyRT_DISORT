The forward_scattering module
=============================
.. warning:: This is about the point I ran out of gas vectorizing things and
   writing good documentation. I need some more feedback before continuing
   too much further.

With our vertical profile defined, let's define the forward scattering
properties of our dust. We can do that with the obtusely named
:class:`~forward_scattering.NearestNeighborSingleScatteringAlbedo` (I'm open
to better names!). This class essentially just takes in the scattering and
extinction coefficients, along with the grid they're defined on, and regrids
them onto a user-defined particle size and wavelength grid by finding the
nearest neighbor values. pyRT_DISORT only really cares about the single
scattering albedo, so that's why the class is named as such...

I included the dust scattering properties in some tests, so let's just grab
the empirical values.

.. code-block:: python

   from astropy.io import fits

   prop = '~/pyRT_DISORT/tests/aux/dust_properties.fits'
   hdul = fits.open(f)
   cext = hdul['primary'].data[:, :, 0]
   csca = hdul['primary'].data[:, :, 1]
   wavs = hdul['wavelengths'].data
   psizes = hdul['particle_sizes'].data

Let's say we have some knowledge of the particle sizes to use in our model.
We can make them here too

.. code-block:: python

   gradient = np.linspace(1, 1.5, num=15)
   w = short_wavelength[:, 0, 0]  # Recall wavelength is 3D

We can now add these to the our very lengthy class name.

.. code-block:: python

   nnssa = NearestNeighborSingleScatteringAlbedo(csca, cext, psizes, wavs, w, gradient)

If you just want to play with things, I included the scattering and extinction
cross sections on this new grids as properties, along with the single scattering
albedo. I also have a method (*gasp!*) to make the extinction coefficient
at a reference wavelength. It will make an array of shape
(n_sizes, n_wavelengths) of the extinction coefficient at the reference
wavelength assuming the input particle sizes remain unchanged. This will be
needed for use in our next module.

.. note::
   I plan to add some more classes, like to linearly interpolate between values
   and not force you to use nearest neighbor, at some point. But hopefully this
   is serviceable for now.
