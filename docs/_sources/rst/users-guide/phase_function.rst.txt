The phase_function module
=========================
The last thing we need to do with dust is create its phase function array.
Right now I have some utilities for this that are very poorly designed and
poorly documented, but that'll change... Since they'll change, I won't spend
much time writing the user's guide on them.

For the time being, I'll be using
:class:`~phase_function.RadialSpectralTabularLegendreCoefficients` (I'm also
amenable to a better name...) to create a phase function array of shape
(n_moments, n_layers, n_wavelengths) by getting the nearest neighbor input
coefficients to a user-defined particle size grid and wavelength grid.


Let's import the module and construct the phase function array.

.. code-block:: python

   from pyRT_DISORT.phase_function import RadialSpectralTabularLegendreCoefficients

   dust_phsfn_file = fits.open('~/pyRT_DISORT/tests/aux/dust_phase_function.fits')
   coeff = dust_phsfn_file['primary'].data
   pf_wavs = dust_phsfn_file['wavelengths'].data
   pf_psizes = dust_phsfn_file['particle_sizes'].data
   pf = RadialSpectralTabularLegendreCoefficients(coeff, pf_psizes, pf_wavs, z_grid,
        spectral.short_wavelength[:, 0, 0], pgrad)

Then you can get the phase function in the :code:`phase_function` property.
