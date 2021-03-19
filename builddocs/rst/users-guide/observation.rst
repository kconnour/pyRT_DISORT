The observation module
======================
Let's begin by considering some of the quantities found in a typical
observation---the angles and wavelengths at which some instrument collected
data. For this example I'll consider the
most general case: a hyperspectral imager containing an MxN grid of pixels
that takes data at W wavelengths.

Angles
------
Each pixel will presumably have its own unique combination of incidence,
emission, and phase angles that don't depend on wavelength. This data would
normally be found in a data product, but we need some values to play with
for the time being. Let's start by importing the modules we'll need.

.. code-block:: python

   import numpy as np
   from pyRT_DISORT.observation import Angles, Spectral

We can go ahead and create some angles (for the simplicity of the example let's
assume that the incidence, emission, and phase angles are all the same).

.. code-block:: python

   dummy_angles = np.outer(np.linspace(5, 10, num=15), np.linspace(5, 8, num=20))

We can then add these to :class:`~observation.Angles`. This object simply holds
on to the input angles and computes the angular quantities required by DISORT.
You can access the input angles and these computed angular quantities via the
class properties, as shown below.

.. code-block:: python
   :emphasize-lines: 1

   angles = Angles(dummy_angles, dummy_angles, dummy_angles)
   incidence = angles.incidence
   emission = angles.emission
   phase = angles.phase
   mu = angles.mu
   mu0 = angles.mu0
   phi = angles.phi
   phi0 = angles.phi0

These newly created arrays have the same shape as the input
arrays---(15, 20)---allowing a 1-to-1 correspondence between them.

.. note:: If you're reading in data from astropy's fits reader, all arrays
   are from the :code:`np.ndarray` class. My fake observation should therefore
   be structurally identical to how you'd apply this code to your data.

Spectral
--------
If our hyperspectral imager contains W wavelengths and an image contains MxN
pixels, the data will have shape WxMxN (or at least it can be coerced into
that shape). I'll go ahead and make a spectral
array where each pixel took data at the same wavelength, although that's not a
necessity. I'll also define a spectral width to each bin.

.. code-block:: python

   dummy_wavelengths = np.array([1, 2, 3, 4, 5])
   pixel_wavelengths = np.broadcast_to(dummy_wavelengths, (20, 15, 5)).T
   width = 0.05

.. note:: The wavelengths must be in microns.

Once we have these values, we can add them to :class:`~observation.Spectral`.
This class holds the input wavelengths and computes the corresponding
wavenumbers. As before, these values can be accessed via the class properties.

.. code-block:: python
   :emphasize-lines: 1

   spectral = Spectral(pixel_wavelengths - width, pixel_wavelengths + width)
   short_wavelength = spectral.short_wavelength
   long_wavelength = spectral.long_wavelength
   high_wavenumber = spectral.high_wavenumber
   low_wavenumber = spectral.low_wavenumber

These spectral quantities have shape (5, 15, 20)---the same as the data.
Computing all of these values at once can lead to significant speed increases
when retrieving many quantities.

The only other thing you'd need from an observation is the signal your
instrument recorded. We won't need that value until much later on, so let's
wait until later to input those values.

.. note::
   The shape of pyRT_DISORT arrays is (# of moments, # of model layers,
   # of wavelengths, (# of pixels)). For instance, a hyperspectral imager can
   sometimes have 5D arrays, whereas a point spectrometer could have up to 3D
   arrays (dimensions of 1 are removed). pyRT_DISORT can
   handle ND observations (although if you have N > 2 you may consider a
   career in string theory instead of radiative transfer).
