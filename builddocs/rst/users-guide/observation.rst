The observation module
======================
Let's begin by considering some of the quantities found in a typical
observation---the angles and wavelengths at which some instrument collected
data. For this example I'll consider the most general case: a hyperspectral
imager containing an MxN grid of pixels that takes data at W wavelengths.

Angles
------
Each pixel will presumably have its own unique combination of incidence,
emission, and phase angles that don't depend on wavelength. This data would
normally be found in a data product, but we need some values to play with
for the time being. Let's start by importing the modules we'll need.

.. code-block:: python

   import numpy as np
   from pyRT_DISORT.observation import Angles

We can go ahead and create some angles (for the simplicity of the example let's
assume that the incidence, emission, and phase angles are all the same).

.. code-block:: python

   dummy_angles = np.outer(np.linspace(5, 10, num=15), np.linspace(5, 8, num=20))

We can then add these to :class:`~observation.Angles`. This object simply holds
on to the input angles and computes the angular quantities required by DISORT.
You can access the input angles and these computed angular quantities via the
class properties, as shown below.

.. code-block:: python

   angles = Angles(dummy_angles, dummy_angles, dummy_angles)

   incidence = angles.incidence
   emission = angles.emission
   phase = angles.phase
   mu = angles.mu
   mu0 = angles.mu0
   phi = angles.phi
   phi0 = angles.phi0

.. attention::
   The angles must be in degrees.

.. caution::
   It's not memory efficient to make a new variable that's simply a copy of a
   variable you already have as part of a class. I'm only showing you this to
   highlight the object's properties.

These newly created arrays have the same shape as the input
arrays---(15, 20)---so there's a 1-to-1 correspondence between them. This
vectorization allows us to compute all of the angular quantities across the
observation at once, which can provide some significant computational benefits.
In general, pyRT_DISORT can handle N-dimensional input to its classes; however,
I want to keep things somewhat simple for this example so for the remainder of
this retrieval I'll only consider quantities on a per pixel basis. Let's get
quantities from one of the pixels.

.. code-block:: python

   UMU = mu[0, 0]
   UMU0 = mu0[0, 0]
   PHI = phi[0, 0]
   PHI0 = phi0[0, 0]

.. note::
   For those of that have experience working with DISORT directly, I'll name
   the variables in this example with the same names that DISORT uses. For
   those unfamiliar with DISORT/FORTRAN, variables in ALL CAPS will be the ones
   that we ultimately plug into DISORT.

With that, we already computed 4 of the variables DISORT needs!

Spectral
--------
Let's assume that our hyperspectral imager takes data at W wavelengths in our
pixel and that there's a constant spectral width to each bin. I'll go ahead and
define some wavelengths here so we have some values to work with.

.. code-block:: python

   pixel_wavelengths = np.array([1, 2, 3, 4, 5])
   n_wavelengths = len(pixel_wavelengths)
   width = 0.05

.. attention::
   The wavelengths must be in microns.

Once we have these values, we can add them to :class:`~observation.Spectral`.
This class holds the input wavelengths and computes the corresponding
wavenumbers. As before, these values can be accessed via the class properties.

.. code-block:: python

   from pyRT_DISORT.observation import Spectral

   spectral = Spectral(pixel_wavelengths - width, pixel_wavelengths + width)

   short_wavelength = spectral.short_wavelength
   long_wavelength = spectral.long_wavelength
   WVNMHI = spectral.high_wavenumber
   WVNMLO = spectral.low_wavenumber

These spectral quantities have shape (5,)---the same as the input wavelengths.
For now, I'll keep the spectral dimension but be aware that we'll cut off the
spectral dimension closer to when we do the simulation.

The only other thing you'd need from an observation is the signal your
instrument recorded. We won't need that value until much later on, so let's
wait until later to input those values.
