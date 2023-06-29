The observation module
======================
Let's begin by assuming we have a hyperspectral imager on an orbiter that takes
2D images such that the data have shape (15, 20) and each pixel in this grid
contains the same 5 wavelengths. Real data might have more pixels and more
wavelengths, but the scenario seems plausible enough.

Angles
------
Let's begin by considering the angles present in the data products. Each pixel
will have its own unique combination of incidence, emission, and phase
angles---angles that don't depend on wavelength. Let's create a set of angles
defined in each of these 300 pixels to use in this example (for simplicity of
the example, let's assume that all three of these angles are the same).

.. code-block:: python

   import numpy as np

   dummy_angles = np.outer(np.linspace(5, 10, num=15),
                           np.linspace(5, 8, num=20))

Our goal is to create an instance of :class:`~pyrt.observation.Angles` to hold
on to all the angular values DISORT wants. This class turns incidence and
emission angles into :math:`\mu_0` and :math:`\mu` and also holds on to azimuth
angles. We cannot directly instantiate this class because we have phase angles
and not azimuthal angles, but pyRT_DISORT comes with a helper function
(:func:`~pyrt.observation.phase_to_angles`) that creates azimuth angles and
returns an instance of Angles. Let's do this below
and look at the object's properties.

.. code-block:: python

   from pyrt.observation import phase_to_angles

   angles = phase_to_angles(dummy_angles, dummy_angles, dummy_angles)

   mu = angles.mu
   mu0 = angles.mu0
   phi = angles.phi
   phi0 = angles.phi0

.. attention::
   The angles must be in degrees.

In this case, the shapes of both :code:`mu0` and :code:`phi0` are (15, 20)---
the same shape as the input angles---whereas :code:`mu` and :code:`phi` both
have shapes (15, 20, 1). That's to say, each pixel has only 1 set of emission
and azimuth angles. We can then obtain appropriate values by choosing a pixel
index. If we want to get one of the pixel corners, we can do so as shown below.

.. code-block:: python

   UMU = mu[0, 0, :]
   UMU0 = mu0[0, 0]
   PHI = phi[0, 0, :]
   PHI0 = phi0[0, 0]

DISORT expects the input of :code:`UMU0` and :code:`PHI0` to be floats, which
we obtained by choosing the pixel's indices. It expects :code:`UMU` and
:code:`PHI` to both be 1D arrays (here, both are length 1 since, again, each
pixel ha only 1 set of emission and azimuth angles) which we got the same way.

We just computed all the angular quantities required by DISORT in all pixels of
the observation at once, which has the potential to offer some significant
computational time savings. Unfortunately DISORT can only accept inputs on a
pixel-by-pixel basis with this geometry, so for simplicity I'll stick to only
using a single pixel for the remainder of the example.

.. note::
   Variables defined in all caps will be the ones that we ultimately plug into
   the DISORT call, and they adhere to the same naming convention that DISORT
   uses (for the benefit of those who have worked with DISORT before).

Spectral
--------
Let's now turn our attention to the spectral information provided by the
imager and suppose that each spectral pixel had a width of 100 nm. I'll define
some wavelengths so we have some values to work with.

.. code-block:: python

   pixel_wavelengths = np.array([1, 2, 3, 4, 5])
   n_wavelengths = len(pixel_wavelengths)
   width = 0.1

.. attention::
   The wavelengths must be in microns.

Our goal is to create an instance of :class:`~pyrt.observation.Spectral` to
hold on to all the spectral values DISORT wants. This class will compute the
wavenumbers at the edges of each spectral bin. We could instantiate this class
directly, but let's use another helper function that comes with pyRT_DISORT
(:func:`~pyrt.observation.constant_width`) do the work for us and then look at
the object's properties.

.. code-block:: python

   from pyrt.observation import constant_width

   spectral = constant_width(pixel_wavelengths, width)

   WVNMHI = spectral.high_wavenumber
   WVNMLO = spectral.low_wavenumber

These spectral quantities have shape (5,)---the same as the input wavelengths.
For now, I'll keep the spectral dimension but be aware that we'll cut off the
spectral dimension closer to when we do the simulation because DISORT requires
a separate call for each wavelength.

Creating the wavenumbers isn't necessary unless we want to consider thermal
emission. We won't use thermal emission in this example, but now you're
familiar with how you'd create the wavenumbers should you want to use thermal
emission in the future.

The only other thing we need from an observation is the signal from the
instrument. We won't need that value until much later on, so let's wait until
later to input those values.
