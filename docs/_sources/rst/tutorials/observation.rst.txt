The observation module
======================
Let's begin by considering some of the quantities found in a typical
observation---the angles and wavelengths at which some instrument collected
data. I'll go through two typical, distinct cases in this example in parallel:

1. "Rover": An image contains a single incidence angle but it images over
   M polar angles and N azimuthal angles.
2. "Spacecraft": Each pixel has a single incidence, emission, and phase angle.
   These values are different for each pixel. I call this the spacecraft case
   since this scenario would apply to a typical orbiter.

You can choose either one case or the other as an example for your case.
However, I recommend reading both cases for a complete discussion of the
code's behavior.

Rover Angles
------------
Let's say we have some angles defined over a grid. These values would normally
be found in a data product, but we need some values to play with. Let's suppose
we have an image of shape (40, 25), where we have 40 emission angles and 25
azimuthal angles. I'll define that here, along with the scalar incidence angle
and azimuthal angle of the incidence beam.

.. code-block:: python

   import numpy as np

   emission_angles = np.linspace(20, 50, num=40)
   azimuthal_angles = np.linspace(80, 30, num=25)
   incidence_angle = 35
   azimuth0 = 20

.. attention::
   The angles must be in degrees.

Our goal is to create an instance of :class:`~observation.Angles` to hold on to
all the values we'll need. Instead of creating this object directly, let's
use a function designed for this case---one that simply coerces these inputs
into a form that :code:`Angles` likes and returns an instance of it. We can
get the attributes from this object as shown below.

.. code-block:: python

   from pyRT_DISORT.observation import sky_image_angles

   angles = sky_image_angles(incidence_angle, emission_angles,
                             azimuthal_angles, azimuth0)

   incidence = angles.incidence
   emission = angles.emission
   mu = angles.mu
   mu0 = angles.mu0
   phi = angles.phi
   phi0 = angles.phi0

The shapes of both :code:`mu0` and :code:`phi0` are (1,), whereas :code:`mu`
has shape (1, 40) and :code:`phi` has shape (1, 25).
This class creates the angular variables that DISORT wants all at once and can
even compute all these variables at multiple incidence and beam azimuth angles.
Consequently, *you must pick the index for this set of angles* in order to get
something that DISORT wants. We only have one set of these angles so let's do
that below.

.. code-block:: python

   UMU = mu[0, :]
   UMU0 = mu0[0]
   PHI = phi[0, :]
   PHI0 = phi0[0]

Now the variables ending in 0 are floats and the others are 1D vectors, which
is precisely what DISORT wants.

.. note::
   For those of that have experience working with DISORT directly, I'll name
   the variables in this example with the same names that DISORT uses. For
   those unfamiliar with DISORT/FORTRAN, variables in ALL CAPS will be the ones
   that we ultimately plug into DISORT.

.. warning::
   I originally designed this example to only go through the spacecraft case,
   so some of the upcoming modules may not work well with this case. I will
   update them when I have the opportunity.

Spacecraft Angles
-----------------
Each pixel will presumably have its own unique combination of incidence,
emission, and phase angles that don't depend on wavelength. This data would
normally be found in a data product, but we need some values to play with
for the time being. We can go ahead and create some angles (for the simplicity
of the example let's
assume that the incidence, emission, and phase angles are all the same).

.. code-block:: python

   dummy_angles = np.outer(np.linspace(5, 10, num=15), np.linspace(5, 8, num=20))

Like the rover example, we can call a helper function that'll turn phase angles
into azimuthal angles and return an instance of :code:`Angles`.

.. code-block:: python

   from pyRT_DISORT.observation import angles_from_phase

   angles = angles_from_phase(dummy_angles, dummy_angles, dummy_angles)

   incidence = angles.incidence
   emission = angles.emission
   phase = angles.phase
   mu = angles.mu
   mu0 = angles.mu0
   phi = angles.phi
   phi0 = angles.phi0

In this case, the shapes of both :code:`mu0` and :code:`phi0` are (15, 20)---
the same shape as the input angles---whereas :code:`mu` and :code:`phi` both
have shapes (15, 20, 1). That's to say, each incidence angle has only 1 set
emission and azimuth angle. We can choose a single pixel index like below.

.. code-block:: python

   UMU = mu[0, 0, :]
   UMU0 = mu0[0, 0]
   PHI = phi[0, 0, :]
   PHI0 = phi0[0, 0]

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
