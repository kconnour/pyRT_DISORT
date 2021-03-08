Sample Observation
==================

Many of the modules in pyRT_DISORT are designed to operate independently, so it
doesn't really matter what order you do many of these steps. I'm arbitrarily
choosing to start by making up some observations. Let's start by importing
classes from the relevant module:

.. literalinclude:: example_simulation.py
   :language: python
   :lines: 1-2

Spectral
--------
pyRT_DISORT attempts to keep the structure of your data wherever possible.
Let's consider the most generalized case: a hyperspectral imager. If the image
is :code:`MxN` pixels with :code:`W` wavelengths, the data will have shape
:code:`MxNxW`. I'll go ahead and make a fake spectral array. For simplicity,
I'll assume each pixel has the same wavelengths, although that's not a
necessity.

.. literalinclude:: example_simulation.py
   :language: python
   :lines: 4-6

.. note:: The wavelengths must be in microns.

I've defined wavelengths at 1, 2, 3, 4, and 5 microns, and the spectral width
of each bin is 0.1 microns. pyRT_DISORT's :class:`observation.Spectral`
class holds these
values and the associated wavenumbers of each pixel. These can be accessed via
the class properties at any time.

.. literalinclude:: example_simulation.py
   :language: python
   :lines: 8-12

All of these structures will have the same shape as the input wavelengths,
here (15, 20, 5).

Angles
------
Each pixel will presumably have its own unique combination of incidence,
emission, and phase angles. Let's again create some example angles, leaving
off the wavelength dimension since angles don't depend on wavelength. Once we
have
them (for the simplicity of the example I'm saying that the incidence,
emission, and phase angles are all the same) we can add them to
:class:`observation.Angles`, and then access all the angular quantities that
DISORT wants via the class properties.

.. literalinclude:: example_simulation.py
   :language: python
   :lines: 14-19

As before, these properties will have the same shape as the inputs, here
(15, 20).

.. note:: If you're reading in data from astropy's fits reader, all arrays
   are from the :code:`np.ndarray` class. My fake observation should therefore
   be structurally identical to how you'd apply this code to your data.
