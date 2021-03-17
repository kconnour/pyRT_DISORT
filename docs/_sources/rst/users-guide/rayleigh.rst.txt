Rayleigh
========
Let's start building our model atmosphere. The first thing we'll want to do
is include Rayleigh scattering in the model. To do that, we'll first need to
construct arrays of the optical depth and phase function for Rayleigh
scattering.

The generic Rayleigh class
--------------------------
The phase function is actually the easy part of this process. All Rayleigh
scattering has the same phase function. The phase function array is determined
solely by the number of layers to use in the model. This is done in the
abstract :class:`rayleigh.Rayleigh` class. I don't mean for you to actually
instantiate this class---instead, each atmospheric species will have its own
way of constructing the optical depths. If pyRT_DISORT doesn't come with the
species you're looking for (and right now it only comes with CO2), you'll want
to extend this class to the species for your application

Rayleigh CO2
------------
Mars has a CO2 atmosphere, so let's instantiate a :class:`rayleigh.RayleighCO2`
object. This will inherit the generic Rayleigh phase function and it will also
create the optical depths in each of the model layers. We just need to provide
it the model layers, the column density in the layers, and the wavenumbers
at which to make the optical depths (I choose wavenumbers instead of
wavelengths since the original paper uses them). We've already created these
inputs, so we have everything we need to make the Rayleigh arrays.

.. literalinclude:: example_simulation.py
   :language: python
   :lines: 40-44

Since Rayleigh optical depth is completely due to scattering, there's no
"total_optical_depth" property.
