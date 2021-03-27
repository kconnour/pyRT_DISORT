The radiation module
====================
Let's now specify the flux and thermal quantities used in the model.

IncidentFlux
------------
We can define the beam flux and the isotropic flux at the top of the atmosphere
in the :class:`~radiation.IncidentFlux` class. Instances of this class don't do
anything but containerize your desired flux, and provide some default values.

.. code-block:: python

   from pyRT_DISORT.radiation import IncidentFlux

   flux = IncidentFlux()

   FBEAM = flux.beam_flux
   FISOT = flux.isotropic_flux

By default, :code:`beam_flux` is pi, and :code:`isotropic_flux` is 0. At least
for our Martian simulation, there's no real need to worry about the isotropic
flux from space.

ThermalEmission
---------------
We can also define whether thermal emission is used in the model with the
:class:`~radiation.ThermalEmission` class. As before, pyRT_DISORT has default
values to get you up and running, though you can easily override them.

.. code-block:: python

   from pyRT_DISORT.radiation import ThermalEmission

   te = ThermalEmission()

   PLANK = te.thermal_emission
   BTEMP = te.bottom_temperature
   TTEMP = te.top_temperature
   TEMIS = te.top_emissivity

By default, :code:`thermal_emission` is set to :code:`False` and no other
values are used. But
you're free to turn on thermal emission and add values for the boundaries.
