Spacecraft Retrieval
====================
This tutorial will walk you through how to simulate reflectance spectra of an
atmosphere containing Martian dust as observed from an orbiter. Then, we'll use
these simulations to perform a retrieval.

Each step will show off one of the modules included in pyRT_DISORT, though it
won't cover all of them. For an in-depth look at them, check out
:doc:`../api-reference`.

.. note::
   For efficiency, pyRT_DISORT does computations using
   `ndarray
   <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
   objects whenever possible.

The modules in pyRT_DISORT are designed to operate independently---that is,
generally speaking, you won't need to input a pyrt class instance into another
class. All the relevant classes can be instantiated from scratch, and you
can choose to completely skip making some of these classes to your heart's
content. That also means that the order in which you make these modules
doesn't matter for the most part. I'm going to go through them in a way that's
sensible to me, but there's no drawback to ordering them a different way when
you adapt them to your own code.

.. toctree::
   :maxdepth: 1
   :caption: A sample retrieval from orbit

   spacecraft_retrieval/observation
   spacecraft_retrieval/eos
   spacecraft_retrieval/rayleigh
   spacecraft_retrieval/aerosol
   spacecraft_retrieval/atmosphere
   spacecraft_retrieval/controller
   spacecraft_retrieval/radiation
   spacecraft_retrieval/output
   spacecraft_retrieval/surface
   spacecraft_retrieval/running_the_model
   spacecraft_retrieval/retrieval
