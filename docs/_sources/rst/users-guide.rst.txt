User's Guide
============
This user's guide will walk you through how to install pyRT_DISORT and use it
to simulate reflectance spectra of an atmosphere containing Martian dust. Then,
we'll use these simulations to parallelize our code and perform a retrieval.
I'll also demonstrate some not-strictly-necessary features included in
pyRT_DISORT that may make your life easier.

Each step will show off one of the modules included in pyRT_DISORT. For an
in-depth look at them, check out :doc:`api-reference`.

.. note::
   The modules in pyRT_DISORT are designed to operate independently---that is,
   (generally speaking) you won't need to input a class instance into another
   class. All the relevant classes can be instantiated from scratch, and you
   can choose to completely skip making some of these classes to your heart's
   content. That also means that the order in which you make these modules
   doesn't matter. I'm going to go through them in a way that's sensible to me,
   but there's no drawback to ordering them a different way when you adapt them
   to your own code.

   Additionally, it may seem kinda silly that I make a class and then
   pipe an array it created into another class, but it's quite possible that
   you won't want to do every time. If you do, you can always create a function
   that does the piping.

.. toctree::
   :maxdepth: 1
   :caption: A sample retrieval

   users-guide/installation
   users-guide/observation
   users-guide/eos
   users-guide/rayleigh
   users-guide/vertical_profile
   users-guide/forward_scattering
   users-guide/optical_depth
   users-guide/phase_function
   users-guide/model_build
   users-guide/controller
   users-guide/surface
   users-guide/radiation
   users-guide/output
   users-guide/running_the_model
