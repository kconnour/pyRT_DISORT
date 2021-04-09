About pyRT_DISORT
=================

What it is
----------
pyRT_DISORT is a collection of tools that helps make radiative transfer easier
by performing many of the pre-processing steps needed to create the arrays
required by DISORT. There's no one way to do radiative transfer, so this
package is designed to just provide tools to make your life easier, assuming
you combine them in a useful way.

Philosophy
----------
There are 3 major things I'm attempting to accomplish with pyRT_DISORT:

* Encapsulation
   pyRT_DISORT is a collection of radiative transfer-based tools which are
   designed to be as independent as possible. It accomplishes this by
   encapsulating logic in classes. This object oriented paradigm may scare
   scientists (the envisioned target users), but classes provide a convenient
   way to encapsulate all of the logic that goes into constructing the
   variables required by DISORT. It also makes it clear what the minimum
   requirements are for constructing the various pieces used in a retrieval,
   which should help anyone get up and running quickly.

* Independence
   The classes in pyRT_DISORT are designed to accept common input: ndarrays,
   floats, ints, etc. In other words, pyRT_DISORT doesn't require you to create
   instances of once class and put it into another class. So as long as you
   know the necessary quantities to compute a given variable or set of
   variables, you can do so directly.

* Speed
   pyRT_DISORT attempts to leverage the readability of Python with the speed of
   C/FORTRAN. Computations are done using ndarrays whenever possible.

History
-------
pyRT_DISORT originally started as a Python-based spiritual successor to
DISORT_MULTI---the brainchild of M.J. Wolff. I essentially transcribed its
logic into numpy and broke the code up into modules and classes.

Papers using pyRT_DISORT
------------------------
None yet.

If you use this code in your research, please consider acknowledging the use of
this package in your paper. If you don't that's fine too, though it would be
nice to add your work to the known papers pyRT_DISORT helped create.
