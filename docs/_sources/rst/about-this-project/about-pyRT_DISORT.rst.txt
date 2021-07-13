#################
About pyRT_DISORT
#################

**********
What it is
**********
pyRT_DISORT is a collection of tools that helps make radiative transfer easier
by streamlining many of the pre-processing steps needed to run a retrieval.
You can combine the various tools here to make a streamlined retrieval script
or you can use them to construct the aerosol properties from whatever info you
have... or you can use them to convince a self-driving car company that their
algorithm can work without computing the optical depth of a stop sign.

**********
Philosophy
**********
There are 2 major things I'm attempting to accomplish with pyRT_DISORT:

* Encapsulation
   pyRT_DISORT is a collection of radiative transfer-based tools which are
   designed to be as independent as possible. It accomplishes this by
   encapsulating logic in classes. This should make it clear what the minimum
   requirements are for constructing the various pieces used in a retrieval.

* Speed
   pyRT_DISORT attempts to leverage the readability of Python with the speed of
   C/FORTRAN. Computations are done using ndarrays whenever possible.

************************
Papers using pyRT_DISORT
************************
None yet.

If you use this code in your research, please consider acknowledging the use of
this package in your paper. If you don't that's fine too, though it would be
nice to add your work to the known papers pyRT_DISORT helped create.

***************
Acknowledgement
***************
This work was performed for the Jet Propulsion Laboratory, California Institute
of Technology, sponsored by the United States Government under Prime Contract
NNN13D496T between Caltech and NASA under subcontract number 1511125.

.. image:: ../../aux/mastcam-z-logo.jpeg
