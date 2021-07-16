Release notes
=============

v0.0.2
------
This version was focused on improving the observation module.

* Change observation.Angles to accept a different call signature, grouping
  the incident beam together instead of phi and phi0 together.
* Update documentation page on the observation module to be more like numpy.
  Now, the module has its own "home" page and each public class and function
  has its own page.
* Angles and Spectral now both contain __str__ and __repr__ methods.
* All classes and functions in the observation module now accept ArrayLike
  inputs, instead of strictly requiring ndarrays.
* Improved error handling by creating a number of abstract arrays to validate
  inputs.
* Minor speed and memory improvements when using this module. 
