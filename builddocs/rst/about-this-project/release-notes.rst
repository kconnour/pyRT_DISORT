Release notes
=============

..
   Warning: docutils 0.17 breaks bulleted lists! I had to downgrade to 0.16 to
   get the functionality back. See
   https://stackoverflow.com/questions/67542699/readthedocs-sphinx-not-rendering-bullet-list-from-rst-file

v0.0.4
------
* Add a CITATION.cff file so we can be cited.

v0.0.3
------
This update was focused on documentation tweaks and ease of use.

* Change the package name from pyRT_DISORT to pyrt. This should make it less
  cumbersome to import the package. This had a secondary effect of being
  Sphinx-compatible with autodoc. See `this issue
  <https://github.com/sphinx-doc/sphinx/issues/9479>`_ for details.
* All documentation now correctly display the package structure in the call
  signatures.
* Update the examples to reflect this new renaming scheme.
* Update the rover angles example. Most of that example will overlap with the
  spacecraft example so it wasn't a well conceived idea.
* Update the grammar used in the spacecraft retrieval to sound less like a
  closet hillbilly.

v0.0.2
------
This version was focused on improving the observation module.

* Change observation.Angles to accept a different call signature, grouping
  the incident beam together instead of phi and phi0 together.
* Remove the incidence and emission properties from Angles, as it's just a
  copy of the input to the class.
* Remove the short and long wavelength properties from Spectral, for the same
  reasons described above.
* Import all functions / classes in the global namespace for ease of use.
* Update documentation page on the observation module to be more like numpy.
  Now, the module has its own "home" page and each public class and function
  has its own page.
* Add a link to the quasi-source code in the signature of each class and
  function.
* Angles and Spectral now both contain __str__ and __repr__ methods.
* All classes and functions in the observation module now accept ArrayLike
  inputs, instead of strictly requiring ndarrays.
* Improved error handling by creating a number of abstract arrays to validate
  inputs.
* Improved docstring for the Angles class.
* Improved descriptions in the examples.
* Minor speed and memory improvements when using this module.
