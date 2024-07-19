Release notes
=============

v1.2.0
------
The major change is separating the FORTRAN from the Python code. These are now
in separate repositories, with the Python code living in this repository.
Consequently, the surface API is now updated so that any functions that would've
called disobrdf now just return a list of arguments to pass to disobrdf.

v1.0.0
------
Initial release! This update transitions pyRT_DISORT into a must lighter package
that integrates with numpy/scipy, instead of wrapping that functionality into
a framework. The entire API is new, so be sure to check out the API page for
more details.

..
   Warning: docutils 0.17 breaks bulleted lists! I had to downgrade to 0.16 to
   get the functionality back. See
   https://stackoverflow.com/questions/67542699/readthedocs-sphinx-not-rendering-bullet-list-from-rst-file

v0.0.4
------
This update continues documentation tweaks and incorporates bug fixes from
Mike that will make rover retrievals possible.

* Add a CITATION.cff file so we can be cited.
* Change the emission angles so they accept angles from 0 to 180 degrees,
  instead of 0 to 90 degrees. Update the documentation on this.
* Change incidence angles to only be from 0 to 90 degrees. Remove warning for
  angles >90 degrees. Update the documentation on this.
* Add the Henyey Greenstein asymmetry parameter to the forward scattering
  properties.
* Add a "docs" option to the installation script for help when making
  documentation.
* Add matplotlib extensions to conf.py to allow for future uses of pyplots.
* Update the documentation to use the pydata theme, instead of Read the Docs
  theme. This new theme is much cleaner.

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
