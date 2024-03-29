Installation
============
.. note::
   The installation and unit tests are run using Python 3.10 and 3.11 on
   Ubuntu 20.04, Ubuntu 22.04, MacOS 11, MacOS 12, and MacOS 13. If these won't
   work for you, please raise an issue.

To install pyRT_DISORT, you must have FORTRAN installed on your computer. Once
you have that, simply clone the repo (using
:code:`git clone https://github.com/kconnour/pyRT_DISORT.git` from Terminal, or
clone using your favorite GUI) and move into the directory where it was cloned.
You can then install it with :code:`pip install .`. By default, this will
create two importable packages:

1. ``disort``
   This is a binarized version of the standard FORTRAN-based DISORT code. This
   can take some time to install.
2. ``pyrt``
   This is the pure Python package that will do the atmospheric pre-processing.

You can now import the libraries with :code:`import disort` and/or
:code:`import pyrt`.

.. warning::
   Some users have reported troubles with the installation. If you're having
   issues, check out `this issue
   <https://github.com/kconnour/pyRT_DISORT/issues/2>`_ to look for possible
   solutions as we work to get a more robust installation.

Using DISORT
------------
I use numpy's `f2py <https://numpy.org/doc/stable/f2py/>`_ utility to
generate importable FORTRAN code. I named the module :code:`disort` and all
subroutines included in the distribution are functions in that module. For
example, within LINPAK.f, the first subroutine is SGBCO. If you want to call
this directly, you can do so with :code:`disort.sgbco(<inputs>)`.

I imagine you'll only really be concerned with calling :code:`disort.disort`
but it's worth mentioning the whole distribution is available to you via the
:code:`disort` module.

.. warning::
   I made a minor modification to the source code in the shapes of :code:`RHOQ`
   and :code:`RHOU` (without this modification, the code could not run) so
   :code:`disort` isn't precisely the same as the official distribution. I'm
   working to resolve this discrepancy.


