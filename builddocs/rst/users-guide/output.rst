Output
======
We're nearly done with our simulation. The last thing we need to do before we
can run DISORT is create some output parameters and arrays.

Output Arrays
-------------
Let's start by making the output arrays. We only need a few parameters to
create the arrays (they define the shape of the arrays), and we've already made
the parameters we'll need. Let's add them to :class:`output.OutputArrays`.

.. literalinclude:: example_simulation.py
   :language: python
   :lines: 40-44

All the arrays are set to 0 and will be populated as DISORT runs.

User Levels
-----------
We need an oddball variable to make the user levels. We can do that here.

.. literalinclude:: example_simulation.py
   :language: python
   :lines: 40-44

This is a variable you need to completely create yourself if your run will
use it; otherwise pyRT_DISORT will make an array of 0s of the correct shape
to appease DISORT.

Output Behavior
---------------
The final thing we need to do is set some switches to define how DISORT will
run. This is done with :class:`output.OutputBehavior`, which sets some default
values for how DISORT should run, but as usual you're free to override them.

.. literalinclude:: example_simulation.py
   :language: python
   :lines: 55-59


