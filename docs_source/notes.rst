:orphan:

Notes on Sphinx
===============
This is info I could only find after much StackExchange digging... It could
be a result of me just not knowing things though. If you install Sphinx, it
gets associated with the Python version that you installed it with. This means
that if you installed it with python3.8 and then write code in python3.9,
Sphinx will be unable (as far as I know...) to make documentation for the new
code. Plus, it's difficult for me to ensure I'm using the most recent version
of Sphinx.

Quickstart
----------
To get started with a project you'd usually run :code:`sphinx-quickstart` from
Terminal. In code form, :code:`sphinx-quickstart` is equivalent to
:code:`<python> -m sphinx.cmd.quickstart` where <python> is the absolute path
to your interpreter.

Building docs
-------------
When you're ready to build docs you'd usually run :code:`make html` from
Terminal. In code form, :code:`make html` =
:code:`<python> -m sphinx.cmd.build -b html <path to conf.py>
<path to where to put the html files>`. You can also add a :code:`-E` flag
to tell Sphinx to overwrite the old docs and rebuild them all each time. I
prefer this, since sometimes changes to the header of one file aren't
registered in other files.

Suppose pyRT_DISORT is in the repos directory in your home folder. The command
will look like:
:code:`~/repos/pyRT_DISORT/venv/bin/python -m sphinx.cmd.build -b html
~/repos/pyRT_DISORT/docs_source ~/repos/pyRT_DISORT/docs -E`
