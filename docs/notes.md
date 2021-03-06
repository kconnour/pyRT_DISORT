## Intro
This is info I could only find after much StackExchange digging... It could
be a result of me just not knowing things though. If you install Sphinx, it
gets associated with the Python version that you installed it with. This means
that if you installed it with python3.8 and then write code in python3.9,
Sphinx will be unable (as far as I know...) to make documentation for the new
code. Plus, it's difficult for me to ensure I'm using the most recent version
of Sphinx.

## Sphinx quickstart
This is how you usually start a project with Sphinx. You'd usually do
`sphinx-quickstart` from Terminal but again I don't find that helpful. In code
form, `sphinx-quickstart` = `<python> -m sphinx.cmd.quickstart`. In my case 
this is `/home/kyle/repos/pyRT_DISORT/venv/bin/python -m sphinx.cmd.quickstart`

## Building docs
This is how you'd usually make the HTML documentation. You'd usually do
`make html` from Terminal but again I don't find that helpful. In code
form, `make html` = `<python> -m sphinx.cmd.build -b html <path to conf.py> 
<path to where to put the html files>`. In my case this is 
`/home/kyle/repos/pyRT_DISORT/venv/bin/python -m sphinx.cmd.build -b html /home/kyle/repos/pyRT_DISORT/docs /home/kyle/repos/pyRT_DISORT/docs/html`
