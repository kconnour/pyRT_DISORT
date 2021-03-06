****************
New user's guide
****************

Welcome to the new user's guide for pyRT_DISORT. This guide will walk you
through an example of how to perform a simulation in DISORT.

##########
Philosophy
##########
My idea is that I can't realistically design a retrieval code that accounts for
everything you want to do. Therefore I've made a variety of "swappable" parts
to help you build the model you want. For instance, you can use a
Henyey-Greenstein phase function, or swap it out for an empirical phase function
and the rest of the code will run unchanged. In addition, different instruments
will have different data structures. The goal is to retain the original
data structure when constructing the different pieces of the model. This makes
it easier to determine which arrays/objects correspond to a given pixel and
allows you to do more with fewer objects. This has the effect of increasing the
one time computational cost, but will reduce the computational cost when
iterating over multiple pixels (when retrieving best fit values, for instance).

############
Installation
############
Clone the repo onto your computer and run `pip install .`

##################
A first simulation
##################
