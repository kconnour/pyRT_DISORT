Using f2py

f2py is a wonderful utility but I found the examples were too simplistic... more to the point, I wanted to turn the DISORT subroutine (found in DISORT.f) into a Python-importable module. Unlike the simple examples though, the subroutine needed other subroutines/functions found in separate files. 

The solution's pseudocode looked like this:
python -m numpy.f2py -c <my module.f and all dependent modules.f> -m <new module name> 

For the case of DISORT, it was the following:
python -m numpy.f2py -c BDREF.f DISOBRDF.f DISORT.f ERRPACK.f LAPACK.f LINPAK.f RDI1MACH.f -m disort

This creates a shared object file which is rather smart regarding what inputs and outputs are necessary. But more on that later