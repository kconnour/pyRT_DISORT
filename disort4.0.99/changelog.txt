Things I changed about DISORT to modify our needs:

DISORT.f
--- (lines 388--395) I added f2py code to inform it what the output variables should be. These are interpreted by fortran as comments so
    it didn't change anything about how the fortran code works
--- (line 435) I modified RHOU's 1st dimension to be MAXCMU instead of MAXUMU. This allows us to use DISOBRDF.f to make RHOU

DISOBRDF.f
--- (lines 101--105) I added f2py code to inform it what the output variables should be. These are interpreted by fortran as comments so
    it didn't change anything about how the fortran code works
--- (line 83) I changed REAL BRDF_ARG(4) to REAL BRDF_ARG(6). This allows us to add 6 parameters to a surface phase function

BDREF.f
--- (line 60) I changed REAL BRDF_ARG(4) to REAL BRDF_ARG(6). This allows us to add 6 parameters to a surface phase function
--- (lines 71--72) I added a line at the top REAL ASYM, FRAC, ROUGHNESS to accommodate these parameters
--- (lines 157--169) I added ELSEIF (IREF.eq.5) to handle a Hapke HG2 phase function
--- (lines 170--183) I added ELSEIF (IREF.eq.6) to handle a Hapke HG2 + surface roughness phase function
--- (lines 505 onward) Copy disort_multi code to add the 2 Hapke surfaces
