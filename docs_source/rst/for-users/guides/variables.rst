Variables
=========

This page describes what I think the variables used by DISORT mean, and a bit
about how I think they're used. I emphasize *think* because DISORT comes with
documentation for v2 but no documentation afterwards. Some variables have no
documentation, and others may have changed during later releases.

Required variables
------------------
* USRANG:
* USRTAU:
* IBCND:
* ONLYFL:
* PRNT:
* PLANK:
* LAMBER:
* DELTAMPLUS:
* DO_PSEUDO_SPHERE:
* DTAUC:
* SSALB:
* PMOM:
* TEMPER:
* WVNMLO:
* WVNMHI:
* UTAU:
* UMU0:
* PHI0:
* UMU:
* PHI:
* FBEAM:
* FISOT:
* ALBEDO:
* BTEMP:
* TTEMP:
* TEMIS:
* EARTH_RADIUS:
* H_LYR:
* RHOQ:
* RHOU:
* RHO_ACCURATE:
* BEMST:
* EMUST:
* ACCUR:
* HEADER:
* RFLDIR:
* RFLDN:
* FLUP:
* DFDT:
* UAVG:
* UU:
* ALBMED:
* TRNMED:

Optional variables
------------------
* MAXCLY: MAXimum Computational LaYers. This is the number of layers to use in
  the model.
* MAXMOM: MAXimum MOMents. This the number of Legendre coefficients - 1,
  presumably because the 0th moment is 1.
* MAXCMU: MAXimum Computationl MUs. This is the number of computational polar
  angles (aka, the number of "streams") to use in the model. This number should
  be even.
* MAXUMU: MAXimum User MUs. This is the number of polar angles where DISORT
  should return radiant quantities.
* MAXPHI: MAXimum PHI. This is the number of azimuthal angles where DISORT
  should return radiant quantities.
* MAXULV: MAXimum User LeVels. This is the number of user levels to use in the
  model, and is only used if you want radiant quantities returned at user
  levels.
