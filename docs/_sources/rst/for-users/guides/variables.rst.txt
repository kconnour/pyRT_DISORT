Variables
=========

This page describes what I think the variables used by DISORT mean, and a bit
about how I think they're used. I emphasize *think* because DISORT comes with
documentation for v2 but no documentation afterwards. Some variables have no
documentation, and others may have changed during later releases.

Required variables
------------------
Several of these descriptions are copied directly from the documentation;
some are descriptions I made up given what I can deduce.

* USRANG: USeR ANGles. Denote whether radiant quantities should be returned at user angles.
  If :code:`False`, radiant quantities are to be returned at computational
  polar angles. Also, :code:`UMU` will return the cosines of the computational
  polar angles and n_polar will return
  their number ( = n_streams). :code:`UMU` must
  be large enough to contain n_streams elements. If :code:`True`,
  radiant quantities are to be returned at user-specified polar
  angles, as follows: NUMU No. of polar angles (zero is a legal value
  only when 'only_fluxes' == True ) UMU(IU) IU=1 to NUMU, cosines of
  output polar angles in increasing order---starting with negative
  (downward) values (if any) and on through positive (upward)
  values; **MUST NOT HAVE ANY ZERO VALUES**.
* USRTAU: USeR TAU. Denote whether radiant quantities should be returned at user-specified
  optical depths.
* IBCND: Incidence Beam CoNDitions. Denote what functions of the incidence beam
  angle should be included. If True, return the albedo and transmissivity of the
  entire medium as a function of incidence beam angle. In this case, the following
  inputs are the only ones considered by DISORT: MAXCLY, DTAUC, SSALB, PMOM,
  NSTR, USRANG, MAXUMU, UMU, ALBEDO, PRNT, HEADER... at least according to the
  documentation. I would think all the surface phase function arrays would also
  be included.
* ONLYFL: ONLY FLuxes. Determine if only the fluxes should be returned by the
  model. If True, DISORT will return fluxes, flux divergence, and mean intensities; if false,
  return all those quantities and intensities. Additionally, if Ture, the number of polar angles
  can be 0, the number of azimuth angles can be 0, PHI is not used, and all values
  of UU wil lbe set to 0.
* PRNT: PRiNT variables. This is an array of 5 boolean values that controls
  what DISORT prints as it runs.
  1. Input variables (except ``PMOM``)
  2. Fluxes
  3. Intensities at user levels and angles
  4. Planar transmissivity and planar albedo as a function of solar zenith angle (only used if IBCND is true)
  5. PMOM for each layer (but only if flag #1 == True and only for layers with scattering)
* PLANK: True if thermal emission should be included; False otherwise. If True,
  DISORT needs these variables: BTEMP, TTEMP, TEMIS, WVNMLO, WVNMHI, TEMPER.
* LAMBER: True if the surface is a Lambert surface, False otherwise. I think
  if True, it ignores all the other surface phase function arrays.
* DELTAMPLUS: Denote whether to use the delta-M+ method of Lin et al. (2018).
* DO_PSEUDO_SPHERE: Denote whether to use a pseudo-spherical correction.
* DTAUC:
* SSALB:
* PMOM:
* TEMPER: TEMPERature of the atmosphere. I think it's only used if PLANK == True
* WVNMLO: WaVeNuMber LOw. The low wavenumber. I think it's only used if PLANK == True
* WVNMHI: WaVeNuMber HIgh. The high wavenumber. I think it's only used if PLANK == True
* UTAU:
* UMU0:
* PHI0:
* UMU:
* PHI:
* FBEAM: Flux of the BEAM. This is the beam flux at the top boundary.
* FISOT: Flux ISOTropic. This is the isotropic flux at the top boundary.
* ALBEDO: This is the Lambert albedo of the surface.
* BTEMP: Bottom TEMPerature of the atmosphere. I think it's only used if PLANK == True
* TTEMP: Top TEMPerature of the atmosphere. I think it's only used if PLANK == True
* TEMIS: Top EMISsivity of the atmosphere. I think it's only used if PLANK == True
* EARTH_RADIUS: The planetary radius. I don't know why they specified Earth.
  This is presumably only used if DO_PSEUDO_SPHERE == True, but there is no
  documentation on this variable.
* H_LYR: The scale height of each layer. This is presumably only used if
  DO_PSEUDO_SPHERE == True, but there is no documentation on this variable.
* RHOQ: Some sort of phase function array.
* RHOU: Some sort of phase function array.
* RHO_ACCURATE: Some sort of phase function array.
* BEMST: Some sort of phase function array.
* EMUST: Some sort of phase function array.
* ACCUR: ACCURacy. The convergence criterion for azimuthal (Fourier cosine) series.
  Will stop when the following occurs twice: largest term being added
  is less than ACCUR times total series sum (twice because
  there are cases where terms are anomalously small but azimuthal
  series has not converged). Should be between 0 and 0.01 to avoid
  risk of serious non-convergence. Has no effect on problems lacking a
  beam source, since azimuthal series has only one term in that case.
* HEADER: Use a 127- (or less) character header for prints, embedded in the
  DISORT banner. Input headers greater than 127 characters will be
  truncated. Setting :code:`HEADER=''` will eliminate both the banner and the
  header, and this is the only way to do so (:code:`HEADER` is not
  controlled by any of the :code:`PRNT` flags); :code:`header` can be used
  to mark the progress of a calculation in which DISORT is called many
  times, while leaving all other printing turned off.
* RFLDIR: FLux DIRect. This is an output array.
* RFLDN: diffuse FLux DowNward. This is an output array which is the total downward
  flux minus the direct beam flux.
* FLUP: diffuse FLux UPward. This is an output array.
* DFDT: flux divergence. This is an output array.
* UAVG: AVeraGe intensity. This is an output array.
* UU: intensity. This is an output array.
* ALBMED: ALBedo of the MEDium. This is an output array.
* TRNMED: TRaNsmissivity of the MEDium. This is an output array.

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
