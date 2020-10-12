c ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c $Rev: 55 $ $Date: 2014-12-31 12:16:59 -0500 (Wed, 31 Dec 2014) $
c FORTRAN 77
c ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      REAL FUNCTION BDREF(MU, MUP, DPHI,
     &                     BRDF_TYPE, BRDF_ARG)

c     Supplies surface bi-directional reflectivity.
c
c     NOTE 1: Bidirectional reflectivity in DISORT is defined
c             by Eq. 39 in STWL.
c     NOTE 2: Both MU and MU0 (cosines of reflection and incidence
c             angles) are positive.
c
c  INPUT:
c
c    MU     : Cosine of angle of reflection (positive)
c
c    MUP    : Cosine of angle of incidence (positive)
c
c    DPHI   : Difference of azimuth angles of incidence and reflection
c                (radians)
c
c  LOCAL VARIABLES:
c
c    IREF   : bidirectional reflectance options
c             1 - Hapke's BDR model
c             2 - Cox-Munk BDR model
c             3 - RPV BDR model
c             4 - Ross-Li BDR model
c
c    B0     : empirical factor to account for the finite size of
c             particles in Hapke's BDR model
c
c    B      : term that accounts for the opposition effect
c             (retroreflectance, hot spot) in Hapke's BDR model
c
c    CTHETA : cosine of phase angle in Hapke's BDR model
c
c    GAMMA  : albedo factor in Hapke's BDR model
c
c    H0     : H( mu0 ) in Hapke's BDR model
c
c    H      : H( mu ) in Hapke's BDR model
c
c    HH     : angular width parameter of opposition effect in Hapke's
c             BDR model
c
c    P      : scattering phase function in Hapke's BDR model
c
c    THETA  : phase angle (radians); the angle between incidence and
c             reflection directions in Hapke's BDR model
c
c    W      : single scattering albedo in Hapke's BDR model
c
c
c   Called by- DREF, SURFAC
c +-------------------------------------------------------------------+
c     .. Scalar Arguments ..
      REAL      DPHI, MU, MUP, BRDF_ARG(6)
      INTEGER   BRDF_TYPE
c     ..
c     .. Local Scalars ..
      INTEGER   IREF
      REAL      B0, H0, HH, W
      REAL      PWS, REFRAC_INDEX, BDREF_F
      REAL      PI
      REAL      RHO0, KAPPA, G  
      REAL      K_ISO, K_VOL, K_GEO, ALPHA0 
      LOGICAL   DO_SHADOW
c     Additions for pyRT_DISORT
      REAL      ASYM, FRAC, ROUGHNESS
c     ..
c     .. External Subroutines ..
      EXTERNAL  ERRMSG
c     ..
c     .. Intrinsic Functions ..
      INTRINSIC COS, SQRT
c     ..

      PI   = 2.*ASIN(1.)

      IREF = BRDF_TYPE

c     ** 1. Hapke BRDF
      IF ( IREF.EQ.1 ) THEN

c       ** Hapke's BRDF model (times Pi/Mu0) (Hapke, B., Theory of reflectance
c       ** and emittance spectroscopy, Cambridge University Press, 1993, Eq.
c       ** 8.89 on page 233. Parameters are from Fig. 8.15 on page 231, expect
c       ** for w.)

        B0 = BRDF_ARG(1) !1.0
        HH = BRDF_ARG(2) !0.06
        W  = BRDF_ARG(3) !0.6

        CALL BRDF_HAPKE(MUP, MU, DPHI,
     &                  B0, HH, W, PI,
     &                  BDREF)

c     ** 2. Cox-Munk BRDF
      ELSEIF(IREF.EQ.2) THEN

c        PRINT *, "Calling oceabrdf"

        PWS          =  BRDF_ARG(1)
        REFRAC_INDEX =  BRDF_ARG(2)

        IF(BRDF_ARG(3) .EQ. 1) THEN
          DO_SHADOW = .TRUE.
        ELSEIF(BRDF_ARG(3) .EQ. 0) THEN
          DO_SHADOW = .FALSE.
        ELSE
          PRINT *, "ERROR SHADOW ARGUMENTS"
        ENDIF

        CALL OCEABRDF2(DO_SHADOW,
     &                 REFRAC_INDEX, PWS, 
     &                 MUP, MU, DPHI,
     &                 BDREF_F)

        BDREF = BDREF_F

c     ** 3. RPV BRDF
      ELSEIF(IREF .EQ. 3) THEN

        RHO0  =  BRDF_ARG(1) !0.027
        KAPPA =  BRDF_ARG(2) !0.647
        G     =  BRDF_ARG(3) !-0.169   !asymmetry factor for HG
        H0    =  BRDF_ARG(4) !0.100

        CALL BRDF_RPV(MUP, MU, DPHI,
     &                RHO0, KAPPA, G, H0,
     &                BDREF_F)

        BDREF = BDREF_F

c     ** 4. Ross-Li BRDF
      ELSEIF(IREF .EQ. 4) THEN
        
        K_ISO  = BRDF_ARG(1)   !0.200
        K_VOL  = BRDF_ARG(2)   !0.020
        K_GEO  = BRDF_ARG(3)   !0.300
        ALPHA0 = 1.5*pi/180.

        CALL BRDF_ROSSLI(MUP, MU, DPHI,
     &                   K_ISO, K_VOL, K_GEO,
     &                   ALPHA0,
     &                   BDREF_F)

        BDREF = BDREF_F

        IF(BDREF .LT. 0.00) THEN
          BDREF = 0.00
        ENDIF

c     ** 5. Hapke + HG2 BRDF
      ELSEIF ( IREF.EQ.5 ) THEN

        B0 = BRDF_ARG(1) !1.0
        HH = BRDF_ARG(2) !0.06
        W  = BRDF_ARG(3) !0.6
        ASYM = BRDF_ARG(4)
        FRAC = BRDF_ARG(5)

        CALL BRDF_HAPKE_HG2(MUP, MU, DPHI,
     &                  B0, HH, W, ASYM, FRAC, PI,
     &                  BDREF)

c     ** 6. Hapke + HG2 with roughness
      ELSEIF ( IREF.EQ.6 ) THEN

        B0 = BRDF_ARG(1) !1.0
        HH = BRDF_ARG(2) !0.06
        W  = BRDF_ARG(3) !0.6
        ASYM = BRDF_ARG(4)
        FRAC = BRDF_ARG(5)
        ROUGHNESS = BRDF_ARG(6)

        CALL BRDF_HAPKE_HG2_ROUGHNESS(MUP, MU, DPHI,
     &                  B0, HH, W, ASYM, FRAC, ROUGHNESS, PI,
     &                  BDREF)

      ELSE

        CALL ERRMSG( 'BDREF--Need to supply surface BDRF model',
     &                 .TRUE.)

      ENDIF

      RETURN
      END FUNCTION
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c +--------------------------------------------------------------------
      SUBROUTINE BRDF_HAPKE ( MUP, MU, DPHI,
     &                        B0, HH, W, PI,
     &                        BRDF )

c +--------------------------------------------------------------------
c Hapke "Theory of Reflectance and Emittance Spectroscopy" Chapter 10, Page 262
c Eq. (10.2).
c Version 3 fix: definition of phase angle / scattering angle see DISORT3
c paper Eqs. (25-26).
c +--------------------------------------------------------------------
      IMPLICIT NONE
      REAL MUP, MU, DPHI
      REAL B0, HH, W, PI
      REAL BRDF
      REAL CALPHA, ALPHA, P, B, H0, GAMMA, H

      CALPHA = MU * MUP - (1.-MU**2)**.5 * (1.-MUP**2)**.5
     &         * COS( DPHI )

      ALPHA = ACOS( CALPHA )

      P     = 1. + 0.5 * CALPHA

      B     = B0 * HH / ( HH + TAN( ALPHA/2.) )

      GAMMA = SQRT( 1. - W )
      H0   = ( 1. + 2.*MUP ) / ( 1. + 2.*MUP * GAMMA )
      H    = ( 1. + 2.*MU ) / ( 1. + 2.*MU * GAMMA )

c     ** Version 3: add factor PI
      BRDF = W / (4.*PI) / (MU+MUP) * ( (1.+B)* P + H0 * H - 1.0 )
c     BRDF = W / 4. / (MU+MUP) * ( (1.+B)* P + H0 * H - 1.0 )

      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c +--------------------------------------------------------------------
      SUBROUTINE BRDF_RPV(MU_I, MU_R, DPHI,
     &                    RHO0, KAPPA, G_HG, H0,
     &                    BRDF)

c +--------------------------------------------------------------------
c DISORT Version 3: RPV BRDF
c   Input:
c
c   MU_I:  absolute cosine of incident polar angle (positive)
c   MU_R:  absolute cosine of reflected polar angle (positive)
c   DPHI:  relative azimuth to incident vector; (pi - dphi), sun-view relative
c          azimuth sun located at phi = 180, while incident solar beam located
c          at phi = 0
c   RHO0:  RPV BRDF parameter, control reflectance
c   KAPPA: PRV BRDF parameter, control anisotropy
c   G:     RPV BRDF parameter, H-G asymmetry factor
c   H0:    RPV BRDF parameter, control hot spot (back scattering direction)
c
c   Output:
c
c   BRDF:  RPV BRDF
c +--------------------------------------------------------------------
      IMPLICIT NONE
      REAL MU_I, MU_R, DPHI
      REAL RHO0, KAPPA, G_HG, H0
      REAL BRDF
      REAL PI
      REAL COS_ALPHA
      REAL SIN_I, SIN_R, TAN_I, TAN_R
      REAL G_SQ, G, F

      PI    = 2.*ASIN(1.)

      SIN_I = SQRT(1. - MU_I*MU_I)
      SIN_R = SQRT(1. - MU_R*MU_R)
      TAN_I = SIN_I/MU_I
      TAN_R = SIN_R/MU_R

      COS_ALPHA = MU_I*MU_R - SIN_I*SIN_R
     & *COS(DPHI)

      G_SQ = TAN_I*TAN_I + TAN_R*TAN_R 
     &    + 2.*TAN_I*TAN_R*COS(DPHI)

c     ** hot spot
      G = SQRT(G_SQ)

c     ** HG phase function
      F = (1. - G_HG*G_HG)/
     &     (1+G_HG*G_HG+2.*G_HG*COS_ALPHA)**1.5


c     ** BRDF semiempirical function
      BRDF = RHO0 
     &      * (MU_I*MU_R*(MU_I+MU_R))**(KAPPA-1.)
     &      * F
     &      * (1. + ((1.-H0)/(1.+G)))

      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c +--------------------------------------------------------------------
      SUBROUTINE BRDF_ROSSLI(MU_I, MU_R, DPHI,
     &                       K_ISO, K_VOL, K_GEO,
     &                       ALPHA0,
     &                       BRDF)

c +--------------------------------------------------------------------
c Version 3: Ross-Li BRDF
c   Input:
c
c   MU_I:    absolute cosine of incident polar angle (positive)
c   MU_R:    absolute cosine of reflected polar angle (positive)
c   DPHI:  relative azimuth to incident vector; (pi - dphi), sun-view relative
c          azimuth sun located at phi = 180, while incident solar beam located
c          at phi = 0
c   K_ISO:   BRDF parameter, isotropic scattering kernel
c   K_VOL:   BRDF parameter, volume scattering kernel
c   K_GEO:   BRDF parameter, geometry scattering kernel
c   ALPHA0:  BRDF parameter, control hot spot (back scattering direction)
c
c   Output:
c   BRDF:  Ross-Li BRDF
c
c +--------------------------------------------------------------------
      IMPLICIT NONE
      REAL MU_I, MU_R, DPHI
      REAL F_GEO, F_VOL
      REAL K_ISO, K_GEO, K_VOL
      REAL RATIO_HB, RATIO_BR
      REAL BRDF
      REAL PI
      REAL COS_ALPHA, SIN_ALPHA
      REAL COS_ALPHA1
      REAL ALPHA
      REAL SIN_I, SIN_R, TAN_I, TAN_R
      REAL SIN_I1, SIN_R1, COS_I1, COS_R1, TAN_I1, TAN_R1
      REAL G_SQ, COS_T, T       
      REAL C, ALPHA0
c +--------------------------------------------------------------------

c      PRINT *, MU_I, MU_R, DPHI,
c     &        K_ISO, K_GEO, K_VOL,
c     &        THETA0
c      PRINT *,

      RATIO_HB = 2.
      RATIO_BR = 1.
      PI       = 2.*ASIN(1.)

      SIN_I = SQRT(1. - MU_I*MU_I)
      SIN_R = SQRT(1. - MU_R*MU_R)
      TAN_I = SIN_I/MU_I
      TAN_R = SIN_R/MU_R

      COS_ALPHA = MU_I*MU_R - SIN_I*SIN_R
     & *COS(DPHI)
      SIN_ALPHA = SQRT(1. - COS_ALPHA*COS_ALPHA)
      ALPHA = ACOS(COS_ALPHA)

c     ** Compute KERNEL RossThick
      C     = 1. + 1./(1.+ALPHA/ALPHA0)
      F_VOL = 4./(3.*PI) * (1./(MU_I+MU_R))
     &       * ((PI/2. - ALPHA)*COS_ALPHA+SIN_ALPHA)*C - 1./3.

c      K1 = ((PI/2. - ALPHA)*COS_ALPHA + SIN_ALPHA)
c     &       /(MU_I + MU_R) - PI/4.


c     ** Compute KERNEL LSR
      TAN_I1 = RATIO_BR * TAN_I
      TAN_R1 = RATIO_BR * TAN_R
      SIN_I1 = TAN_I1/SQRT(1.+ TAN_I1*TAN_I1)
      SIN_R1 = TAN_R1/SQRT(1.+ TAN_R1*TAN_R1)
      COS_I1 = 1./SQRT(1.+ TAN_I1*TAN_I1)
      COS_R1 = 1./SQRT(1.+ TAN_R1*TAN_R1)

      COS_ALPHA1 = COS_I1*COS_R1 - SIN_I1*SIN_R1
     &            *COS(DPHI)

      G_SQ = TAN_I1*TAN_I1 + TAN_R1*TAN_R1 
     &      + 2.*TAN_I1*TAN_R1*COS(DPHI)

c      M = 1./COS_I1 + 1./COS_R1

      COS_T = RATIO_HB *(COS_I1*COS_R1)/(COS_I1+COS_R1)
     &       *SQRT(G_SQ + (TAN_I1*TAN_R1*SIN(DPHI))**2)
  
      IF(COS_T .LE. 1. .AND. COS_T .GE. -1.) THEN
        T = ACOS(COS_T)
      ELSE
        T = 0.
      ENDIF

      F_GEO = (COS_I1+COS_R1)/(PI*COS_I1*COS_R1)*(T-SIN(T)*COS(T)-PI)   
     &       + (1.+ COS_ALPHA1)/(2.*COS_I1*COS_R1)

c     Compute BRDF

c      PRINT *, RATIO_HB, D_SQ, 
c     &    TAN_I1*TAN_R1*SIN(DPHI),
c     &    M, COS_T

c      BRDF = K1
      BRDF = K_ISO + K_GEO*F_GEO + K_VOL*F_VOL

      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c +--------------------------------------------------------------------
      SUBROUTINE OCEABRDF2
     &       ( DO_SHADOW, 
     &         REFRAC_INDEX, WS,
     &         MU_I, MU_R, DPHI,
     &         BRDF)

c +--------------------------------------------------------------------
c Version 3: 1D Gaussian Rough Ocean BRDF
c   Input:
c
c   mu_i:         absolute cosine of incident polar angle (positive)
c   mu_r:         absolute cosine of reflected polar angle (positive)
c   dphi:         relative azimuth (radians) 
c   do_shadow:    BRDF parameter, open/close shadow effect 
c   refrac_index: BRDF parameter, refractive index of boundary media (water)
c   ws:           BRDF parameter, wind speed (m/s)
c
c   Output:
c
c   brdf:         1D Gaussian Rough Ocean BRDF
c          
c +--------------------------------------------------------------------
      LOGICAL  DO_SHADOW
      REAL     REFRAC_INDEX, WS
      REAL     SIN_I, SIN_R, MU_I, MU_R, DPHI, BRDF
      REAL     COS_THETA, SIGMA_SQ, MU_N_SQ, P
      REAL     N_I, N_T, COS_LI, COS_LT, SIN_LI, SIN_LT
      REAL     R_S, R_P, R
      REAL     SHADOW
      REAL     PI

      PI = 2.*ASIN(1.)

c     ** Cox Munk slope distribution
      SIN_I = SQRT(1. - MU_I*MU_I)
      SIN_R = SQRT(1. - MU_R*MU_R)

      COS_THETA = -MU_I*MU_R + SIN_I*SIN_R*COS(DPHI)
      MU_N_SQ   = (MU_I + MU_R)*(MU_I + MU_R)/(2.*(1.-COS_THETA))   

      SIGMA_SQ  = 0.003 + 0.00512*WS

      P = 1./(PI*SIGMA_SQ) * EXP( -(1-MU_N_SQ)/(SIGMA_SQ*MU_N_SQ) )

c     ** Fresnel reflectance

      N_I = 1.0
      N_T = REFRAC_INDEX

      SIN_LI = SQRT( 1.-0.5*(1.-COS_THETA) ) 
      COS_LI = SQRT( 0.5*(1.-COS_THETA) ) 
      SIN_LT = N_I*SIN_LI/N_T
      COS_LT = SQRT(1. - SIN_LT*SIN_LT)

      R_S = (N_I*COS_LI-N_T*COS_LT)/(N_I*COS_LI+N_T*COS_LT)
      R_P = (N_T*COS_LI-N_I*COS_LT)/(N_I*COS_LT+N_T*COS_LI)

      R = 0.5*(R_S*R_S + R_P*R_P)

c     ** Rough surface BRDF
      BRDF = (P*R)/(4.*MU_I*MU_R*MU_N_SQ*MU_N_SQ)

c     Shadowing effect (see Tsang, Kong, Shin, Theory of Microwave Remote
c     Sensing, Wiley-Interscience, 1985) 
      IF(DO_SHADOW) THEN
        SHADOW = 1./( SHADOW_ETA(MU_I, SIGMA_SQ, PI) 
     &          + SHADOW_ETA(MU_R, SIGMA_SQ, PI) + 1. )
        BRDF = BRDF*SHADOW
      ENDIF

      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c +--------------------------------------------------------------------
      REAL FUNCTION SHADOW_ETA(COS_THETA, SIGMA_SQ, PI)
c +--------------------------------------------------------------------
c Version 3: shadow effect function
c            called by OCEABRDF2
c   Input:
c
c   COS_THETA     absolute cosine of incident/reflected polar angle (positive)
c   SIGMA_SQ      slope variance 
c   PI            3.141592653... constant
c
c   Output:
c
c   SHADOW_ETA:   shadow function
c +--------------------------------------------------------------------
      REAL COS_THETA, SIN_THETA
      REAL MU, SIGMA_SQ, PI
      REAL TERM1, TERM2

      SIN_THETA = SQRT(1.-COS_THETA*COS_THETA)
      MU = COS_THETA/SIN_THETA

      TERM1 = SQRT(SIGMA_SQ/PI)/MU*EXP( -MU*MU/(SIGMA_SQ) )
      TERM2 = ERFC( MU/SQRT(SIGMA_SQ) )

      SHADOW_ETA = 0.5*(TERM1 - TERM2)

      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c Start pyRT_DISORT additions
c Hapke + HG2 surface
c +--------------------------------------------------------------------
      SUBROUTINE BRDF_HAPKE_HG2 ( MUP, MU, DPHI,
     &                        B0, HH, W, ASYM, FRAC, PI,
     &                        BRDF )

c +--------------------------------------------------------------------

c +--------------------------------------------------------------------
      IMPLICIT NONE
      REAL MUP, MU, DPHI
      REAL B0, HH, W, PI
      REAL BRDF
      REAL CALPHA, ALPHA, P, B, H0, GAMMA, H
      REAL ASYM, FRAC
      REAL CTHETA, THETA, FORWARD, BACKWARD, PSI

      CALPHA = MU * MUP - (1.-MU**2)**.5 * (1.-MUP**2)**.5
     &         * COS( DPHI )

      ALPHA = ACOS( CALPHA )

      B     = B0 * HH / ( HH + TAN( ALPHA/2.) )

      GAMMA = SQRT( 1. - W )
      H0   = ( 1. + 2.*MUP ) / ( 1. + 2.*MUP * GAMMA )
      H    = ( 1. + 2.*MU ) / ( 1. + 2.*MU * GAMMA )
      

      FORWARD = (1. - ASYM**2.) / (1. + 2. * ASYM * COS(ALPHA) + 
     &           ASYM**2.)**1.5
      BACKWARD = (1. - ASYM**2.) / (1. - 2. * ASYM * COS(ALPHA) + 
     &           ASYM**2.)**1.5
      
      P = FRAC * FORWARD + (1-FRAC)*BACKWARD
      
c     ** Version 3: add factor PI
      BRDF = W / (4.*PI) / (MU+MUP) * ( (1.+B)* P + H0 * H - 1.0 )
      
c      BRDF = W / 4. / (MU+MUP) * ( (1.+B)* P + H0 * H - 1.0 )

      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c Just copy Mike's code/logic here
c +--------------------------------------------------------------------
      SUBROUTINE BRDF_HAPKE_HG2_ROUGHNESS ( MUP, MU, DPHI,
     &                        B0, HH, W, ASYM, FRAC, ROUGHNESS, PI,
     &                        BRDF )

c +--------------------------------------------------------------------

c +--------------------------------------------------------------------
      IMPLICIT NONE
      REAL MUP, MU, DPHI
      REAL B0, HH, W, PI
      REAL BRDF, P, B
      REAL ASYM, FRAC, ROUGHNESS
      REAL CALPHA, ALPHA, FORWARD, BACKWARD, i, e, hapke_emue
      REAL H_function
      REAL hapke_imue, imue, emue, H_imue, H_emue, S, S_function
      LOGICAL flag

      CALPHA = MU * MUP - (1.-MU**2)**.5 * (1.-MUP**2)**.5
     &         * COS( DPHI )

      ALPHA = ACOS( CALPHA )      
      
      FORWARD = (1. - ASYM**2.) / (1. + 2. * ASYM * COS(ALPHA) + 
     &           ASYM**2.)**1.5
      BACKWARD = (1. - ASYM**2.) / (1. - 2. * ASYM * COS(ALPHA) + 
     &           ASYM**2.)**1.5
      P = FRAC * FORWARD + (1-FRAC)*BACKWARD
      B     = B0 * HH / ( HH + TAN( ALPHA/2.) )    
      
      flag = .false.
      i = ACOS(MU)
      e = ACOS(MUP)
      imue = Hapke_imue(i, e, ALPHA, ROUGHNESS)
      emue = Hapke_emue(i, e, ALPHA, ROUGHNESS)
      H_imue = H_function(imue, w, flag)
      H_emue = H_function(emue, w, flag)
      S = S_function(i, e, ALPHA, ROUGHNESS)

      BRDF = W / (4.*PI * MU) * (imue / (imue + emue)) *
     &         ((1.0 + B) * p + H_imue * H_emue - 1.0) * S
      
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c subroutines written by Frank Seelos, ported from f90 to f77 by M.
c Wolff (that fiend!)
c
c history:
c 2005/07/27 (mjw):  changed "END FUNCTION" syntax to just "END"
c


      REAL FUNCTION H_function(x, w, H_approx)

cFUNCTION: 
c	H_function
c
cCALLED BY:
c	HapkeBDREF
c
cCALLS:
c	Hapke_gamma
c	Hapke_r0 (conditionally)
c
cINPUT PARAMETERS:
c	x : the cosine of either the incidence or emission angle depending on the calling 
c		circumstance
c	w : single scattering albedo
c
cPURPOSE:
c	H_function is an approximation to Chandreskehar H-function which is fundamental to the 
c	calculation of the bidirectional reflectance of a semiinfinite medium of isotropic scatterers.
c
cREFERENCE:
c	Hapke (1993); Eqn. 8.55; p. 212


      IMPLICIT NONE
      REAL Hapke_gamma
      REAL Hapke_r0
      LOGICAL H_approx

      REAL x, w
      REAL gamma, r0

      gamma = Hapke_gamma(w)	


      if (H_approx .EQV. .FALSE.) then
         H_function = (1.0 + 2.0 * x) / (1.0 + 2.0 * x * gamma)
      else
         r0 = Hapke_r0(gamma)
         H_function  = 1.0 / (1.0 - (1.0 - gamma) * x * 
     &     (r0 + (1.0 - 0.5 * r0 - r0 * x) * alog((1.0 + x) / x)))

      end if

      RETURN
      END

c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c


      REAL FUNCTION B_function(g, h, B_approx)

cFUNCTION: 
c	B_function
c
cCALLED BY:
c	HapkeBDREF
c
cCALLS:
c	NONE
c
cINPUT PARAMETERS:
c	g : phase angle
c	h : compaction parameter
c
cPURPOSE:
c	The opposition effect function (B_function) calculates the effect of shadow hiding on the
c	bidirectional reflectance function
c
cREFERENCE:
c	Hapke (1993); Eqn. 8.81; p. 224


      IMPLICIT NONE

      REAL  pi

      REAL  Hapke_y, ERF
      REAL  y
	


      REAL  g, h
      LOGICAL  B_approx

      pi = 2.0 * asin(1.0)

c	write (*,*), 'B_approx_flag: ', B_approx_flag

      if (B_approx .EQV. .FALSE.) then 
         B_function = 1.0 / (1.0 + 1.0 / h * tan(g / 2.0))
      else
         y = Hapke_y(g, h)
         B_function = sqrt((4.0 * pi) / y) * exp(1.0 / y) *
     &        (ERF(sqrt(4.0 / y)) - ERF(sqrt(1.0 / y))) +
     &        exp(-3.0 / y) - 1.0
      endif

      
      RETURN
      END

c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c


      REAL FUNCTION S_function(i, e, psi, theta_bar)

cFUNCTION: 
c	S_function
c
cCALLED BY:
c	HapkeBDREF
c
cCALLS:
c	Hapke_mue0
c	Hapke_imue
c	Hapke_emue
c	Hapke_chi
c	Hapke_fpsi
c
cINPUT PARAMETERS:
c	i : incidence angle
c	e : emission angle
c	psi : difference in azimuth angle between incident and emergent rays
c	theta_bar : macroscopic roughness parameter (mean slope angle)
c
cPURPOSE:
c	The shadowing function (S_function) calculates the effect of macroscopic 
c	roughness on the bidirectional reflectance function
c
cREFERENCE:
c	Hapke (1993); Eqn. 12.50 & 12.54; p. 345


      IMPLICIT NONE

      REAL Hapke_mue0, Hapke_imue, Hapke_emue, Hapke_chi, Hapke_fpsi
      REAL  i, e, psi, theta_bar
      REAL imu, emu, imue0, emue0, imue, emue, S
      
      imu = cos(i)
      emu = cos(e)
      
      imue0 = Hapke_mue0(i, theta_bar)
      emue0 = Hapke_mue0(e, theta_bar)
      
      imue = Hapke_imue(i, e, psi, theta_bar)
      emue = Hapke_emue(i, e, psi, theta_bar)
      
      if (i .le. e) then 
         S = (emue / emue0) * (imu / imue0) * 
     &        (Hapke_chi(theta_bar) / (1.0 - Hapke_fpsi(psi) + 
     &        Hapke_fpsi(psi) * Hapke_chi(theta_bar) * (imu / imue0)))

      else
         S = (emue / emue0) * (imu / imue0) * 
     &        (Hapke_chi(theta_bar) / (1.0 - Hapke_fpsi(psi) + 
     &        Hapke_fpsi(psi) * Hapke_chi(theta_bar) * (emu / emue0)))

      endif

      S_function = S

      RETURN

      END

c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c


      REAL FUNCTION Hapke_imue(i, e, psi, theta_bar)

cFUNCTION: 
c	Hapke_imue
c
cCALLED BY:
c	S_function
c
cCALLS:
c	Hapke_chi
c	Hapke_E1
c	Hapke_E2
c
cINPUT PARAMETERS:
c	i : incidence angle
c	e : emission angle
c	psi : difference in azimuth angle between incident and emergent rays
c	theta_bar : macroscopic roughness parameter (mean slope angle)
c
cPURPOSE:
c	Hapke_imue is the cosine of the effective incidence angle when theta_bar NE 0.0
c
cREFERENCE:
c	Hapke (1993); Eqn. 12.46 & 12.52; p. 344 & 345


      IMPLICIT NONE

      REAL  pi
      REAL Hapke_chi, Hapke_E1, Hapke_E2
      REAL  i, e, psi, theta_bar
      REAL imue

      pi = 2.0 * asin(1.0)


      if (i .le. e) then 

c		write (*,*), 'IMUE'
c		write (*,*), i, e, psi, theta_bar
c		write (*,*), Hapke_chi(theta_bar)
c		write (*,*), Hapke_E1(e, theta_bar), Hapke_E1(i, theta_bar)
c		write (*,*), Hapke_E2(e, theta_bar), Hapke_E2(i, theta_bar) 
c		write (*,*)


         imue = Hapke_chi(theta_bar) * (cos(i) + sin(i) * 
     &        tan(theta_bar) *	((cos(psi) * Hapke_E2(e, theta_bar) 
     &        + (sin(psi/2.0))**2.0 * Hapke_E2(i, theta_bar)) /
     &        (2.0 - Hapke_E1(e, theta_bar) - (psi/pi) * 
     &        Hapke_E1(i, theta_bar))))
      else

         imue = Hapke_chi(theta_bar) * (cos(i) + sin(i) * 
     &        tan(theta_bar) * ((Hapke_E2(i, theta_bar) -
     &        (sin(psi/2.0))**2.0 * Hapke_E2(e, theta_bar)) /
     &        (2.0 - Hapke_E1(i, theta_bar) - (psi/pi) *
     &        Hapke_E1(e, theta_bar))))
      endif


      Hapke_imue = imue

      RETURN

      END

c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c


      REAL FUNCTION Hapke_emue(i, e, psi, theta_bar)

cFUNCTION: 
c	Hapke_emue
c
cCALLED BY:
c	S_function
c
cCALLS:
c	Hapke_chi
c	Hapke_E1
c	Hapke_E2
c
cINPUT PARAMETERS:
c	i : incidence angle
c	e : emission angle
c	psi : difference in azimuth angle between incident and emergent rays
c	theta_bar : macroscopic roughness parameter (mean slope angle)
c
cPURPOSE:
c	Hapke_emue is the cosine of the effective emission angle when theta_bar NE 0.0
c
cREFERENCE:
c	Hapke (1993); Eqn. 12.47 & 12.53; p. 344 & 345


      IMPLICIT NONE

      REAL pi
      REAL Hapke_chi, Hapke_E1, Hapke_E2
      REAL  i, e, psi, theta_bar
      REAL  emue

      pi = 2.0 * asin(1.0)

      if (i .le. e) then 
         emue = Hapke_chi(theta_bar) * (cos(e) + sin(e) *
     &        tan(theta_bar) * ((Hapke_E2(e, theta_bar) -
     &        (sin(psi/2.0))**2.0 * Hapke_E2(i, theta_bar)) /
     &        (2.0 - Hapke_E1(e, theta_bar) - (psi/pi) *
     &        Hapke_E1(i, theta_bar))))
      else
         emue = Hapke_chi(theta_bar) * (cos(e) + sin(e) *
     &        tan(theta_bar) * ((cos(psi) * Hapke_E2(i, theta_bar) +
     &        (sin(psi/2.0))**2.0 * Hapke_E2(e, theta_bar)) /
     &        (2.0 - Hapke_E1(i, theta_bar) - (psi/pi) *
     &        Hapke_E1(e, theta_bar))))

      endif

      Hapke_emue = emue

      RETURN
      END

c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c


      REAL FUNCTION Hapke_mue0(theta, theta_bar)

cFUNCTION: 
c	Hapke_mue0
c
cCALLED BY:
c	S_function
c
cCALLS:
c	Hapke_chi
c	Hapke_E1
c	Hapke_E2
c
cINPUT PARAMETERS:
c	theta : incidence or emission angle (i or e) depending on calling circumstance
c	theta_bar : macroscopic roughness parameter (mean slope angle)
c
cPURPOSE:
c	Hapke_mue0 is the cosine of the effective incidence or emission angle when psi EQ 0.0
c
cREFERENCE:
c	Hapke (1993); Eqn. 12.48 & 12.49; p. 344


      IMPLICIT NONE
      REAL Hapke_chi, Hapke_E1, Hapke_E2
      REAL  theta, theta_bar
	
      Hapke_mue0 = Hapke_chi(theta_bar) * (cos(theta) + sin(theta)
     &     * tan(theta_bar) * Hapke_E2(theta, theta_bar) /
     &     (2.0 - Hapke_E1(theta, theta_bar)))

      RETURN
      END

c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c


      REAL FUNCTION Hapke_y(g, h)

      IMPLICIT NONE

      REAL  g, h
	
      Hapke_y = tan(g/2.0) / h
      RETURN

      END

c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c


      REAL FUNCTION Hapke_r0(gamma)

cFUNCTION: 
c	Hapke_r0
c
cCALLED BY:
c	H_function (conditionally)
c
cCALLS:
c	NONE
c
cINPUT PARAMETERS:
c	gamma : albedo factor
c
cPURPOSE:
c	Hapke_r0 (diffusive reflectance) is used in the more exact approximation to the 
c	Chandreskehar H-function	
c
cREFERENCE:
c	Hapke (1993); Eqn. 8.25; p. 196


      IMPLICIT NONE

      REAL  gamma
	
      Hapke_r0 = (1.0 - gamma) / (1.0 + gamma)

      RETURN

      END

c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c


      REAL FUNCTION Hapke_gamma(w)

cFUNCTION: 
c	Hapke_gamma
c
cCALLED BY:
c	H_function
c	B_function
c
cCALLS:
c	NONE
c
cINPUT PARAMETERS:
c	w : single scattering albedo
c
cPURPOSE:
c	Hapke_gamma (albedo factor) is a component in the calculation of the Chandrasekhar
c	H-functions and the contribution of the opposition effect
c
cREFERENCE:
c	Hapke (1993); Eqn. 8.22b; p. 195


      IMPLICIT NONE

      REAL  w
		
      Hapke_gamma = sqrt(1.0 - w)

      RETURN

      END

c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c


      REAL FUNCTION Hapke_chi(theta_bar)

cFUNCTION: 
c	Hapke_chi
c
cCALLED BY:
c	S_function
c	Hapke_imue
c	Hapke_emue
c	Hapke_mue0
c
cCALLS:
c	NONE
c
cINPUT PARAMETERS:
c	theta_bar : macroscopic roughness parameter (mean slope angle)
c
cPURPOSE:
c	Hapke_chi is a component in the calculation of the shadowing function and the 
c	cosines of the effective incidence and emission angles when theta_bar NE 0.0
c
cREFERENCE:
c	Hapke (1993); Eqn. 12.45a; p. 344


      IMPLICIT NONE
      
      REAL  pi 
        
      REAL  theta_bar
	
      pi = 2.0 * asin(1.0)


      Hapke_chi = 1.0 / sqrt(1.0 + pi * tan(theta_bar)**2.0)
      
      RETURN

      END

c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c


      REAL FUNCTION Hapke_fpsi(psi)

cFUNCTION: 
c	Hapke_fpsi
c
cCALLED BY:
c	S_function
c
cCALLS:
c	NONE
c
cINPUT PARAMETERS:
c	psi : difference in azimuth angle between incident and emergent rays
c
cPURPOSE:
c	Hapke_fpsi is a component in the calculation of the shadowing function when theta_bar NE 0.0
c
cREFERENCE:
c	Hapke (1993); Eqn. 12.51; p. 345


      IMPLICIT NONE
      REAL pi
      REAL  psi
	
      pi = 2.0 * asin(1.0)


      if (psi .eq. pi) then 
         Hapke_fpsi = 0.0
      else 
         Hapke_fpsi = exp(-2.0 * tan(psi / 2.0))
      endif

      RETURN

      END

c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c


      REAL FUNCTION Hapke_E1(x, theta_bar)

cFUNCTION: 
c	Hapke_E1
c
cCALLED BY:
c	Hapke_imue
c	Hapke_emue
c	Hapke_mue0
c
cCALLS:
c	NONE
c
cINPUT PARAMETERS:
c	x : incidence or emission angle (i or e) depending on input geometry
c	theta_bar : macroscopic roughness parameter (mean slope angle)
c
cPURPOSE:
c	Hapke_E1 is a component in the calculation of the cosines of the effective incidence
c	and emission angles when theta_bar NE 0.0
c
cREFERENCE:
c	Hapke (1993); Eqn. 12.45b; p. 344


      IMPLICIT NONE
      REAL  pi
      REAL  x, theta_bar

      pi = 2.0 * asin(1.0)

      if (x .eq. 0.0) then
         Hapke_E1 = 0.0
      else 
         Hapke_E1 = exp(-2.0 / pi * 1.0/tan(theta_bar) * 1.0/tan(x))
      endif

      RETURN
      END

c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c


      REAL FUNCTION Hapke_E2(x, theta_bar)

cFUNCTION: 
c	Hapke_E2
c
cCALLED BY:
c	Hapke_imue
c	Hapke_emue
c	Hapke_mue0
c
cCALLS:
c	NONE
c
cINPUT PARAMETERS:
c	x : incidence or emission angle (i or e) depending on input geometry
c	theta_bar : macroscopic roughness parameter (mean slope angle)
c
cPURPOSE:
c	Hapke_E2 is a component in the calculation of the cosines of the effective incidence
c	and emission angles when theta_bar NE 0.0
c
cREFERENCE:
c	Hapke (1993); Eqn. 12.45c; p. 344


      IMPLICIT NONE
      REAL  pi

      REAL x, theta_bar
      pi = 2.0 * asin(1.0)

      if (x .eq. 0.0) then 
         Hapke_E2 = 0.0
      else
         Hapke_E2 = exp(-1.0 / pi * (1.0/tan(theta_bar))**2.0 *
     &        (1.0/tan(x))**2.0)
      endif

      RETURN
      END

c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c 

      REAL FUNCTION phf_isotropic()
c
cFUNCTION: 
c	phf_isotropic 
c
cCALLED BY:
c	HapkeBDREFPhaseFunctions
c
cCALLS:
c	NONE
c
cINPUT PARAMETERS:
c	NONE
c
cPURPOSE:
c	Isotropic phase function
c
cREFERENCE:

      IMPLICIT NONE

      phf_isotropic = 1.0

      RETURN
      END

c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c


      REAL FUNCTION phf_aniso_neg(g)
c
cFUNCTION: 
c	phf_aniso_neg
c
cCALLED BY:
c	HapkeBDREFPhaseFunctions
c
cCALLS:
c	NONE
c
cINPUT PARAMETERS:
c	g : phase angle
c
cPURPOSE:
c	Calculate anisotropic phase function [1 - cos(g)]
c
cREFERENCE:
c	Hapke (1993); p. 214

      IMPLICIT NONE
      
      REAL g

      phf_aniso_neg = 1.0 - cos(g)

      RETURN

      END

c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c


      REAL FUNCTION phf_aniso_pos(g)
c
cFUNCTION: 
c	phf_aniso_pos
c
cCALLED BY:
c	HapkeBDREFPhaseFunctions
c
cCALLS:
c	NONE
c
cINPUT PARAMETERS:
c	g : phase angle
c
cPURPOSE:
c	Calculate anisotropic phase function [1 + cos(g)]
c
cREFERENCE:
c	Hapke (1993); p. 214

      IMPLICIT NONE

      REAL g

      phf_aniso_pos = 1.0 + cos(g)

      RETURN

      END
      
c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c

      REAL FUNCTION phf_rayleigh(g)
c
cFUNCTION: 
c	phf_rayleigh
c
cCALLED BY:
c	HapkeBDREFPhaseFunctions
c
cCALLS:
c	NONE
c
cINPUT PARAMETERS:
c	g : phase angle
c
cPURPOSE:
c	Calculate Rayleigh phase function
c
cREFERENCE:
c	

      IMPLICIT NONE

      REAL g

      phf_rayleigh = (3.0/4.0) * (1.0 + (cos(g))**2.0)

      RETURN

      END

c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c


      REAL FUNCTION phf_hg1(g, a)
c
cFUNCTION: 
c	phf_hg1
c
cCALLED BY:
c	HapkeBDREFPhaseFunctions
c
cCALLS:
c	NONE
c
cINPUT PARAMETERS:
c	g : phase angle (radians)
c	a : asymmetry parameter [-1,1];  
c		Negative is back; Positive is forward.
c
cPURPOSE:
c	Calculate one parameter Henyey-Greenstein phase function
c
cREFERENCE:
c

      IMPLICIT NONE

      REAL g, a

      phf_hg1 = (1.0 - a**2.0) / ((1.0 + 2.0 * a * cos(g) 
     $     + a**2.0)**1.5)

      RETURN

      END

c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c


      REAL FUNCTION phf_hg2(g, a, f)
c
cFUNCTION: 
c	phf_hg2
c
cCALLED BY:
c	HapkeBDREFPhaseFunctions
c
cCALLS:
c	NONE
c
cINPUT PARAMETERS:
c	g : phase angle (radians)
c	a : asymmetry parameter [0,1]
c	f : forward fraction [0,1]
c
cPURPOSE:
c	Calculate two parameter Henyey-Greenstein phase function
c
cREFERENCE:
c

      IMPLICIT NONE

      REAL g, a, f
      REAL forward, backward
      REAL phf_hg1

      forward = phf_hg1(g, a)
      backward = phf_hg1(g, -1.0 * a)

      phf_hg2 = f * forward + (1.0 - f) * backward

      RETURN
      END

c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c


      REAL FUNCTION phf_hg3(g, af, ab, f)
c
cFUNCTION: 
c	phf_hg3
c
cCALLED BY:
c	HapkeBDREFPhaseFunctions
c
cCALLS:
c	NONE
c
cINPUT PARAMETERS:
c	g  : phase_angle (radians)
c	af : forward asymmetry parameter [0,1]
c	ab : backward asymmetry parameter [-1,0]
c	f  : forward fraction [0,1]
c
cPURPOSE:
c	Calculate three parameter Henyey-Greenstein phase function
c
cREFERENCE:
c

      IMPLICIT NONE

      REAL g, af, ab, f
      REAL phf_hg1
      REAL forward, backward

      forward = phf_hg1(g, af)
      backward = phf_hg1(g, ab)

      phf_hg3 = f * forward + (1.0 - f) * backward

      RETURN
      END

c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c
c---------------------------------------------------------------------------c
