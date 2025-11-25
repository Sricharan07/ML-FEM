      subroutine vumat(
C Read only -
     1     nblock, ndir, nshr, nstatev, nfieldv, nprops, lanneal,
     2     stepTime, totalTime, dt, cmname, coordMp, charLength,
     3     props, density, strainInc, relSpinInc,
     4     tempOld, stretchOld, defgradOld, fieldOld,
     5     stressOld, stateOld, enerInternOld, enerInelasOld,
     6     tempNew, stretchNew, defgradNew, fieldNew,
C Write only -
     7     stressNew, stateNew, enerInternNew, enerInelasNew)
C
      include 'vaba_param.inc'
C
      dimension props(nprops), density(nblock), coordMp(nblock,*),
     1     charLength(nblock), strainInc(nblock,ndir+nshr),
     2     relSpinInc(nblock,nshr), tempOld(nblock),
     3     stretchOld(nblock,ndir+nshr),
     4     defgradOld(nblock,ndir+nshr+nshr),
     5     fieldOld(nblock,nfieldv), stressOld(nblock,ndir+nshr),
     6     stateOld(nblock,nstatev), enerInternOld(nblock),
     7     enerInelasOld(nblock), tempNew(nblock),
     8     stretchNew(nblock,ndir+nshr),
     9     defgradNew(nblock,ndir+nshr+nshr),
     1     fieldNew(nblock,nfieldv),
     2     stressNew(nblock,ndir+nshr), stateNew(nblock,nstatev),
     3     enerInternNew(nblock), enerInelasNew(nblock)
!
      character*80 cmname
      integer k, k1

C --- Dense layer weights and biases,Batch normalization parameters,Scaler parameters for input and output
C --- UPDATED: Changed from 18 to 17 input features to match training model
      COMMON /NNMODEL/ W1(17,128),b1(128),W2(128,96),b2(96),W3(96,64),b3(64),W4(64,32),b4(32),
     1 W5(32,6),b5(6),bn1_gamma(128),bn1_beta(128),bn1_mean(128),bn1_var(128),bn2_gamma(96),
     2 bn2_beta(96),bn2_mean(96),bn2_var(96),bn3_gamma(64),bn3_beta(64),bn3_mean(64),bn3_var(64),
     3 dinput_mean(17),dinput_scale(17),output_mean(6),output_scale(6)

C inputs
      real*8 e, nu, mu, alamda
      real*8 :: totalStrain(6),AI_INPUT(17),AI_OUTPUT(nblock,6)
      real*8 :: eff_strain,delta_eff_strain,strain_energy_inc
      REAL*8 :: delta_stress(6), delta_eff_stress
      real*8 :: BN_EPSILON = 1.d-03

C --- UPDATED: Changed from 18 to 17 input features to match training model
      INTEGER, PARAMETER :: n_input=17, n_h1=128, n_h2=96, n_h3=64, n_h4=32, n_output=6

c        open(206,file='/home/dbaruah/Sricharan_AI_Model/Flat_Plate/AI_S11.dat',position='append')

C      CLOSE(206)

C --- Scaled input vector and hidden layer activations
      real*8 :: DINPUT_SCALED(n_input)
      real*8 :: h1_pre(n_h1), h1_bn(n_h1), h1_relu(n_h1), h1(n_h1)
      real*8 :: h2_pre(n_h2), h2_bn(n_h2), h2_relu(n_h2), h2(n_h2)
      real*8 :: h3_pre(n_h3), h3_bn(n_h3), h3_relu(n_h3), h3(n_h3)
      real*8 :: h4(n_h4)
      real*8 :: OUTPUT_SCALED(n_output)
      PARAMETER(TINY_VAL = 1.d-16)

      integer :: i, j

      open(206,file='/home/dbaruah/Sricharan_AI_Model/Flat_Plate/AI_S11.dat',position='append')

      e = props(1)                  ! Young's modulus
      nu = props(2)                 ! poisson's ratio

c lame's parameters
      mu = e/(2.d0*(1.d0 + nu))
      alamda = e*nu/((1.d0 + nu) * (1.d0 - 2.d0*nu))

      ncount=0

c stress increment evaluation for each element
      do 10 k = 1, nblock

c         WRITE(*,*) 'Integration Point Number: ', k
C        WRITE(*,*) 'Coordinates: ', coordMp(k,1),coordMp(k,2),coordMp(k,3)

       totalStrain(1) = stateOld(k,1) + strainInc(k,1)
       totalStrain(2) = stateOld(k,2) + strainInc(k,2)
       totalStrain(3) = stateOld(k,3) + strainInc(k,3)
       totalStrain(4) = stateOld(k,4) + 2.d0*strainInc(k,4)
       totalStrain(5) = stateOld(k,5) + 2.d0*strainInc(k,5)
       totalStrain(6) = stateOld(k,6) + 2.d0*strainInc(k,6)

C --- UPDATED: 15 FEATURES TO MATCH TRAINING MODEL
C --- Features 1-6: Current total strains (strain_n+1) ✅ CORRECT
       AI_INPUT(1) = totalStrain(1)   ! LE-LE11 (strain_n+1)
       AI_INPUT(2) = totalStrain(2)   ! LE-LE22
       AI_INPUT(3) = totalStrain(3)   ! LE-LE33
       AI_INPUT(4) = totalStrain(4)   ! LE-LE12
       AI_INPUT(5) = totalStrain(5)   ! LE-LE13
       AI_INPUT(6) = totalStrain(6)   ! LE-LE23

C --- Features 7-12: Strain increments - CORRECTED to match Python exactly!
C --- Python: For first increment, delta = current_strain; for others, delta = current - previous
C --- In VUMAT: strainInc(k,i) is the increment from Abaqus, which matches Python logic
       AI_INPUT(7) = strainInc(k,1)              ! delta_LE-LE11
       AI_INPUT(8) = strainInc(k,2)              ! delta_LE-LE22
       AI_INPUT(9) = strainInc(k,3)              ! delta_LE-LE33
       AI_INPUT(10) = strainInc(k,4)             ! delta_LE-LE12
       AI_INPUT(11) = strainInc(k,5)             ! delta_LE-LE13
       AI_INPUT(12) = strainInc(k,6)             ! delta_LE-LE23

C --- Feature 13: Effective strain - CORRECTED to match Python exactly!
C --- Python formula: sqrt(2/3 * (ε11² + ε22² + ε33² + 2*(ε12² + ε13² + ε23²)))
      eff_strain = SQRT(2.d0/3.d0 * (
     1    totalStrain(1)**2 + totalStrain(2)**2 + totalStrain(3)**2 +
     2    2.d0*(totalStrain(4)**2 + totalStrain(5)**2 + 
     3           totalStrain(6)**2)))
      eff_strain = MAX(eff_strain, TINY_VAL)
       AI_INPUT(13) = eff_strain

C --- Feature 14: Maximum effective strain (work hardening memory)
C --- Get maximum effective strain from state variables
        IF (nstatev .GE. 7) THEN
          dmax_eff_strain = MAX(stateold(k,7), eff_strain)
        ELSE
          dmax_eff_strain = eff_strain
        END IF
      AI_INPUT(14) = dmax_eff_strain

C --- Feature 15: Virgin loading flag - CORRECTED to match Python exactly!
C --- Virgin loading: currently at maximum strain ever reached
      IF (ABS(eff_strain - dmax_eff_strain) .LT. 1.d-08) THEN
          Virgin_loading = 1.d0
      ELSE
          Virgin_loading = 0.d0
      END IF
      AI_INPUT(15) = Virgin_loading

! Calculate loading flag based on effective strain comparison
! If(eff_strain .ge. stateOld(k,15)) dloading_flag = 1.d0 else dloading_flag = 0.d0 end if
      IF (eff_strain .GE. stateOld(k,15)) THEN
          dloading_flag = 1.d0
      ELSE
          dloading_flag = 0.d0
      END IF
      AI_INPUT(16) = dloading_flag

! Calculate AI_INPUT based on stress threshold
      IF (stateOld(k,16) .GE. 140.d0) THEN
          AI_INPUT(17) = stateOld(k,14)
      ELSE
          AI_INPUT(17) = 0.d0
      END IF

C --- STEP 1: Scale input features
C --- Formula: x_scaled = (x_raw - mean) / scale
      DO i = 1, n_input
        DINPUT_SCALED(i) = (AI_INPUT(i) - dinput_mean(i)) / dinput_scale(i)
      END DO

C --- STEP 2: Forward pass with exact TensorFlow architecture
C --- TensorFlow: Dense(ReLU) -> BatchNorm -> Dropout (Dropout ignored in inference)

C --- Layer 1: Dense -> ReLU -> BatchNorm (TensorFlow order)
      DO j = 1, n_h1
        h1_pre(j) = b1(j)
        DO i = 1, n_input
          h1_pre(j) = h1_pre(j) + DINPUT_SCALED(i) * W1(i, j)
        END DO
      END DO
      ! Apply ReLU activation
      DO j = 1, n_h1
        IF (h1_pre(j).le.0.d0) THEN
          h1_relu(j) = 0.d0
        ELSE
          h1_relu(j) = h1_pre(j)
        END IF
      END DO
      ! Apply batch normalization to ReLU output
      DO j = 1, n_h1
        h1(j) = bn1_gamma(j) * (h1_relu(j) - bn1_mean(j)) / 
     &          SQRT(bn1_var(j) + BN_EPSILON) + bn1_beta(j)
      END DO

C --- Layer 2: Dense -> ReLU -> BatchNorm (TensorFlow order)
      DO j = 1, n_h2
        h2_pre(j) = b2(j)
        DO i = 1, n_h1
          h2_pre(j) = h2_pre(j) + h1(i) * W2(i, j)
        END DO
      END DO
      ! Apply ReLU activation
      DO j = 1, n_h2
        IF (h2_pre(j).le.0.d0) THEN
          h2_relu(j) = 0.d0
        ELSE
          h2_relu(j) = h2_pre(j)
        END IF
      END DO
      ! Apply batch normalization to ReLU output
      DO j = 1, n_h2
        h2(j) = bn2_gamma(j) * (h2_relu(j) - bn2_mean(j)) / 
     &          SQRT(bn2_var(j) + BN_EPSILON) + bn2_beta(j)
      END DO

C --- Layer 3: Dense -> ReLU -> BatchNorm (TensorFlow order)
      DO j = 1, n_h3
        h3_pre(j) = b3(j)
        DO i = 1, n_h2
          h3_pre(j) = h3_pre(j) + h2(i) * W3(i, j)
        END DO
      END DO
      ! Apply ReLU activation
      DO j = 1, n_h3
        IF (h3_pre(j).le.0.d0) THEN
          h3_relu(j) = 0.d0
        ELSE
          h3_relu(j) = h3_pre(j)
        END IF
      END DO
      ! Apply batch normalization to ReLU output
      DO j = 1, n_h3
        h3(j) = bn3_gamma(j) * (h3_relu(j) - bn3_mean(j)) / 
     &          SQRT(bn3_var(j) + BN_EPSILON) + bn3_beta(j)
      END DO

C --- Layer 4: Dense + ReLU (NO Batch Normalization)
      DO j = 1, n_h4
        h4(j) = b4(j)
        DO i = 1, n_h3
          h4(j) = h4(j) + h3(i) * W4(i, j)
        END DO
        IF (h4(j).le.0.d0) h4(j) = 0.d0  ! ReLU
      END DO

C --- Layer 5: Dense + Linear (NO Batch Normalization, NO ReLU)
      DO j = 1, n_output
        OUTPUT_SCALED(j) = b5(j)
        DO i = 1, n_h4
          OUTPUT_SCALED(j) = OUTPUT_SCALED(j) + h4(i) * W5(i, j)
        END DO
        ! Linear output (no activation)
      END DO

C --- STEP 3: Inverse transform output to get actual stress values
C --- Formula: y_actual = y_scaled * scale + mean
      DO j = 1, n_output
        AI_OUTPUT(k,j) = OUTPUT_SCALED(j) * output_scale(j) + output_mean(j)
      END DO

c            WRITE(7,*) 'AI_OUTPUT: ', AI_OUTPUT

! Calculate delta effective strain from strain increments
      delta_eff_strain = SQRT(2.d0/3.d0 * (strainInc(k,1)**2 + strainInc(k,2)**2 + 
     1 strainInc(k,3)**2 + 2.d0*(strainInc(k,4)**2 + strainInc(k,5)**2 + strainInc(k,6)**2)))

! Calculate delta stress components
      DO i = 1, 6
          delta_stress(i) = AI_OUTPUT(k,i) - stateOld(k,i+7)
      END DO

! Calculate delta effective stress
      delta_eff_stress = SQRT((1.d0/2.d0)*((delta_stress(1) - delta_stress(2))**2 +
     1 (delta_stress(2) - delta_stress(3))**2 + (delta_stress(3) - delta_stress(1))**2 +
     2 3.d0*((delta_stress(4))**2 + (delta_stress(5))**2 + (delta_stress(6))**2)))

! Calculate delta path length
      delta_pathlength = SQRT(((delta_eff_stress/140.d0)**2) + (delta_eff_strain**2))

      Total_pathlength = stateOld(k,14) + delta_pathlength

! Calculate max S11 stress
      dmax_stress11 = MAX(stateOld(k,8), AI_OUTPUT(k,1))

c stress increment evaluation for each element
         trInc = sum(strainInc(k, 1:3))        ! strain trace
         do k1 = 1, ndir
           stressNew(k, k1) = stressOld(k, k1) + alamda*trInc 
     1      + 2.d0*mu*strainInc(k, k1)
         end do

c shear stress
         do k1 = 1, nshr
           stressNew(k, ndir+k1) = stressOld(k, ndir+k1) + 
     1      2.d0*mu*strainInc(k, ndir+k1)
         end do

       if(mod(ncount,5000).le.1e-03) then
           if(k.eq.1) then
           WRITE(7,*) 'Time, : ', totalTime, dinput_scale
           end if
       end if

c        WRITE(206,*) 'totalStress11: ', stressNew(k,1)
c       WRITE(206,*) 'totalStress22: ', stressNew(k,2)
c       WRITE(206,*) 'totalStress33: ', stressNew(k,3)
c       WRITE(206,*) 'totalStress12: ', stressNew(k,4)
c       WRITE(206,*) 'totalStress13: ', stressNew(k,5)
c       WRITE(206,*) 'totalStress23: ', stressNew(k,6)

C       Update state variables

      do i=1,ndir+nshr
          stateNew(k,i)=totalStrain(i)
      end do

      stateNew(k,7) = dmax_eff_strain  ! Maximum effective strain
C

      do k1 = 1, ndir+nshr
          stateNew(k, k1+7) = AI_OUTPUT(k,k1)
      end do

      stateNew(k,14) =  Total_pathlength

      stateNew(k,15) =  eff_strain

      stateNew(k,16) =  dmax_stress11

      ncount=ncount+1
 10   continue
      return
      end