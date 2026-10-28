!
! rtsc: GPU versions of host-side ph.x steps around the batched Sternheimer engine (the "CPU tail",
! groups/compute/PH-DESIGN.md P4). Each routine is used only when RTSC_STH=1 (sth_tail_on); with
! RTSC_STH unset or 0 the vanilla code runs unchanged.
!
!   symdvscf_acc   symdvscf (symmetrization of drho / dvscf of one irrep) as one OpenACC loop over
!                  the FFT grid. Same formula; the phase exp(i g.r) is evaluated directly instead of
!                  by running products along the grid (rounding-level difference only).
!
MODULE sth_tail
  USE kinds, ONLY : DP
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: sth_tail_on, symdvscf_acc
  !
  INTEGER, SAVE :: tail_flag = -1
  !
CONTAINS
  !
  LOGICAL FUNCTION sth_tail_on()
    CHARACTER(LEN=16) :: s
    INTEGER :: ios
    IF (tail_flag < 0) THEN
      tail_flag = 0
      CALL get_environment_variable('RTSC_STH', s, STATUS=ios)
      IF (ios == 0 .AND. TRIM(s) == '1') tail_flag = 1
      CALL get_environment_variable('RTSC_TAIL', s, STATUS=ios)     ! RTSC_TAIL=0: engine only
      IF (ios == 0 .AND. TRIM(s) == '0') tail_flag = 0
    ENDIF
    sth_tail_on = (tail_flag == 1)
  END FUNCTION sth_tail_on
  !
  !------------------------------------------------------------------------------------------------
  SUBROUTINE symdvscf_acc(dvtosym, dfft)
    !------------------------------------------------------------------------------------------------
    USE constants,        ONLY : tpi
    USE fft_base,         ONLY : fft_type_descriptor
    USE cell_base,        ONLY : at
    USE symm_base,        ONLY : s, ft, t_rev
    USE noncollin_module, ONLY : nspin_lsda, nspin_mag
    USE control_lr,       ONLY : lgamma
    USE lr_symm_base,     ONLY : minus_q, irotmq, nsymq, gi, gimq, lr_npert, upert, upert_mq
    TYPE(fft_type_descriptor), INTENT(IN) :: dfft
    COMPLEX(DP), INTENT(INOUT) :: dvtosym(dfft%nr1x, dfft%nr2x, dfft%nr3x, nspin_mag, lr_npert)
    !
    INTEGER, PARAMETER :: MAXP = 6
    INTEGER :: ftau(3, nsymq), ss(3, 3, nsymq), trev(nsymq)
    REAL(DP) :: gf(3, nsymq), gfm(3), n(3), arg, rn
    COMPLEX(DP), ALLOCATABLE :: dvsym(:,:,:,:)
    COMPLEX(DP) :: up(lr_npert, lr_npert, nsymq), upm(lr_npert, lr_npert)
    COMPLEX(DP) :: ac(MAXP), ph, x
    INTEGER :: nr1, nr2, nr3, nr1x, nr2x, nr3x, np, ns, is, i, j, k, ri, rj, rk, isym, ipert, jpert
    INTEGER :: is_, irm
    !
    IF (nsymq == 1 .AND. .NOT. minus_q) RETURN
    np = lr_npert
    IF (np > MAXP) CALL errore('symdvscf_acc', 'lr_npert > 6', np)
    CALL start_clock('symdvscf')
    nr1 = dfft%nr1; nr2 = dfft%nr2; nr3 = dfft%nr3
    nr1x = dfft%nr1x; nr2x = dfft%nr2x; nr3x = dfft%nr3x
    ns = nsymq
    n(1) = tpi / DBLE(nr1); n(2) = tpi / DBLE(nr2); n(3) = tpi / DBLE(nr3)
    CALL scale_sym_ops(nsymq, s, ft, nr1, nr2, nr3, ss, ftau)
    DO isym = 1, ns
      gf(:, isym) = gi(1, isym) * at(1, :) * n(:) + gi(2, isym) * at(2, :) * n(:) + gi(3, isym) * at(3, :) * n(:)
      trev(isym) = t_rev(isym)
      up(:, :, isym) = upert(1:np, 1:np, isym)
    ENDDO
    gfm(:) = gimq(1) * at(1, :) * n(:) + gimq(2) * at(2, :) * n(:) + gimq(3) * at(3, :) * n(:)
    IF (minus_q .AND. .NOT. lgamma) upm(:, :) = upert_mq(1:np, 1:np)
    irm = 1
    IF (minus_q) irm = irotmq
    rn = 1.d0 / DBLE(ns)
    !
    ALLOCATE(dvsym(nr1x, nr2x, nr3x, np))
    !$acc data copy(dvtosym) create(dvsym) copyin(ss, ftau, gf, gfm, trev, up, upm)
    DO is_ = 1, nspin_lsda
      is = is_
      IF (minus_q) THEN
        IF (lgamma) THEN
          !$acc parallel loop collapse(4)
          DO ipert = 1, np
            DO k = 1, nr3x
              DO j = 1, nr2x
                DO i = 1, nr1x
                  dvtosym(i, j, k, is, ipert) = CMPLX(DBLE(dvtosym(i, j, k, is, ipert)), 0.d0, KIND=DP)
                ENDDO
              ENDDO
            ENDDO
          ENDDO
        ELSE
          !$acc parallel loop collapse(3) gang vector private(ac)
          DO k = 1, nr3
            DO j = 1, nr2
              DO i = 1, nr1
                ri = ss(1, 1, irm) * (i - 1) + ss(2, 1, irm) * (j - 1) + ss(3, 1, irm) * (k - 1) - ftau(1, irm)
                ri = MOD(ri, nr1) + 1
                IF (ri < 1) ri = ri + nr1
                rj = ss(1, 2, irm) * (i - 1) + ss(2, 2, irm) * (j - 1) + ss(3, 2, irm) * (k - 1) - ftau(2, irm)
                rj = MOD(rj, nr2) + 1
                IF (rj < 1) rj = rj + nr2
                rk = ss(1, 3, irm) * (i - 1) + ss(2, 3, irm) * (j - 1) + ss(3, 3, irm) * (k - 1) - ftau(3, irm)
                rk = MOD(rk, nr3) + 1
                IF (rk < 1) rk = rk + nr3
                arg = gfm(1) * (i - 1) + gfm(2) * (j - 1) + gfm(3) * (k - 1)
                ph = CMPLX(COS(arg), SIN(arg), KIND=DP)
                !$acc loop seq
                DO ipert = 1, np
                  x = (0.d0, 0.d0)
                  !$acc loop seq
                  DO jpert = 1, np
                    x = x + upm(jpert, ipert) * dvtosym(ri, rj, rk, is, jpert) * ph
                  ENDDO
                  dvsym(i, j, k, ipert) = (dvtosym(i, j, k, is, ipert) + CONJG(x)) * 0.5d0
                ENDDO
              ENDDO
            ENDDO
          ENDDO
          !$acc parallel loop collapse(4)
          DO ipert = 1, np
            DO k = 1, nr3
              DO j = 1, nr2
                DO i = 1, nr1
                  dvtosym(i, j, k, is, ipert) = dvsym(i, j, k, ipert)
                ENDDO
              ENDDO
            ENDDO
          ENDDO
        ENDIF
      ENDIF
      IF (ns == 1) CYCLE
      !
      !$acc parallel loop collapse(4)
      DO ipert = 1, np
        DO k = 1, nr3x
          DO j = 1, nr2x
            DO i = 1, nr1x
              dvsym(i, j, k, ipert) = (0.d0, 0.d0)
            ENDDO
          ENDDO
        ENDDO
      ENDDO
      !$acc parallel loop collapse(3) gang vector private(ac)
      DO k = 1, nr3
        DO j = 1, nr2
          DO i = 1, nr1
            !$acc loop seq
            DO ipert = 1, np
              ac(ipert) = (0.d0, 0.d0)
            ENDDO
            !$acc loop seq
            DO isym = 1, ns
              ri = ss(1, 1, isym) * (i - 1) + ss(2, 1, isym) * (j - 1) + ss(3, 1, isym) * (k - 1) - ftau(1, isym)
              ri = MOD(ri, nr1) + 1
              IF (ri < 1) ri = ri + nr1
              rj = ss(1, 2, isym) * (i - 1) + ss(2, 2, isym) * (j - 1) + ss(3, 2, isym) * (k - 1) - ftau(2, isym)
              rj = MOD(rj, nr2) + 1
              IF (rj < 1) rj = rj + nr2
              rk = ss(1, 3, isym) * (i - 1) + ss(2, 3, isym) * (j - 1) + ss(3, 3, isym) * (k - 1) - ftau(3, isym)
              rk = MOD(rk, nr3) + 1
              IF (rk < 1) rk = rk + nr3
              arg = gf(1, isym) * (i - 1) + gf(2, isym) * (j - 1) + gf(3, isym) * (k - 1)
              ph = CMPLX(COS(arg), SIN(arg), KIND=DP)
              !$acc loop seq
              DO ipert = 1, np
                x = (0.d0, 0.d0)
                !$acc loop seq
                DO jpert = 1, np
                  x = x + up(jpert, ipert, isym) * dvtosym(ri, rj, rk, is, jpert) * ph
                ENDDO
                IF (trev(isym) == 0) THEN
                  ac(ipert) = ac(ipert) + x
                ELSE
                  ac(ipert) = ac(ipert) + CONJG(x)
                ENDIF
              ENDDO
            ENDDO
            !$acc loop seq
            DO ipert = 1, np
              dvsym(i, j, k, ipert) = ac(ipert)
            ENDDO
          ENDDO
        ENDDO
      ENDDO
      !$acc parallel loop collapse(4)
      DO ipert = 1, np
        DO k = 1, nr3x
          DO j = 1, nr2x
            DO i = 1, nr1x
              dvtosym(i, j, k, is, ipert) = dvsym(i, j, k, ipert) / DBLE(ns)
            ENDDO
          ENDDO
        ENDDO
      ENDDO
    ENDDO
    !$acc end data
    DEALLOCATE(dvsym)
    CALL stop_clock('symdvscf')
  END SUBROUTINE symdvscf_acc
  !
END MODULE sth_tail
