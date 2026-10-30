!
! rtsc: modified Broyden mixing of the DFPT induced potential with a Kerker preconditioner
! (RTSC_MIX_Q0 = q0 > 0, in bohr^-1; groups/compute/SOLVE-LOG.md).
!
! Same algebra as mix_potential (LR_Modules/mix_pot.f90, in-memory branch: the residual differences
! df and input differences dv of the last n_iter iterations, Johnson's weights w0 = 0.01, w = 1),
! except that the step along the residual is preconditioned:
!   vin_new = vin + alpha * P [ R - sum_i gamma_i df_i ] - sum_i gamma_i dv_i,
! with R = vout - vin and P(q+G) = |q+G|^2 / (|q+G|^2 + q0^2) in reciprocal space (1 outside the G
! sphere and at q+G = 0). P is the initial inverse Jacobian of Broyden's method: in a metal the
! long-wavelength components of the residual are screened by eps ~ 1 + k_TF^2/|q+G|^2, so plain
! mixing overshoots them (the |ddv|^2 jump at iteration 2 and the slow irreps at small q).
! The convergence test (dr2 < tr2 on the raw residual) is unchanged, so the SCF stops by the same
! criterion; only the path changes.
!
MODULE rtsc_mix
  USE kinds, ONLY : DP
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: rtsc_mix_q0, rtsc_mix_kerker
  COMPLEX(DP), ALLOCATABLE, SAVE :: df(:,:), dv(:,:)
  REAL(DP), ALLOCATABLE, SAVE :: pfac(:)
  REAL(DP), SAVE :: pfac_q(3) = 1.d10, pfac_q0 = -1.d0
  !
CONTAINS
  !
  REAL(DP) FUNCTION rtsc_mix_q0()
    !! RTSC_MIX_Q0 (bohr^-1); 0 = off
    CHARACTER(LEN=32) :: s
    INTEGER :: ios
    REAL(DP) :: v
    rtsc_mix_q0 = 0.d0
    CALL get_environment_variable('RTSC_MIX_Q0', s, STATUS=ios)
    IF (ios /= 0) RETURN
    READ(s, *, IOSTAT=ios) v
    IF (ios == 0 .AND. v > 0.d0) rtsc_mix_q0 = v
  END FUNCTION rtsc_mix_q0
  !
  SUBROUTINE make_pfac(q0)
    USE fft_base,  ONLY : dfftp
    USE gvect,     ONLY : ngm, g
    USE cell_base, ONLY : tpiba2
    USE qpoint,    ONLY : xq
    REAL(DP), INTENT(IN) :: q0
    INTEGER :: ig
    REAL(DP) :: g2
    IF (ALLOCATED(pfac)) THEN
      IF (SIZE(pfac) == dfftp%nnr .AND. ALL(pfac_q == xq) .AND. pfac_q0 == q0) RETURN
      DEALLOCATE(pfac)
    ENDIF
    ALLOCATE(pfac(dfftp%nnr))
    pfac = 1.d0
    DO ig = 1, ngm
      g2 = ((xq(1) + g(1, ig))**2 + (xq(2) + g(2, ig))**2 + (xq(3) + g(3, ig))**2) * tpiba2
      IF (g2 > 1.d-8) pfac(dfftp%nl(ig)) = g2 / (g2 + q0**2)
    ENDDO
    pfac_q = xq
    pfac_q0 = q0
  END SUBROUTINE make_pfac
  !
  SUBROUTINE precondition(nnr, npert, u, q0)
    !! u(:, ip) <- P u(:, ip) (dense grid)
    USE fft_base,       ONLY : dfftp
    USE fft_interfaces, ONLY : fwfft, invfft
    INTEGER, INTENT(IN) :: nnr, npert
    COMPLEX(DP), INTENT(INOUT) :: u(nnr, npert)
    REAL(DP), INTENT(IN) :: q0
    INTEGER :: ip
    CALL make_pfac(q0)
    DO ip = 1, npert
      CALL fwfft('Rho', u(:, ip), dfftp)
      u(:, ip) = u(:, ip) * pfac(:)
      CALL invfft('Rho', u(:, ip), dfftp)
    ENDDO
  END SUBROUTINE precondition
  !
  SUBROUTINE rtsc_mix_kerker(nnr, npert, vout, vin, alphamix, dr2, tr2, iter, n_iter, conv, q0)
    !! Drop-in for mix_potential(2*nnr*npert, vout, vin, alphamix, dr2, tr2, iter, n_iter, ' ', conv)
    !! for nspin_mag = 1, one band group, no PAW. vout is overwritten (as in mix_potential).
    USE mp_bands, ONLY : intra_bgrp_comm
    USE mp,       ONLY : mp_sum
    INTEGER, INTENT(IN) :: nnr, npert, n_iter
    INTEGER, INTENT(INOUT) :: iter
    COMPLEX(DP), INTENT(INOUT) :: vout(nnr, npert), vin(nnr, npert)
    REAL(DP), INTENT(IN) :: alphamix, tr2, q0
    REAL(DP), INTENT(OUT) :: dr2
    LOGICAL, INTENT(OUT) :: conv
    COMPLEX(DP), ALLOCATABLE :: vinsave(:,:), u(:,:)
    REAL(DP), ALLOCATABLE :: beta(:,:), work(:), w(:)
    INTEGER, ALLOCATABLE :: iwork(:)
    INTEGER :: iter_used, ipos, inext, i, j, info, ndimtot
    REAL(DP) :: norm, gamma
    REAL(DP), PARAMETER :: w0 = 0.01_DP, w1 = 1.0_DP
    !
    CALL start_clock('mix_pot')
    IF (iter < 1) CALL errore('rtsc_mix_kerker', 'iter must be positive', 1)
    vout = vout - vin
    dr2 = SUM(REAL(vout, DP)**2 + AIMAG(vout)**2)
    ndimtot = 2 * nnr * npert
    CALL mp_sum(dr2, intra_bgrp_comm)
    CALL mp_sum(ndimtot, intra_bgrp_comm)
    dr2 = (SQRT(dr2) / ndimtot)**2
    conv = dr2 < tr2
    IF (iter > 1 .AND. .NOT. ALLOCATED(df)) THEN
      CALL infomsg('rtsc_mix_kerker', 'no mixing history, restarting')
      iter = 1
    ENDIF
    IF (iter == 1) THEN
      IF (ALLOCATED(df)) DEALLOCATE(df, dv)
      ALLOCATE(df(nnr * npert, n_iter), dv(nnr * npert, n_iter))
    ENDIF
    IF (conv) THEN
      IF (ALLOCATED(df)) DEALLOCATE(df, dv)
      CALL stop_clock('mix_pot')
      RETURN
    ENDIF
    ALLOCATE(vinsave(nnr, npert))
    iter_used = MIN(iter - 1, n_iter)
    ipos = iter - 1 - ((iter - 2) / n_iter) * n_iter
    IF (iter > 1) THEN
      df(:, ipos) = RESHAPE(vout, (/ nnr * npert /)) - df(:, ipos)
      dv(:, ipos) = RESHAPE(vin, (/ nnr * npert /)) - dv(:, ipos)
      norm = SUM(REAL(df(:, ipos), DP)**2 + AIMAG(df(:, ipos))**2)
      CALL mp_sum(norm, intra_bgrp_comm)
      norm = SQRT(norm)
      df(:, ipos) = df(:, ipos) / norm
      dv(:, ipos) = dv(:, ipos) / norm
    ENDIF
    vinsave = vin
    ALLOCATE(u(nnr, npert))
    u = vout
    IF (iter_used > 0) THEN
      ALLOCATE(beta(iter_used, iter_used), w(iter_used), work(iter_used), iwork(iter_used))
      w(:) = w1
      beta = 0.0_DP
      DO i = 1, iter_used
        DO j = i + 1, iter_used
          beta(i, j) = w(i) * w(j) * SUM(REAL(df(:, j)) * REAL(df(:, i)) + AIMAG(df(:, j)) * AIMAG(df(:, i)))
          CALL mp_sum(beta(i, j), intra_bgrp_comm)
        ENDDO
        beta(i, i) = w0**2 + w(i)**2
      ENDDO
      CALL DSYTRF('U', iter_used, beta, iter_used, iwork, work, iter_used, info)
      CALL errore('broyden', 'factorization', info)
      CALL DSYTRI('U', iter_used, beta, iter_used, iwork, work, info)
      CALL errore('broyden', 'DSYTRI', info)
      DO i = 1, iter_used
        DO j = i + 1, iter_used
          beta(j, i) = beta(i, j)
        ENDDO
      ENDDO
      DO i = 1, iter_used
        work(i) = SUM(REAL(df(:, i)) * REAL(RESHAPE(vout, (/ nnr * npert /))) + &
                      AIMAG(df(:, i)) * AIMAG(RESHAPE(vout, (/ nnr * npert /))))
      ENDDO
      CALL mp_sum(work, intra_bgrp_comm)
      DO i = 1, iter_used
        gamma = 0.d0
        DO j = 1, iter_used
          gamma = gamma + beta(j, i) * w(j) * work(j)
        ENDDO
        ! u = R - sum gamma df ; vin -= sum gamma dv
        u = u - RESHAPE(w(i) * gamma * df(:, i), (/ nnr, npert /))
        vin = vin - RESHAPE(w(i) * gamma * dv(:, i), (/ nnr, npert /))
      ENDDO
      DEALLOCATE(beta, w, work, iwork)
    ENDIF
    CALL precondition(nnr, npert, u, q0)
    vin = vin + alphamix * u
    inext = iter - ((iter - 1) / n_iter) * n_iter
    df(:, inext) = RESHAPE(vout, (/ nnr * npert /))
    dv(:, inext) = RESHAPE(vinsave, (/ nnr * npert /))
    DEALLOCATE(vinsave, u)
    CALL stop_clock('mix_pot')
  END SUBROUTINE rtsc_mix_kerker
  !
END MODULE rtsc_mix
