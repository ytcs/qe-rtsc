!
! rtsc: vkb of many k-points in one set of kernels (moved verbatim from PHonon/PH/rtsc_phq_fuse.f90, NSCFROT-LOG
! RTSC_PHQ_VKB, so that the engine adapter's per-k loop can use it too: RTSC_STH_KDEV, SOLVE-LOG.md session 4).
!
MODULE rtsc_vkb
  USE kinds, ONLY : DP
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: vkb_batch
CONTAINS
  !------------------------------------------------------------------------------------------
  SUBROUTINE vkb_batch(n, ikl, vkbb)
    !------------------------------------------------------------------------------------------
    !! vkbb(:, :, j) (device) = init_us_2's vkb of k-point ikl(j), j = 1..n, with the same formulas and the same
    !! device routines (ylmr2, interp_beta), element by element, so the result is bitwise init_us_2's; the kernels
    !! run once for the n k-points instead of once per k.
    USE wvfct,      ONLY : npwx
    USE klist,      ONLY : xk, ngk, igk_k
    USE gvect,      ONLY : g, mill, eigts1, eigts2, eigts3
    USE cell_base,  ONLY : tpiba
    USE ions_base,  ONLY : nat, ityp, tau
    USE uspp,       ONLY : nkb, nhtol, nhtolm, indv
    USE uspp_param, ONLY : lmaxkb, nbetam, nhm, nh, nsp
    USE beta_mod,   ONLY : interp_beta
    USE constants,  ONLY : tpi
    INTEGER, INTENT(IN) :: n, ikl(n)
    COMPLEX(DP), INTENT(INOUT) :: vkbb(:,:,:)
    INTEGER :: nx, ntot, j, ig, f, iv, nt, na, ih, jkb, nhnt, nb, lm, ik, lmax2
    INTEGER, ALLOCATABLE :: igkb(:,:), npwb(:)
    REAL(DP), ALLOCATABLE :: xkb(:,:), argb(:,:), gk(:,:), qg(:), ylm(:,:), vq(:,:), vkb1(:,:)
    COMPLEX(DP), ALLOCATABLE :: sk(:)
    COMPLEX(DP) :: pref
    !
    IF (lmaxkb < 0) RETURN
    CALL start_clock('init_us_2')
    nx = npwx; ntot = nx * n; lmax2 = (lmaxkb + 1)**2
    ALLOCATE(igkb(nx, n), npwb(n), xkb(3, n), argb(nat, n))
    DO j = 1, n
      ik = ikl(j)
      npwb(j) = ngk(ik)
      igkb(:, j) = 1
      igkb(1:ngk(ik), j) = igk_k(1:ngk(ik), ik)
      xkb(:, j) = xk(:, ik)
      DO na = 1, nat
        argb(na, j) = (xk(1, ik) * tau(1, na) + xk(2, ik) * tau(2, na) + xk(3, ik) * tau(3, na)) * tpi
      ENDDO
    ENDDO
    ALLOCATE(gk(3, ntot), qg(ntot), ylm(ntot, lmax2), vq(ntot, nbetam), vkb1(ntot, nhm), sk(ntot))
    !$acc data copyin(igkb, npwb, xkb, argb, nhtol, nhtolm, indv) create(gk, qg, ylm, vq, vkb1, sk) &
    !$acc      present(g, mill, eigts1, eigts2, eigts3, vkbb)
    !$acc parallel loop collapse(2)
    DO j = 1, n
      DO ig = 1, nx
        f = (j - 1) * nx + ig
        IF (ig <= npwb(j)) THEN
          iv = igkb(ig, j)
          gk(1, f) = xkb(1, j) + g(1, iv)
          gk(2, f) = xkb(2, j) + g(2, iv)
          gk(3, f) = xkb(3, j) + g(3, iv)
          qg(f) = gk(1, f) * gk(1, f) + gk(2, f) * gk(2, f) + gk(3, f) * gk(3, f)
        ELSE
          gk(1, f) = 0.0_DP; gk(2, f) = 0.0_DP; gk(3, f) = 0.0_DP; qg(f) = 0.0_DP
        ENDIF
      ENDDO
    ENDDO
    CALL ylmr2(lmax2, ntot, gk, qg, ylm)
    !$acc parallel loop
    DO f = 1, ntot
      qg(f) = SQRT(qg(f)) * tpiba
    ENDDO
    !$acc parallel loop collapse(3)
    DO j = 1, n
      DO ih = 1, nkb
        DO ig = 1, nx
          vkbb(ig, ih, j) = (0.0_DP, 0.0_DP)
        ENDDO
      ENDDO
    ENDDO
    jkb = 0
    DO nt = 1, nsp
      CALL interp_beta(nt, ntot, qg, vq)
      nhnt = nh(nt)
      !$acc parallel loop collapse(2)
      DO ih = 1, nhnt
        DO f = 1, ntot
          nb = indv(ih, nt)
          lm = nhtolm(ih, nt)
          vkb1(f, ih) = ylm(f, lm) * vq(f, nb)
        ENDDO
      ENDDO
      DO na = 1, nat
        IF (ityp(na) /= nt) CYCLE
        !$acc parallel loop collapse(2)
        DO j = 1, n
          DO ig = 1, nx
            f = (j - 1) * nx + ig
            IF (ig <= npwb(j)) THEN
              iv = igkb(ig, j)
              sk(f) = eigts1(mill(1, iv), na) * eigts2(mill(2, iv), na) * eigts3(mill(3, iv), na) * &
                      CMPLX(COS(argb(na, j)), -SIN(argb(na, j)), KIND=DP)
            ENDIF
          ENDDO
        ENDDO
        !$acc parallel loop collapse(3)
        DO j = 1, n
          DO ih = 1, nhnt
            DO ig = 1, nx
              IF (ig <= npwb(j)) THEN
                f = (j - 1) * nx + ig
                pref = (0.d0, -1.d0)**nhtol(ih, nt)
                vkbb(ig, jkb + ih, j) = vkb1(f, ih) * sk(f) * pref
              ENDIF
            ENDDO
          ENDDO
        ENDDO
        jkb = jkb + nhnt
      ENDDO
    ENDDO
    !$acc end data
    DEALLOCATE(igkb, npwb, xkb, argb, gk, qg, ylm, vq, vkb1, sk)
    CALL stop_clock('init_us_2')
  END SUBROUTINE vkb_batch
END MODULE rtsc_vkb
