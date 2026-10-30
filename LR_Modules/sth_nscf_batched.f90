!
! rtsc: QE-side adapter of the batched nscf Davidson (rtsc/sth/nscf_engine.f90). c_bands_nscf_ph calls
! nscf_batched_run before its k loop. With RTSC_STH=1 (and RTSC_NSCF_BATCH not 0) and a supported case,
! all k-points are diagonalized together on the GPU, from the same starting vectors as the vanilla loop
! (rotated scf wavefunctions, or init_wfc), to the same threshold ethr; their wavefunctions are written
! to iunwfc and their eigenvalues to et, and kdone(ik) is set. k-points not done (unsupported case,
! zhegv failure, no convergence) go through the vanilla per-k Davidson in c_bands_nscf_ph.
!
! Supported: norm-conserving, collinear, nspin = 1, no DFT+U, no meta-GGA / EXX / real_space,
! Davidson (isolve = 0), one band group, serial FFT, no restart inside the nscf.
!
MODULE sth_nscf_batched
  USE kinds, ONLY : DP
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: nscf_batched_run
  !
CONTAINS
  !
  SUBROUTINE nscf_batched_run(ik_, kdone, avg_iter, kskip)
    USE io_global,        ONLY : stdout
    USE klist,            ONLY : nks, xk, ngk, igk_k
    USE wvfct,            ONLY : et, nbnd, npwx, g2kin, current_k
    USE uspp,             ONLY : vkb, nkb, deeq, ofsbeta, okvan
    USE uspp_param,       ONLY : nh
    USE uspp_init,        ONLY : init_us_2
    USE ions_base,        ONLY : nat, ityp
    USE scf,              ONLY : vrs, v_of_0
    USE fft_base,         ONLY : dffts
    USE wavefunctions,    ONLY : evc
    USE io_files,         ONLY : iunwfc, nwordwfc
    USE buffers,          ONLY : save_buffer
    USE control_flags,    ONLY : ethr, isolve, io_level, gamma_only, david
    USE noncollin_module, ONLY : noncolin, npol
    USE lsda_mod,         ONLY : lsda
    USE ldaU,             ONLY : lda_plus_u
    USE mp_bands,         ONLY : nproc_bgrp, nbgrp
    USE realus,           ONLY : real_space
    USE xc_lib,           ONLY : exx_is_active, xclib_dft_is
    USE sth_nscf_start,   ONLY : nscf_start_get
#if defined(__CUDA)
    USE nscf_engine,      ONLY : nscf_eng_init, nscf_eng_set_k, nscf_eng_solve, nscf_eng_get_k, &
                                 nscf_eng_free, nscf_eng_kmax, nscf_eng_nit, nscf_eng_t_eig, nscf_eng_t_tot
#endif
    !
    INTEGER, INTENT(IN) :: ik_
    LOGICAL, INTENT(INOUT) :: kdone(nks)
    REAL(DP), INTENT(INOUT) :: avg_iter
    LOGICAL, INTENT(IN), OPTIONAL :: kskip(nks)
    ! rtsc (RTSC_NSCF_SYMROT): k-points already done (kdone) or marked kskip are not diagonalized here;
    ! the caller initializes kdone
    !
#if defined(__CUDA)
    CHARACTER(LEN=16) :: s
    INTEGER :: ios, ik, k0, j, nkc, chunk, npw, na, ih, jh, niter, nfail, maxit
    INTEGER(8) :: nhpsi
    INTEGER, ALLOCATABLE :: idx(:), kl(:)
    INTEGER :: nkl
    COMPLEX(DP), ALLOCATABLE :: dmat(:,:)
    LOGICAL :: ok, strict, unsup
    INTEGER(8) :: c0, c1, cr
    REAL(DP) :: tset, tsol, teig
    INTEGER :: nit
#endif
    !
#if defined(__CUDA)
    CALL get_environment_variable('RTSC_STH', s, STATUS=ios)
    IF (.NOT. (ios == 0 .AND. TRIM(s) == '1')) RETURN
    CALL get_environment_variable('RTSC_NSCF_BATCH', s, STATUS=ios)
    IF (ios == 0 .AND. TRIM(s) == '0') RETURN
    CALL get_environment_variable('RTSC_STRICT', s, STATUS=ios)
    strict = (ios == 0 .AND. TRIM(s) == '1')
    unsup = ik_ /= 0 .OR. isolve /= 0 .OR. david < 2
    unsup = unsup .OR. okvan .OR. noncolin .OR. npol /= 1 .OR. lsda .OR. lda_plus_u .OR. gamma_only
    unsup = unsup .OR. xclib_dft_is('meta') .OR. exx_is_active() .OR. real_space
    unsup = unsup .OR. nproc_bgrp /= 1 .OR. nbgrp /= 1 .OR. dffts%has_task_groups
    unsup = unsup .OR. dffts%nnr /= dffts%nr1x * dffts%nr2x * dffts%nr3x
    IF (unsup) THEN
      ! RTSC_STRICT=1 (scored runs): no silent fallback to the per-k Davidson
      IF (strict) CALL errore('nscf_batched_run', 'case not supported by the batched nscf and RTSC_STRICT=1', 1)
      RETURN
    ENDIF
    !
    CALL start_clock('nscf_batch')
    maxit = 100
    ALLOCATE(dmat(MAX(nkb, 1), MAX(nkb, 1)), idx(npwx))
    dmat = (0.d0, 0.d0)
    DO na = 1, nat
      DO ih = 1, nh(ityp(na))
        DO jh = 1, nh(ityp(na))
          dmat(ofsbeta(na) + ih, ofsbeta(na) + jh) = CMPLX(deeq(ih, jh, na, 1), 0.d0, KIND=DP)
        ENDDO
      ENDDO
    ENDDO
    ALLOCATE(kl(nks))
    nkl = 0
    DO ik = 1, nks
      IF (kdone(ik)) CYCLE
      IF (PRESENT(kskip)) THEN
        IF (kskip(ik)) CYCLE
      ENDIF
      nkl = nkl + 1
      kl(nkl) = ik
    ENDDO
    IF (nkl == 0) THEN
      DEALLOCATE(dmat, idx, kl)
      CALL stop_clock('nscf_batch')
      RETURN
    ENDIF
    chunk = MIN(nkl, nscf_eng_kmax(npwx, nbnd, MAX(nkb, 1), david, dffts%nnr))
    nfail = 0
    tset = 0; tsol = 0; teig = 0; nit = 0
    DO k0 = 1, nkl, chunk
      nkc = MIN(chunk, nkl - k0 + 1)
      CALL nscf_eng_init(nkc, npwx, nbnd, nkb, david, dffts%nr1, dffts%nr2, dffts%nr3, dffts%nr1x, &
                         dffts%nr2x, dffts%nr3x, dffts%nnr, vrs(:, 1), dmat, v_of_0)
      CALL SYSTEM_CLOCK(c0, cr)
      DO j = 1, nkc
        ik = kl(k0 + j - 1)
        current_k = ik
        npw = ngk(ik)
        CALL g2_kin(ik)
        IF (nkb > 0) THEN
          CALL init_us_2(npw, igk_k(1, ik), xk(1, ik), vkb, .true.)
          !$acc update host(vkb)
        ENDIF
        IF (.NOT. nscf_start_get(ik, evc)) THEN
          CALL init_wfc(ik)
          !$acc update self(evc)
        ENDIF
        idx(1:npw) = dffts%nl(igk_k(1:npw, ik))
        CALL nscf_eng_set_k(j, npw, idx, g2kin, vkb, evc)
      ENDDO
      CALL SYSTEM_CLOCK(c1)
      tset = tset + REAL(c1 - c0, DP) / REAL(cr, DP)
      CALL nscf_eng_solve(ethr, maxit, nhpsi)
      tsol = tsol + nscf_eng_t_tot; teig = teig + nscf_eng_t_eig; nit = MAX(nit, nscf_eng_nit)
      DO j = 1, nkc
        ik = kl(k0 + j - 1)
        CALL nscf_eng_get_k(j, evc, et(:, ik), ok, niter)
        IF (ok) THEN
          IF (io_level > -1) CALL save_buffer(evc, nwordwfc, iunwfc, ik)
          kdone(ik) = .TRUE.
          avg_iter = avg_iter + niter
        ELSE
          nfail = nfail + 1
        ENDIF
      ENDDO
      CALL nscf_eng_free()
    ENDDO
    !$acc update device(et)
    DEALLOCATE(dmat, idx, kl)
    WRITE(stdout, '(5x,"rtsc nscf: batched Davidson on ",I0," k-points (",I0," per chunk), ",I0, &
         &" left to the per-k Davidson")') nkl - nfail, chunk, nfail
    WRITE(stdout, '(5x,"rtsc nscf: setup ",F7.2," s, solve ",F7.2," s (host eigensolver ",F7.2," s), ",I0, &
         &" steps")') tset, tsol, teig, nit
    CALL stop_clock('nscf_batch')
#endif
  END SUBROUTINE nscf_batched_run
  !
END MODULE sth_nscf_batched
