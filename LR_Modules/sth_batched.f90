!
! rtsc: QE-side adapter of the batched Sternheimer engine (rtsc/sth/sth_engine.f90,
! groups/compute/STH-ENGINE.md). sternheimer_kernel dispatches here when RTSC_STH=1 and the
! case is supported; otherwise (or if the engine declines) the vanilla k x pert loop runs.
!
! Supported: norm-conserving, collinear, nspin_mag = 1, no DFT+U, no time reversal, any q,
! lgauss smearing (no tetrahedra, no twochem), no meta-GGA / EXX / real_space / scissor /
! lelfield, serial FFT within the pool (nproc_bgrp = 1, no task groups), one band group.
! Any number of pools: each pool rank runs its own engine on its own k-points.
!
! At the first call of an irrep (first_iter, or no engine state for this irrep) the adapter
! gathers QE state for all k (buffers, FFT maps, vkb and g2kin at k+q, vrs, deeq, et, wk, the
! orthogonalize weights) into the engine. Each call then runs the engine and writes every dpsi
! back to iudwf, which drhodv, the restart files and the next irrep's consumers read.
!
! RTSC_STH_TIME=1 prints the wall time of every call (engine and vanilla paths).
! RTSC_STRICT=1 (scored runs, rtsc/bench/run-fork.sh): with RTSC_STH=1, a call that does not run on
! the engine (unsupported case or engine declined) stops ph.x instead of taking the vanilla loop.
!
! sth_batched_end_q (do_phonon, after each q) releases the per-q engine state and the device blocks
! the NVHPC runtime keeps after DEALLOCATE, and prints one line with the q's device-memory high-water.
!
MODULE sth_batched
  USE kinds, ONLY : DP
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: sth_batched_kernel, sth_env_flag, sth_batched_flush, sth_batched_becs, sth_batched_flush_end, &
            sth_batched_end_q, sth_strict_check, sth_batched_cold
  !
  LOGICAL, SAVE :: have_state = .FALSE.
  LOGICAL, SAVE :: pending = .FALSE.        ! engine dpsi not yet written to iudwf
  LOGICAL, SAVE :: eng_last = .FALSE.       ! the last sternheimer_kernel call ran on the engine
  INTEGER, SAVE :: st_npert = -1, st_iudvpsi = -1, st_nksq = -1
  COMPLEX(DP), SAVE :: st_key = (0.d0, 0.d0)
  REAL(DP), SAVE :: st_xq(3) = 1.d10        ! q of the per-q engine state
  LOGICAL, SAVE :: cold = .FALSE.           ! not first_iter, but dpsi starts from 0 (sth_batched_cold)
  !
CONTAINS
  !
  LOGICAL FUNCTION sth_env_flag(name)
    CHARACTER(LEN=*), INTENT(IN) :: name
    CHARACTER(LEN=16) :: s
    INTEGER :: ios
    sth_env_flag = .FALSE.
    CALL get_environment_variable(name, s, STATUS=ios)
    IF (ios == 0) sth_env_flag = (TRIM(s) == '1')
  END FUNCTION sth_env_flag
  !
  !--------------------------------------------------------------------------------------------
  SUBROUTINE sth_batched_kernel(first_iter, time_reversed, npert, lrdvpsi, iudvpsi, thresh, &
                                dvscfins, all_conv, avg_iter, drhoout, done)
    !--------------------------------------------------------------------------------------------
    !! One sternheimer_kernel call on the batched engine. done = .FALSE. on return means the case
    !! is not supported (or the engine declined) and the caller must run the vanilla loop.
    USE io_global,        ONLY : stdout
    USE mp,               ONLY : mp_sum
    USE mp_pools,         ONLY : inter_pool_comm
    USE mp_bands,         ONLY : nproc_bgrp, nbgrp
    USE fft_base,         ONLY : dffts
    USE buffers,          ONLY : get_buffer, save_buffer
    USE klist,            ONLY : lgauss, ltetra
    USE noncollin_module, ONLY : noncolin, npol, nspin_mag
    USE lsda_mod,         ONLY : lsda
    USE uspp,             ONLY : okvan
    USE ldaU,             ONLY : lda_plus_u
    USE control_lr,       ONLY : lgamma
    USE control_flags,    ONLY : gamma_only, scissor
    USE realus,           ONLY : real_space
    USE bp,               ONLY : lelfield
    USE xc_lib,           ONLY : exx_is_active, xclib_dft_is
    USE two_chem,         ONLY : twochem
    USE units_lr,         ONLY : lrdwf, iudwf
    USE qpoint,           ONLY : nksq
    USE wvfct,            ONLY : nbnd, npwx
    USE eqv,              ONLY : dpsi, dvpsi
#if defined(__CUDA)
    USE sth_engine,       ONLY : sth_engine_solve, sth_engine_get_dpsi, sth_engine_ready, sth_engine_qready
    USE qpoint,           ONLY : xq
#endif
    !
    LOGICAL, INTENT(IN) :: first_iter, time_reversed
    INTEGER, INTENT(IN) :: npert, lrdvpsi, iudvpsi
    REAL(DP), INTENT(IN) :: thresh
    COMPLEX(DP), INTENT(IN) :: dvscfins(dffts%nnr, nspin_mag, npert)
    LOGICAL, INTENT(OUT) :: all_conv
    REAL(DP), INTENT(OUT) :: avg_iter
    COMPLEX(DP), INTENT(INOUT) :: drhoout(dffts%nnr, nspin_mag, npert)
    LOGICAL, INTENT(OUT) :: done
    !
#if defined(__CUDA)
    INTEGER :: tot_num_iter, tot_cg_calls, nfail, ik, ipert, nrec, ierr
    COMPLEX(DP) :: key
    COMPLEX(DP), ALLOCATABLE :: dv(:,:), dr(:,:)
    LOGICAL :: reinit
#endif
    !
    done = .FALSE.
    eng_last = .FALSE.
#if defined(__CUDA)
    IF (.NOT. sth_env_flag('RTSC_STH')) RETURN
    ! supported case (STH-ENGINE.md, v1)
    IF (time_reversed .OR. okvan .OR. noncolin .OR. lsda .OR. nspin_mag /= 1 .OR. npol /= 1) RETURN
    ! q = Gamma (lgamma, evq => evc): k+q = k, so the engine gets the same psi twice; the Fermi
    ! shift of metals (ef_shift_new on drho, ef_shift_wfc_new on iudwf after the flush) acts
    ! outside this kernel, unchanged
    IF (lda_plus_u .OR. gamma_only .OR. .NOT. lgauss .OR. ltetra .OR. twochem) RETURN
    IF (xclib_dft_is('meta') .OR. exx_is_active() .OR. real_space .OR. scissor .OR. lelfield) RETURN
    IF (nproc_bgrp /= 1 .OR. nbgrp /= 1 .OR. dffts%has_task_groups) RETURN
    IF (dffts%nnr /= dffts%nr1x * dffts%nr2x * dffts%nr3x) RETURN
    IF (nksq < 1 .OR. npert > 6) RETURN
    !
    ! engine state belongs to this irrep? (re-initialised at every first iteration; otherwise the
    ! dV_bare psi of (k=1, pert=1) identifies the irrep, e.g. after a restart)
    CALL get_buffer(dvpsi, lrdvpsi, iudvpsi, 1)
    key = SUM(dvpsi(:, 1)) + (0.d0, 1.d0) * SUM(ABS(dvpsi(:, 1)))
    reinit = first_iter .OR. .NOT. have_state .OR. .NOT. sth_engine_ready() .OR. npert /= st_npert &
             .OR. iudvpsi /= st_iudvpsi .OR. nksq /= st_nksq .OR. key /= st_key
    IF (reinit) THEN
      ! the previous irrep's dpsi: written now unless no later step reads it (sth_batched_flush_end)
      IF (pending .AND. flush_needed()) CALL sth_batched_flush()
      pending = .FALSE.
      have_state = .FALSE.
      ! per-q state (k, k+q data, vkb, h_diag, psi(r)) is kept across the irreps of one q
      IF (.NOT. sth_engine_qready() .OR. ANY(xq /= st_xq) .OR. nksq /= st_nksq) THEN
        CALL sth_batched_init(first_iter, npert, lrdvpsi, iudvpsi, ierr)
        st_xq = xq
      ELSE
        CALL sth_batched_irrep(first_iter, npert, lrdvpsi, iudvpsi, ierr)
      ENDIF
      IF (ierr /= 0) THEN
        WRITE(stdout, '(5x,"rtsc sth: engine declined (code ",I0,"), vanilla path")') ierr
        IF (sth_env_flag('RTSC_STRICT')) CALL errore('sth_batched_kernel', &
             'engine declined (see the sth_engine line above) and RTSC_STRICT=1', ABS(ierr))
        RETURN
      ENDIF
      have_state = .TRUE.
      st_npert = npert; st_iudvpsi = iudvpsi; st_nksq = nksq; st_key = key
    ENDIF
    !
    ALLOCATE(dv(dffts%nnr, npert), dr(dffts%nnr, npert))
    dv(:, :) = dvscfins(:, 1, :)
    dr(:, :) = drhoout(:, 1, :)
    CALL sth_engine_solve(first_iter, thresh, dv, dr, tot_num_iter, tot_cg_calls, nfail)
    drhoout(:, 1, :) = dr(:, :)
    DEALLOCATE(dv, dr)
    !
    ! dpsi goes back to iudwf once per irrep (sth_batched_flush, end of dfpt_kernel) or before the
    ! engine state is replaced; nothing reads iudwf between the calls of one irrep on this path
    pending = .TRUE.
    !
    all_conv = (nfail == 0)
    IF (.NOT. all_conv) WRITE(stdout, '(5x,"rtsc sth: ",I0," (k,pert) solves not converged, thresh ",ES10.3)') &
         nfail, thresh
    CALL mp_sum(tot_num_iter, inter_pool_comm)
    CALL mp_sum(tot_cg_calls, inter_pool_comm)
    avg_iter = REAL(tot_num_iter, DP) / REAL(tot_cg_calls, DP)
    done = .TRUE.
    eng_last = .TRUE.
#endif
  END SUBROUTINE sth_batched_kernel
  !
  !--------------------------------------------------------------------------------------------
  SUBROUTINE sth_batched_cold(flag)
    !--------------------------------------------------------------------------------------------
    !! flag = .TRUE.: the following calls get dV_ind (first_iter = .FALSE.) but solve from dpsi = 0
    !! instead of the iudwf warm start (one-shot pass from an interpolated dvscf, PH/rtsc_dq.f90)
#if defined(__CUDA)
    USE sth_engine,       ONLY : sth_cold_start
#endif
    LOGICAL, INTENT(IN) :: flag
    cold = flag
#if defined(__CUDA)
    sth_cold_start = flag
#endif
  END SUBROUTINE sth_batched_cold
  !
  !--------------------------------------------------------------------------------------------
  SUBROUTINE sth_strict_check(what, done)
    !--------------------------------------------------------------------------------------------
    !! RTSC_STRICT=1 with RTSC_STH=1: a step that did not run on the engine is an error
    CHARACTER(LEN=*), INTENT(IN) :: what
    LOGICAL, INTENT(IN) :: done
    IF (done) RETURN
    IF (.NOT. (sth_env_flag('RTSC_STH') .AND. sth_env_flag('RTSC_STRICT'))) RETURN
    CALL errore(what, 'RTSC_STH=1 but this call is not supported by the engine and RTSC_STRICT=1', 1)
  END SUBROUTINE sth_strict_check
  !
  !--------------------------------------------------------------------------------------------
  SUBROUTINE sth_batched_end_q(iq)
    !--------------------------------------------------------------------------------------------
    !! end of q-point iq (do_phonon): free the per-q engine state (it was kept until the next q's
    !! first call, i.e. through that q's nscf), return the runtime-cached device blocks to the driver,
    !! and print the device-memory high-water of this q (nscf, engine, GPU tail)
    USE io_global,        ONLY : stdout
    USE qpoint,           ONLY : nksq
#if defined(__CUDA)
    USE sth_engine,       ONLY : sth_engine_free, sth_mem_take_hw, sth_mem_sample
#endif
    INTEGER, INTENT(IN) :: iq
#if defined(__CUDA)
    INTEGER(8) :: hw, tot, after
    REAL(DP), PARAMETER :: gib = 1024.d0**3
    IF (.NOT. sth_env_flag('RTSC_STH')) RETURN
    IF (pending .AND. flush_needed()) CALL sth_batched_flush()
    pending = .FALSE.
    have_state = .FALSE.
    eng_last = .FALSE.
    st_xq = 1.d10
    st_nksq = -1
    CALL sth_mem_take_hw(hw, tot)
    CALL sth_engine_free()            ! also returns the runtime-cached blocks (sth_mem_trim)
    CALL sth_mem_sample(after)        ! the next q's high-water starts from this baseline
    WRITE(stdout, '(5x,"rtsc mem: q ",I0,", nksq ",I0,": device high-water ",F6.2," GiB, after release ", &
         &F6.2," GiB (of ",F6.2,")")') iq, nksq, REAL(hw, DP) / gib, REAL(after, DP) / gib, REAL(tot, DP) / gib
#endif
  END SUBROUTINE sth_batched_end_q
  !
  !--------------------------------------------------------------------------------------------
  LOGICAL FUNCTION flush_needed()
    !--------------------------------------------------------------------------------------------
    !! Does any later step read this irrep's dpsi from iudwf? Yes for q = Gamma (ef_shift_wfc_new),
    !! for buffers on files (no reduce_io: restart), and when drhodv does not take the engine path
    !! (RTSC_TAIL=0). Otherwise drhodv reads the engine's device dpsi (sth_batched_becs) and nothing
    !! else reads iudwf; drhodv still flushes first if it falls back to the vanilla path.
    USE control_lr,       ONLY : lgamma, reduce_io
    USE sth_tail,         ONLY : sth_tail_on
    flush_needed = lgamma .OR. .NOT. reduce_io .OR. .NOT. sth_tail_on()
  END FUNCTION flush_needed
  !
  !--------------------------------------------------------------------------------------------
  SUBROUTINE sth_batched_flush_end()
    !--------------------------------------------------------------------------------------------
    !! end of dfpt_kernel: write the engine's dpsi to iudwf if a later step reads it there
    IF (pending .AND. flush_needed()) CALL sth_batched_flush()
  END SUBROUTINE sth_batched_flush_end
  !
  !--------------------------------------------------------------------------------------------
  SUBROUTINE sth_batched_flush()
    !--------------------------------------------------------------------------------------------
    !! write the engine's dpsi of the current irrep to iudwf (drhodv, restart, vanilla warm start)
    USE buffers,          ONLY : save_buffer
    USE units_lr,         ONLY : lrdwf, iudwf
    USE eqv,              ONLY : dpsi
#if defined(__CUDA)
    USE sth_engine,       ONLY : sth_engine_get_dpsi
#endif
    INTEGER :: ik, ipert, nrec
    IF (.NOT. pending) RETURN
#if defined(__CUDA)
    CALL start_clock('sth_flush')
    DO ik = 1, st_nksq
      DO ipert = 1, st_npert
        nrec = (ipert - 1) * st_nksq + ik
        CALL sth_engine_get_dpsi(ik, ipert, dpsi)
        CALL save_buffer(dpsi, lrdwf, iudwf, nrec)
      ENDDO
    ENDDO
    CALL stop_clock('sth_flush')
#endif
    pending = .FALSE.
  END SUBROUTINE sth_batched_flush
  !
  !--------------------------------------------------------------------------------------------
  SUBROUTINE sth_batched_becs(nper, becq, alpq, ok)
    !--------------------------------------------------------------------------------------------
    !! drhodv on the engine: the projections of the current irrep's dpsi for all k at once,
    !! becq(:,:,mu,ik) = vkb^H dpsi and alpq(:,:,ipol,mu,ik) = vkb^H [(k+q+G)_ipol dpsi], from the
    !! engine's device dpsi (identical to what the flush wrote to iudwf). ok = .FALSE.: the last
    !! sternheimer_kernel call was not the engine's, or its state does not match; use the vanilla path.
    USE klist,            ONLY : xk, ngk, igk_k
    USE gvect,            ONLY : g
    USE wvfct,            ONLY : nbnd, npwx
    USE uspp,             ONLY : nkb
    USE qpoint,           ONLY : nksq, ikqs
#if defined(__CUDA)
    USE sth_engine,       ONLY : sth_engine_becs, sth_engine_ready
#endif
    INTEGER, INTENT(IN) :: nper
    COMPLEX(DP), INTENT(OUT) :: becq(nkb, nbnd, nper, nksq), alpq(nkb, nbnd, 3, nper, nksq)
    LOGICAL, INTENT(OUT) :: ok
#if defined(__CUDA)
    REAL(DP), ALLOCATABLE :: kqg(:,:,:)
    INTEGER :: ik, ikq, ig
#endif
    ok = .FALSE.
#if defined(__CUDA)
    IF (.NOT. (eng_last .AND. have_state .AND. sth_engine_ready())) RETURN
    IF (nper /= st_npert .OR. nksq /= st_nksq) RETURN
    CALL start_clock('sth_becs')
    ALLOCATE(kqg(npwx, 3, nksq))
    kqg = 0.d0
    DO ik = 1, nksq
      ikq = ikqs(ik)
      DO ig = 1, ngk(ikq)
        kqg(ig, :, ik) = xk(:, ikq) + g(:, igk_k(ig, ikq))
      ENDDO
    ENDDO
    CALL sth_engine_becs(kqg, becq, alpq)
    DEALLOCATE(kqg)
    CALL stop_clock('sth_becs')
    ok = .TRUE.
#endif
  END SUBROUTINE sth_batched_becs
  !
#if defined(__CUDA)
  !--------------------------------------------------------------------------------------------
  SUBROUTINE sth_batched_irrep(first_iter, npert, lrdvpsi, iudvpsi, ierr)
    !--------------------------------------------------------------------------------------------
    !! new irrep on the per-q engine state: only dV_bare psi (and a warm start) change
    USE klist,            ONLY : ngk
    USE wvfct,            ONLY : npwx
    USE buffers,          ONLY : get_buffer
    USE units_lr,         ONLY : lrdwf, iudwf
    USE control_lr,       ONLY : nbnd_occ
    USE qpoint,           ONLY : nksq, ikks, ikqs
    USE eqv,              ONLY : dpsi, dvpsi
    USE sth_engine,       ONLY : sth_engine_irrep, sth_engine_set_dvbare, sth_engine_set_dpsi, &
                                 sth_engine_irrep_commit
    LOGICAL, INTENT(IN) :: first_iter
    INTEGER, INTENT(IN) :: npert, lrdvpsi, iudvpsi
    INTEGER, INTENT(OUT) :: ierr
    INTEGER :: ik, ipert, nrec, npwq, nock
    CALL start_clock('sth_irrep')
    CALL sth_engine_irrep(npert, ierr)
    IF (ierr /= 0) RETURN
    DO ik = 1, nksq
      npwq = ngk(ikqs(ik)); nock = nbnd_occ(ikks(ik))
      DO ipert = 1, npert
        nrec = (ipert - 1) * nksq + ik
        CALL get_buffer(dvpsi, lrdvpsi, iudvpsi, nrec)
        IF (npwq < npwx) THEN
          IF (ANY(dvpsi(npwq + 1:npwx, 1:nock) /= (0.d0, 0.d0))) THEN
            ierr = 10
            RETURN
          ENDIF
        ENDIF
        CALL sth_engine_set_dvbare(ik, ipert, dvpsi)
        IF (.NOT. (first_iter .OR. cold)) THEN
          CALL get_buffer(dpsi, lrdwf, iudwf, nrec)
          CALL sth_engine_set_dpsi(ik, ipert, dpsi)
        ENDIF
      ENDDO
    ENDDO
    CALL sth_engine_irrep_commit(ierr)
    CALL stop_clock('sth_irrep')
  END SUBROUTINE sth_batched_irrep
  !
  !--------------------------------------------------------------------------------------------
  SUBROUTINE sth_batched_init(first_iter, npert, lrdvpsi, iudvpsi, ierr)
    !--------------------------------------------------------------------------------------------
    USE fft_base,         ONLY : dffts
    USE ions_base,        ONLY : nat, ityp
    USE klist,            ONLY : xk, wk, ngk, igk_k, degauss, ngauss
    USE ener,             ONLY : ef
    USE cell_base,        ONLY : omega
    USE wvfct,            ONLY : nbnd, npwx, et, g2kin
    USE wavefunctions,    ONLY : evc
    USE uspp,             ONLY : vkb, nkb, deeq, ofsbeta
    USE uspp_param,       ONLY : nh
    USE uspp_init,        ONLY : init_us_2
    USE scf,              ONLY : vrs
    USE buffers,          ONLY : get_buffer
    USE units_lr,         ONLY : iuwfc, lrwfc, lrdwf, iudwf
    USE control_lr,       ONLY : nbnd_occ, alpha_pv, lgamma
    USE qpoint,           ONLY : nksq, ikks, ikqs
    USE eqv,              ONLY : dpsi, dvpsi, evq
    USE sth_engine,       ONLY : sth_engine_init, sth_engine_set_global, sth_engine_set_k, &
                                 sth_engine_set_dvbare, sth_engine_set_dpsi, sth_engine_setup
    !
    LOGICAL, INTENT(IN) :: first_iter
    INTEGER, INTENT(IN) :: npert, lrdvpsi, iudvpsi
    INTEGER, INTENT(OUT) :: ierr
    !
    INTEGER :: ik, ikk, ikq, npw, npwq, nock, nockq, ipert, nrec, na, ih, jh, ibnd, jbnd, nslot
    INTEGER, ALLOCATABLE :: idxk(:), idxkq(:)
    REAL(DP), ALLOCATABLE :: wwg(:,:), wg1(:)
    COMPLEX(DP), ALLOCATABLE :: dmat(:,:)
    REAL(DP) :: w0g, wgp, deltae, theta
    REAL(DP), EXTERNAL :: wgauss, w0gauss
    !
    CALL start_clock('sth_init')
    nslot = 0
    DO ik = 1, nksq
      nslot = MAX(nslot, nbnd_occ(ikks(ik)))
    ENDDO
    CALL sth_engine_init(nksq, npwx, nbnd, nkb, npert, nslot, dffts%nr1, dffts%nr2, dffts%nr3, &
                         dffts%nr1x, dffts%nr2x, dffts%nr3x, dffts%nnr, alpha_pv, omega, ierr)
    IF (ierr /= 0) RETURN
    !
    ! dense nkb x nkb D (block-diagonal per atom), as add_vuspsi applies deeq
    ALLOCATE(dmat(nkb, nkb))
    dmat = (0.d0, 0.d0)
    DO na = 1, nat
      DO ih = 1, nh(ityp(na))
        DO jh = 1, nh(ityp(na))
          dmat(ofsbeta(na) + ih, ofsbeta(na) + jh) = CMPLX(deeq(ih, jh, na, 1), 0.d0, KIND=DP)
        ENDDO
      ENDDO
    ENDDO
    CALL sth_engine_set_global(vrs(:, 1), dmat)
    DEALLOCATE(dmat)
    !
    ALLOCATE(idxk(npwx), idxkq(npwx), wwg(nbnd, nbnd), wg1(nbnd))
    DO ik = 1, nksq
      ikk = ikks(ik); ikq = ikqs(ik)
      npw = ngk(ikk); npwq = ngk(ikq)
      nock = nbnd_occ(ikk); nockq = nbnd_occ(ikq)
      idxk(1:npw) = dffts%nl(igk_k(1:npw, ikk))
      idxkq(1:npwq) = dffts%nl(igk_k(1:npwq, ikq))
      CALL get_buffer(evc, lrwfc, iuwfc, ikk)
      IF (.NOT. lgamma) CALL get_buffer(evq, lrwfc, iuwfc, ikq)   ! lgamma: evq => evc, ikq = ikk
      CALL init_us_2(npwq, igk_k(1, ikq), xk(1, ikq), vkb, .true.)
      !$acc update host(vkb)
      CALL g2_kin(ikq)
      !
      ! orthogonalize.f90, lgauss branch (twochem excluded)
      DO ibnd = 1, nock
        wg1(ibnd) = wgauss((ef - et(ibnd, ikk)) / degauss, ngauss)
        w0g = w0gauss((ef - et(ibnd, ikk)) / degauss, ngauss) / degauss
        DO jbnd = 1, nbnd
          wgp = wgauss((ef - et(jbnd, ikq)) / degauss, ngauss)
          deltae = et(jbnd, ikq) - et(ibnd, ikk)
          theta = wgauss(deltae / degauss, 0)
          wwg(jbnd, ibnd) = wg1(ibnd) * (1.d0 - theta) + wgp * theta
          IF (jbnd <= nockq) THEN
            IF (ABS(deltae) > 1.0d-5) THEN
              wwg(jbnd, ibnd) = wwg(jbnd, ibnd) + alpha_pv * theta * (wgp - wg1(ibnd)) / deltae
            ELSE
              wwg(jbnd, ibnd) = wwg(jbnd, ibnd) - alpha_pv * theta * w0g
            ENDIF
          ENDIF
        ENDDO
      ENDDO
      !
      CALL sth_engine_set_k(ik, npw, npwq, nock, nockq, wk(ikk), idxk(1:npw), idxkq(1:npwq), evc, evq, &
                            et(:, ikk), g2kin, vkb, wwg(:, 1:nock), wg1(1:nock))
      DO ipert = 1, npert
        nrec = (ipert - 1) * nksq + ik
        CALL get_buffer(dvpsi, lrdvpsi, iudvpsi, nrec)
        ! the engine keeps rows > npwq at zero; vanilla carries dvbare's padding through cgsolve_all
        ! (h = g on those rows), so the two agree only when that padding is zero
        IF (npwq < npwx) THEN
          IF (ANY(dvpsi(npwq + 1:npwx, 1:nock) /= (0.d0, 0.d0))) THEN
            ierr = 10
            RETURN
          ENDIF
        ENDIF
        CALL sth_engine_set_dvbare(ik, ipert, dvpsi)
        IF (.NOT. (first_iter .OR. cold)) THEN
          ! restart in the middle of an irrep: warm start from iudwf, as vanilla
          CALL get_buffer(dpsi, lrdwf, iudwf, nrec)
          CALL sth_engine_set_dpsi(ik, ipert, dpsi)
        ENDIF
      ENDDO
    ENDDO
    DEALLOCATE(idxk, idxkq, wwg, wg1)
    IF (lgamma) THEN
      !$acc update device(evc)
    ELSE
      !$acc update device(evc, evq)
    ENDIF
    CALL sth_engine_setup(ierr)
    CALL stop_clock('sth_init')
  END SUBROUTINE sth_batched_init
#endif
  !
END MODULE sth_batched
