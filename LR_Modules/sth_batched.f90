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
! RTSC_STH_MP=1: the engine's CG runs in mixed precision (FP32 inner PCG, FP64 defect correction
! to the same thresh; rtsc/sth/sth_engine.f90 header). RTSC_MP_ETA / RTSC_MP_SAFE: the inner
! tolerance max(safe*thresh, eta*|r|) (defaults 1e-4, 0.5). With RTSC_STH_TIME=1 each call also
! prints its rounds and FP64 / FP32 column applications.
!
! RTSC_STH_QCACHE=1 (groups/compute/SETUP-LOG.md): less setup work for the same values. Per irrep: no host stage
! of dV_bare psi / dpsi when nothing is staged, the device work arrays kept across irreps of one shape, the k+q+G
! table built once per q on the device (dvb_build, becs). Per q: vkb at k+q copied device-to-device into the
! engine (no update host), evq/evc uploaded per k (no host W stage), d_inv built on the device.
! Class B: dyn/dvscf/EPW outputs bitwise identical on/off (B1, B2 smoke chains; B1 frozen q7; B2 frozen q2).
! Measured, A40, B1 frozen q7: per-q setup 5.45 -> 2.17 s, per-irrep 1.53 -> 0.455 s (what is left is GPU work).
!
! sth_batched_end_q (do_phonon, after each q) releases the per-q engine state and the device blocks
! the NVHPC runtime keeps after DEALLOCATE, and prints one line with the q's device-memory high-water.
!
MODULE sth_batched
  USE kinds, ONLY : DP
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: sth_call_eta
  TYPE bec_type_copy
    COMPLEX(DP), ALLOCATABLE :: k(:,:)
  END TYPE bec_type_copy
  PUBLIC :: sth_batched_kernel, sth_env_flag, sth_batched_flush, sth_batched_becs, sth_batched_flush_end, &
            sth_batched_end_q, sth_strict_check, sth_batched_dvb_ok, sth_batched_dvb_prepare
  !
  ! dV_bare psi built in the engine (RTSC_STH_DVB=1, KERNEL-LOG lever 4). solve_linter (PHonon) decides it
  ! with sth_batched_dvb_ok() and, instead of the per-(k, pert) dvqpsi_us + save_buffer loop, hands over
  ! the irrep's dvloc, patterns and alphap with sth_batched_dvb_prepare (PHonon modules are not visible
  ! here). RTSC_STH_DVB_CHECK=1: solve_linter runs its loop too and the engine's result is compared.
  LOGICAL, SAVE :: dvb_on = .FALSE.
  INTEGER, SAVE :: dvb_tag = 0
  COMPLEX(DP), ALLOCATABLE, SAVE :: dvb_vl(:,:), dvb_u(:,:)
  TYPE(bec_type_copy), ALLOCATABLE, SAVE :: dvb_al(:,:)
  !
  LOGICAL, SAVE :: have_state = .FALSE.
  LOGICAL, SAVE :: pending = .FALSE.        ! engine dpsi not yet written to iudwf
  LOGICAL, SAVE :: eng_last = .FALSE.       ! the last sternheimer_kernel call ran on the engine
  INTEGER, SAVE :: st_npert = -1, st_iudvpsi = -1, st_nksq = -1
  COMPLEX(DP), SAVE :: st_key = (0.d0, 0.d0)
  REAL(DP), SAVE :: st_xq(3) = 1.d10        ! q of the per-q engine state
  INTEGER(8), SAVE :: tot64 = 0, tot32 = 0   ! column applications of this q (FP64 / FP32)
  !
  ! RTSC_STH_XP=m (m = 1..8 history slots; SOLVE-LOG.md): dpsi extrapolation across the SCF calls of
  ! one irrep. Host side: the dV_in increments of the recorded calls, to fit the next increment.
  INTEGER, SAVE :: xp_m = 0, xp_n = 0, xp_next = 0
  INTEGER, SAVE :: xp_slot_of(8) = 0                 ! history position i -> engine slot
  COMPLEX(DP), ALLOCATABLE, SAVE :: xp_dv(:,:)       ! (nnr*npert, m) dV_in increments, by engine slot
  COMPLEX(DP), ALLOCATABLE, SAVE :: xp_prev(:)       ! dV_in of the previous call
  REAL(DP), SAVE :: xp_rho = 0.d0                    ! last fit: |residual| / |increment|
  COMPLEX(DP), SAVE :: xp_cprint(8) = (0.d0, 0.d0)
  INTEGER, SAVE :: xp_ncprint = 0
  ! relative stopping factor for the next call (set by dfpt_kernel: RTSC_STH_ETA on non-final iterations)
  REAL(DP), SAVE :: sth_call_eta = 0.d0
  REAL(DP), SAVE :: t_kqg = 0.d0                     ! RTSC_STH_TIME: host k+q+G table of the last dvb_build
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
  LOGICAL FUNCTION sth_case_ok(time_reversed, npert)
    !! the engine's supported case (STH-ENGINE.md, v1): sth_batched_kernel runs the engine only if true
    USE fft_base,         ONLY : dffts
    USE mp_bands,         ONLY : nproc_bgrp, nbgrp
    USE klist,            ONLY : lgauss, ltetra
    USE noncollin_module, ONLY : noncolin, npol, nspin_mag
    USE lsda_mod,         ONLY : lsda
    USE uspp,             ONLY : okvan
    USE ldaU,             ONLY : lda_plus_u
    USE control_flags,    ONLY : gamma_only, scissor
    USE realus,           ONLY : real_space
    USE bp,               ONLY : lelfield
    USE xc_lib,           ONLY : exx_is_active, xclib_dft_is
    USE two_chem,         ONLY : twochem
    USE qpoint,           ONLY : nksq
    LOGICAL, INTENT(IN) :: time_reversed
    INTEGER, INTENT(IN) :: npert
    sth_case_ok = .FALSE.
    IF (time_reversed .OR. okvan .OR. noncolin .OR. lsda .OR. nspin_mag /= 1 .OR. npol /= 1) RETURN
    ! q = Gamma (lgamma, evq => evc): k+q = k, so the engine gets the same psi twice; the Fermi
    ! shift of metals (ef_shift_new on drho, ef_shift_wfc_new on iudwf after the flush) acts
    ! outside this kernel, unchanged
    IF (lda_plus_u .OR. gamma_only .OR. .NOT. lgauss .OR. ltetra .OR. twochem) RETURN
    IF (xclib_dft_is('meta') .OR. exx_is_active() .OR. real_space .OR. scissor .OR. lelfield) RETURN
    IF (nproc_bgrp /= 1 .OR. nbgrp /= 1 .OR. dffts%has_task_groups) RETURN
    IF (dffts%nnr /= dffts%nr1x * dffts%nr2x * dffts%nr3x) RETURN
    IF (nksq < 1 .OR. npert > 6) RETURN
    sth_case_ok = .TRUE.
  END FUNCTION sth_case_ok
  !
  LOGICAL FUNCTION sth_batched_dvb_ok(time_reversed, npert)
    !! may solve_linter skip its dV_bare psi loop? Exactly the engine's supported case (sth_case_ok, so
    !! sth_batched_kernel cannot fall back to vanilla on an unwritten buffer), plus no NLCC
    USE uspp,             ONLY : nlcc_any
    LOGICAL, INTENT(IN) :: time_reversed
    INTEGER, INTENT(IN) :: npert
    sth_batched_dvb_ok = .FALSE.
#if defined(__CUDA)
    IF (.NOT. (sth_env_flag('RTSC_STH') .AND. sth_env_flag('RTSC_STH_DVB'))) RETURN
    IF (nlcc_any .OR. .NOT. sth_case_ok(time_reversed, npert)) RETURN
    sth_batched_dvb_ok = .TRUE.
#endif
  END FUNCTION sth_batched_dvb_ok
  !
  SUBROUTINE sth_batched_dvb_prepare(on, npe, imode0, dvl, uq, alphap)
    !! per irrep, from solve_linter: on = engine builds dV_bare psi; dvl(nnr, npe) = compute_dvloc of each
    !! perturbation; uq(3 nat, npe) its pattern; alphap(3, nksq) (phus) for the per-q engine state
    USE becmod,  ONLY : bec_type
    USE qpoint,  ONLY : nksq
    USE fft_base, ONLY : dffts
    LOGICAL, INTENT(IN) :: on
    INTEGER, INTENT(IN) :: npe, imode0
    COMPLEX(DP), INTENT(IN) :: dvl(dffts%nnr, npe), uq(:, :)
    TYPE(bec_type), INTENT(IN) :: alphap(3, nksq)
    INTEGER :: ik, a
    dvb_on = on
    IF (.NOT. on) RETURN
    dvb_tag = imode0 + 1
    IF (ALLOCATED(dvb_vl)) DEALLOCATE(dvb_vl, dvb_u)
    ALLOCATE(dvb_vl(dffts%nnr, npe), dvb_u(SIZE(uq, 1), npe))
    dvb_vl = dvl
    dvb_u = uq(:, 1:npe)
    IF (ALLOCATED(dvb_al)) DEALLOCATE(dvb_al)
    ALLOCATE(dvb_al(3, nksq))
    DO ik = 1, nksq
      DO a = 1, 3
        ALLOCATE(dvb_al(a, ik)%k, SOURCE = alphap(a, ik)%k)
      ENDDO
    ENDDO
  END SUBROUTINE sth_batched_dvb_prepare
  !
  SUBROUTINE env_real(name, val)
    !! val = the environment variable's value if it is set and reads as a number; unchanged otherwise
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL(DP), INTENT(INOUT) :: val
    CHARACTER(LEN=32) :: s
    INTEGER :: ios
    REAL(DP) :: v
    CALL get_environment_variable(name, s, STATUS=ios)
    IF (ios /= 0 .OR. LEN_TRIM(s) == 0) RETURN
    READ(s, *, IOSTAT=ios) v
    IF (ios == 0) val = v
  END SUBROUTINE env_real
  !
  INTEGER FUNCTION sth_env_int(name)
    CHARACTER(LEN=*), INTENT(IN) :: name
    CHARACTER(LEN=16) :: s
    INTEGER :: ios, v
    sth_env_int = 0
    CALL get_environment_variable(name, s, STATUS=ios)
    IF (ios /= 0) RETURN
    READ(s, *, IOSTAT=ios) v
    IF (ios == 0) sth_env_int = v
  END FUNCTION sth_env_int
  !
#if defined(__CUDA)
  !--------------------------------------------------------------------------------------------
  SUBROUTINE xp_prepare(first_iter, reinit, npert, dvscfins)
    !--------------------------------------------------------------------------------------------
    !! RTSC_STH_XP: before an engine call, fit this call's dV_in increment by the recorded ones
    !! (least squares, complex coefficients) and hand the coefficients to the engine, which starts
    !! CG from dpsi(k-1) + sum_j c_j [dpsi(j) - dpsi(j-1)]. Records this call's increment.
    USE fft_base,         ONLY : dffts
    USE noncollin_module, ONLY : nspin_mag
    USE sth_engine,       ONLY : sth_engine_xp_set
    LOGICAL, INTENT(IN) :: first_iter, reinit
    INTEGER, INTENT(IN) :: npert
    COMPLEX(DP), INTENT(IN) :: dvscfins(dffts%nnr, nspin_mag, npert)
    COMPLEX(DP), ALLOCATABLE :: d(:)
    COMPLEX(DP) :: g(8, 8), b(8), c(8)
    INTEGER :: n, i, j, rec, nc, info, ipiv(8), slots(8)
    REAL(DP) :: dn, rn, gmax
    COMPLEX(DP), EXTERNAL :: zdotc
    n = dffts%nnr * npert
    xp_ncprint = 0
    xp_rho = -1.d0
    IF (first_iter .OR. reinit) THEN
      ! dpsi of this call has no recorded predecessor with known dV_in (first iteration: dV_in = 0)
      CALL sth_engine_xp_set(0, slots, c, 0)
      RETURN
    ENDIF
    ALLOCATE(d(n))
    d = RESHAPE(dvscfins(:, 1, :), (/ n /)) - xp_prev
    nc = 0
    IF (xp_n > 0) THEN
      DO i = 1, xp_n
        slots(i) = xp_slot_of(i)
        DO j = i, xp_n
          g(i, j) = zdotc(n, xp_dv(1, xp_slot_of(i)), 1, xp_dv(1, xp_slot_of(j)), 1)
          g(j, i) = CONJG(g(i, j))
        ENDDO
        b(i) = zdotc(n, xp_dv(1, xp_slot_of(i)), 1, d, 1)
      ENDDO
      gmax = 0.d0
      DO i = 1, xp_n
        gmax = MAX(gmax, REAL(g(i, i), DP))
      ENDDO
      DO i = 1, xp_n
        g(i, i) = g(i, i) + 1.d-12 * gmax
      ENDDO
      c(1:xp_n) = b(1:xp_n)
      CALL zgesv(xp_n, 1, g, 8, ipiv, c, 8, info)
      IF (info == 0) THEN
        ! residual of the fit
        dn = REAL(zdotc(n, d, 1, d, 1), DP)
        BLOCK
          COMPLEX(DP), ALLOCATABLE :: r(:)
          ALLOCATE(r(n))
          r = d
          DO i = 1, xp_n
            CALL zaxpy(n, -c(i), xp_dv(1, xp_slot_of(i)), 1, r, 1)
          ENDDO
          rn = REAL(zdotc(n, r, 1, r, 1), DP)
          DEALLOCATE(r)
        END BLOCK
        xp_rho = SQRT(rn / MAX(dn, TINY(1.d0)))
        ! use the extrapolation only if the fit explains a meaningful part of the increment
        IF (xp_rho < 0.95d0) nc = xp_n
      ENDIF
    ENDIF
    xp_ncprint = nc
    IF (nc > 0) xp_cprint(1:nc) = c(1:nc)
    ! record this call: ring of xp_m engine slots
    rec = MOD(xp_next, xp_m) + 1
    xp_next = xp_next + 1
    CALL sth_engine_xp_set(nc, slots, c, rec)
    xp_dv(:, rec) = d
    xp_prev = RESHAPE(dvscfins(:, 1, :), (/ n /))
    ! history positions: newest last
    IF (xp_n < xp_m) THEN
      xp_n = xp_n + 1
      xp_slot_of(xp_n) = rec
    ELSE
      xp_slot_of(1:xp_m - 1) = xp_slot_of(2:xp_m)
      xp_slot_of(xp_m) = rec
    ENDIF
    DEALLOCATE(d)
  END SUBROUTINE xp_prepare
#endif
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
    USE sth_engine,       ONLY : sth_engine_solve, sth_engine_get_dpsi, sth_engine_ready, sth_engine_qready, &
                                 sth_n64, sth_n32, sth_nround, sth_t_cg64, &
                                 sth_xp_m, sth_engine_xp_set, sth_eta, sth_defl, sth_timing, sth_prof, &
                                 sth_t_dpot, sth_t_ortho, sth_t_cg, sth_t_drho, &
                                 sth_nsteps, sth_ncolapp, sth_nvalid, sth_nact0, sth_r0hist, sth_r0med, &
                                 sth_t_gemm, sth_t_fft, sth_stband, sth_r0band, sth_gapst, sth_gapdec, sth_gapn
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
    INTEGER :: tot_num_iter, tot_cg_calls, nfail, ik, ipert, nrec, ierr, i
    COMPLEX(DP) :: key
    COMPLEX(DP), ALLOCATABLE :: dv(:,:), dr(:,:)
    LOGICAL :: reinit
    REAL(DP) :: tscale
#endif
    !
    done = .FALSE.
    eng_last = .FALSE.
#if defined(__CUDA)
    IF (.NOT. sth_env_flag('RTSC_STH')) RETURN
    ! supported case (STH-ENGINE.md, v1); the same test decides sth_batched_dvb_ok
    IF (.NOT. sth_case_ok(time_reversed, npert)) RETURN
    !
    ! engine state belongs to this irrep? (re-initialised at every first iteration; otherwise the
    ! dV_bare psi of (k=1, pert=1) identifies the irrep, e.g. after a restart)
    xp_m = sth_env_int('RTSC_STH_XP')
    xp_m = MAX(0, MIN(8, xp_m))
    sth_xp_m = xp_m
    IF (dvb_on) THEN
      key = CMPLX(REAL(dvb_tag, DP), REAL(npert, DP), KIND=DP)   ! no dV_bare buffer on this path
    ELSE
      CALL get_buffer(dvpsi, lrdvpsi, iudvpsi, 1)
      key = SUM(dvpsi(:, 1)) + (0.d0, 1.d0) * SUM(ABS(dvpsi(:, 1)))
    ENDIF
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
        IF (dvb_on) CALL errore('sth_batched_kernel', 'engine declined with RTSC_STH_DVB=1 (no dV_bare buffer)', &
             ABS(ierr))
        IF (sth_env_flag('RTSC_STRICT')) CALL errore('sth_batched_kernel', &
             'engine declined (see the sth_engine line above) and RTSC_STRICT=1', ABS(ierr))
        RETURN
      ENDIF
      have_state = .TRUE.
      st_npert = npert; st_iudvpsi = iudvpsi; st_nksq = nksq; st_key = key
      ! new irrep: empty extrapolation history. In the first iteration dV_in = 0 (ignored by the solve).
      xp_n = 0
      xp_next = 0
      IF (xp_m > 0) THEN
        IF (ALLOCATED(xp_dv)) DEALLOCATE(xp_dv, xp_prev)
        ALLOCATE(xp_dv(dffts%nnr * npert, xp_m), xp_prev(dffts%nnr * npert))
        xp_prev = (0.d0, 0.d0)
        IF (.NOT. first_iter) xp_prev = RESHAPE(dvscfins(:, 1, :), (/ dffts%nnr * npert /))
      ENDIF
    ENDIF
    IF (xp_m > 0) CALL xp_prepare(first_iter, reinit, npert, dvscfins)
    !
    ALLOCATE(dv(dffts%nnr, npert), dr(dffts%nnr, npert))
    dv(:, :) = dvscfins(:, 1, :)
    dr(:, :) = drhoout(:, 1, :)
    sth_timing = sth_env_flag('RTSC_STH_TIME')
    sth_prof = sth_env_flag('RTSC_STH_PROF')
    sth_eta = sth_call_eta
    sth_defl = 0
    IF (sth_env_flag('RTSC_STH_DEFL')) sth_defl = 1
    ! RTSC_STH_THRESH_SCALE (diagnostic only, changes the answer): the engine solves to scale*thresh
    tscale = 1.d0
    CALL env_real('RTSC_STH_THRESH_SCALE', tscale)
    CALL sth_engine_solve(first_iter, tscale * thresh, dv, dr, tot_num_iter, tot_cg_calls, nfail)
    drhoout(:, 1, :) = dr(:, :)
    DEALLOCATE(dv, dr)
    tot64 = tot64 + sth_n64; tot32 = tot32 + sth_n32
    IF (sth_env_flag('RTSC_STH_TIME')) WRITE(stdout, '(5x,"rtsc sth: thresh ",ES9.2,", rounds ",I0,", steps ",I0, &
         &", column applies FP64 ",I0,", FP32 ",I0,"; s: dpot ",F6.3," ortho ",F6.3," cg ",F6.3," (FP64 ",F6.3, &
         &") drho ",F6.3)') thresh, sth_nround, sth_nsteps, sth_n64, sth_n32, sth_t_dpot, sth_t_ortho, sth_t_cg, &
         sth_t_cg64, sth_t_drho
    ! rtsc solve profile (SOLVE-LOG.md): one line per call
    IF (sth_timing) THEN
      WRITE(stdout, '(5x,"rtsc solve: thr ",ES9.2," ncol ",I0," act0 ",I0," steps ",I0," colapp ",I0, &
           &" r0med ",ES9.2," r0hist ",12(1x,I0))') thresh, sth_nvalid, sth_nact0, sth_nsteps, sth_ncolapp, &
           sth_r0med, sth_r0hist
      WRITE(stdout, '(5x,"rtsc solve bands: tests per column by nock-n = 0..5,6+:",7(1x,F6.2))') sth_stband
      WRITE(stdout, '(5x,"rtsc solve bands: log10(r0/thr) by nock-n = 0..5,6+:   ",7(1x,F6.2))') sth_r0band
      WRITE(stdout, '(5x,"rtsc solve gaps (<.02 .02-.1 .1-.5 >.5 none Ry): n",5(1x,I0)," tests",5(1x,F6.2), &
           &" dec",5(1x,F5.2))') NINT(sth_gapn), sth_gapst, sth_gapdec
      IF (xp_m > 0) WRITE(stdout, '(5x,"rtsc xp: fit residual ",F7.4," coef",8(1x,F8.4))') xp_rho, &
           (ABS(xp_cprint(i)), i = 1, xp_ncprint)
      WRITE(stdout, '(5x,"rtsc solve t: dpot ",F8.3," ortho ",F8.3," cg ",F8.3," drho ",F8.3, &
           &" gemm ",F8.3," fft ",F8.3)') sth_t_dpot, sth_t_ortho, sth_t_cg, sth_t_drho, sth_t_gemm, sth_t_fft
    ENDIF
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
    WRITE(stdout, '(5x,"rtsc sth: q ",I0,": column applies FP64 ",I0,", FP32 ",I0)') iq, tot64, tot32
    tot64 = 0; tot32 = 0
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
    CALL kqg_table(kqg)
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
    USE io_global,        ONLY : stdout
    USE sth_engine,       ONLY : sth_engine_irrep, sth_engine_set_dvbare, sth_engine_set_dpsi, &
                                 sth_engine_irrep_commit, sth_timing, sth_t_commit, sth_t_dvb
    LOGICAL, INTENT(IN) :: first_iter
    INTEGER, INTENT(IN) :: npert, lrdvpsi, iudvpsi
    INTEGER, INTENT(OUT) :: ierr
    INTEGER :: ik, ipert, nrec, npwq, nock
    INTEGER(8) :: c0, c1, c2, c3, crate
    CALL start_clock('sth_irrep')
    CALL SYSTEM_CLOCK(c0, crate)
    CALL sth_engine_irrep(npert, ierr)
    IF (ierr /= 0) RETURN
    CALL SYSTEM_CLOCK(c1)
    DO ik = 1, nksq
      npwq = ngk(ikqs(ik)); nock = nbnd_occ(ikks(ik))
      DO ipert = 1, npert
        nrec = (ipert - 1) * nksq + ik
        IF (.NOT. dvb_on) THEN
          CALL get_buffer(dvpsi, lrdvpsi, iudvpsi, nrec)
          IF (npwq < npwx) THEN
            IF (ANY(dvpsi(npwq + 1:npwx, 1:nock) /= (0.d0, 0.d0))) THEN
              ierr = 10
              RETURN
            ENDIF
          ENDIF
          CALL sth_engine_set_dvbare(ik, ipert, dvpsi)
        ENDIF
        IF (.NOT. first_iter) THEN
          CALL get_buffer(dpsi, lrdwf, iudwf, nrec)
          CALL sth_engine_set_dpsi(ik, ipert, dpsi)
        ENDIF
      ENDDO
    ENDDO
    CALL SYSTEM_CLOCK(c2)
    CALL sth_engine_irrep_commit(ierr)
    CALL SYSTEM_CLOCK(c3)
    IF (ierr == 0 .AND. dvb_on) CALL dvb_build(npert, lrdvpsi, iudvpsi)
    IF (sth_timing) THEN
      BLOCK
        INTEGER(8) :: c4
        CALL SYSTEM_CLOCK(c4)
        WRITE(stdout, '(5x,"rtsc irrep t: engine_irrep ",F7.3," k-loop ",F7.3," commit ",F7.3," (free/check ",F6.3, &
             &" alloc ",F6.3," upload ",F6.3," valid+zero ",F6.3,") dvb ",F7.3," (kqg ",F6.3," local ",F6.3," nl ",F6.3, &
             &") total ",F7.3)') REAL(c1 - c0, DP) / crate, REAL(c2 - c1, DP) / crate, REAL(c3 - c2, DP) / crate, &
             sth_t_commit, REAL(c4 - c3, DP) / crate, t_kqg, sth_t_dvb, REAL(c4 - c0, DP) / crate
      END BLOCK
    ENDIF
    CALL stop_clock('sth_irrep')
  END SUBROUTINE sth_batched_irrep
  !
  !--------------------------------------------------------------------------------------------
  SUBROUTINE sth_batched_init(first_iter, npert, lrdvpsi, iudvpsi, ierr)
    !--------------------------------------------------------------------------------------------
    USE io_global,        ONLY : stdout
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
                                 sth_engine_set_dvbare, sth_engine_set_dpsi, sth_engine_setup, &
                                 sth_mp, sth_mp_eta, sth_mp_safe, sth_pfft_req, sth_pfft_check, sth_dvb_req, &
                                 sth_fftdx_req, sth_fftdx_check, &
                                 sth_fdxdp_req, sth_fdxdp_check, sth_obatch, sth_timing, &
                                 sth_engine_set_nlk, sth_qcache, sth_engine_wdev, sth_engine_set_wdev, &
                                 sth_lift, sth_lift_kchunk, sth_lift_fb, sth_cbsr, sth_cbsr_skip, &
                                 sth_engine_cbsr_map, sth_engine_rwf_count, sth_engine_rwf_put, sth_engine_kdev_map, &
                                 sth_engine_set_wdev_n, sth_engine_kdev_op
    USE sth_nscf_symrot,  ONLY : symrot_kmap, symrot_cache_n, symrot_cache_get, symrot_kop
    USE rtsc_vkb,         ONLY : vkb_batch
    USE lrus,             ONLY : becp1
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
    INTEGER(8) :: c0, c1, c2, crate, ca, cb
    REAL(DP) :: tk(6)      ! RTSC_STH_TIME: per-k loop split (get_buffer, init_us_2, g2kin+weights, set_k, dvbare/dpsi, set_wdev)
    LOGICAL :: wdev
    ! RTSC_STH_KDEV (SOLVE-LOG.md session 4): vkb by vkb_batch in chunks; evc / evq of SYMROT-mapped k rotated on the device
    LOGICAL :: kdev, kmapped, okk, okq, fk, fq
    INTEGER :: i0k, i0q, ngsk, ngsq, trk, trq, ftk, ftq, nkd, ich, nch, j, kdv, sk(3, 3), sq(3, 3), g0k(3), g0q(3)
    INTEGER, ALLOCATABLE :: mapk(:), mapq(:), ikl(:), mil(:,:)
    COMPLEX(DP), ALLOCATABLE :: phk(:), phq(:), wtmp(:,:), vkbb(:,:,:)
    !
    CALL start_clock('sth_init')
    CALL SYSTEM_CLOCK(c0, crate)
    tk = 0.d0
    ! mixed precision (per q; the FP32 copies are built by sth_engine_setup)
    sth_mp = sth_env_flag('RTSC_STH_MP')
    sth_pfft_req = sth_env_flag('RTSC_STH_PFFT')
    sth_pfft_check = sth_env_flag('RTSC_STH_PFFT_CHECK')
    sth_fftdx_req = sth_env_flag('RTSC_STH_FFTDX')
    sth_fftdx_check = sth_env_flag('RTSC_STH_FFTDX_CHECK')
    sth_fdxdp_req = sth_env_flag('RTSC_STH_FDXDP')          ! rtsc amdahl (groups/compute/AMDAHL-LOG.md)
    sth_fdxdp_check = sth_env_flag('RTSC_STH_FDXDP_CHECK')
    sth_obatch = sth_env_flag('RTSC_STH_OBATCH')
    sth_timing = sth_env_flag('RTSC_STH_TIME')
    sth_qcache = sth_env_flag('RTSC_STH_QCACHE')            ! rtsc setup (groups/compute/SETUP-LOG.md)
    sth_dvb_req = dvb_on
    CALL env_real('RTSC_MP_ETA', sth_mp_eta)
    CALL env_real('RTSC_MP_SAFE', sth_mp_safe)
    ! LIFT-F preconditioner (RTSC_STH_LIFT = E_c' in Ry; FP64 CG path; SOLVE-LOG.md "LIFT-F canary")
    sth_lift = 0.d0
    CALL env_real('RTSC_STH_LIFT', sth_lift)
    sth_lift_kchunk = sth_env_int('RTSC_STH_LIFT_KCHUNK')
    sth_lift_fb = sth_env_int('RTSC_STH_LIFT_FB')            ! LIFT-F v2 per-column TPA fallback (SOLVE-LOG "LIFT-F v2")
    sth_cbsr = sth_env_flag('RTSC_STH_CBSR')                 ! CBSR canary: exact spectral solve (SOLVE-LOG "CBSR canary")
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
    wdev = sth_engine_wdev()     ! QCACHE: vkb goes device-to-device into the engine's W
    IF (sth_timing) THEN
      CALL SYSTEM_CLOCK(c1)
      WRITE(stdout, '(5x,"rtsc sth_init: engine_init + D ",F8.3," s, vkb on device: ",L1)') &
           REAL(c1 - c0, DP) / REAL(crate, DP), wdev
    ENDIF
    !
    ALLOCATE(idxk(npwx), idxkq(npwx), wwg(nbnd, nbnd), wg1(nbnd))
    kdv = sth_env_int('RTSC_STH_KDEV')       ! 1: v1 (host maps), 2: v2 (maps on the device)
    kdev = kdv > 0 .AND. wdev .AND. .NOT. lgamma
    nkd = 0
    IF (kdev) THEN
      IF (sth_timing) CALL SYSTEM_CLOCK(ca)
      ! the SYMROT cache entries not yet on the device
      IF (symrot_cache_n() > sth_engine_rwf_count()) THEN
        ALLOCATE(wtmp(npwx, nbnd), mil(3, npwx))
        DO j = sth_engine_rwf_count() + 1, symrot_cache_n()
          CALL symrot_cache_get(j, ngsk, ich, wtmp, mil)
          CALL sth_engine_rwf_put(j, ngsk, ich, wtmp, mil)
        ENDDO
        DEALLOCATE(wtmp, mil)
      ENDIF
      ALLOCATE(mapk(npwx), mapq(npwx), phk(npwx), phq(npwx))
      IF (sth_timing) THEN
        CALL SYSTEM_CLOCK(cb); tk(1) = tk(1) + REAL(cb - ca, DP) / crate; ca = cb
      ENDIF
      ! vkb of every k+q, in chunks (bitwise init_us_2's), device-to-device into W
      nch = MIN(nksq, 256, npwx)
      ALLOCATE(vkbb(npwx, MAX(nkb, 1), nch), ikl(nch))
      !$acc enter data create(vkbb)
      DO ich = 1, nksq, nch
        j = MIN(nch, nksq - ich + 1)
        DO ik = 1, j
          ikl(ik) = ikqs(ich + ik - 1)
          mapk(ik) = ngk(ikl(ik))
        ENDDO
        CALL vkb_batch(j, ikl(1:j), vkbb)
        !$acc host_data use_device(vkbb)
        CALL sth_engine_set_wdev_n(ich, j, mapk(1:j), vkbb)
        !$acc end host_data
      ENDDO
      !$acc exit data delete(vkbb)
      DEALLOCATE(vkbb, ikl)
      IF (sth_timing) THEN
        CALL SYSTEM_CLOCK(cb); tk(2) = tk(2) + REAL(cb - ca, DP) / crate; ca = cb
      ENDIF
    ENDIF
    DO ik = 1, nksq
      IF (sth_timing) CALL SYSTEM_CLOCK(ca)
      ikk = ikks(ik); ikq = ikqs(ik)
      npw = ngk(ikk); npwq = ngk(ikq)
      nock = nbnd_occ(ikk); nockq = nbnd_occ(ikq)
      idxk(1:npw) = dffts%nl(igk_k(1:npw, ikk))
      idxkq(1:npwq) = dffts%nl(igk_k(1:npwq, ikq))
      kmapped = .FALSE.
      IF (kdev .AND. kdv == 2) THEN
        CALL symrot_kop(ikk, i0k, sk, trk, g0k, fk, okk)
        CALL symrot_kop(ikq, i0q, sq, trq, g0q, fq, okq)
        kmapped = okk .AND. okq .AND. .NOT. fk .AND. .NOT. fq
        IF (kmapped) kmapped = (i0k <= sth_engine_rwf_count() .AND. i0q <= sth_engine_rwf_count())
      ELSE IF (kdev) THEN
        CALL symrot_kmap(ikk, i0k, ngsk, trk, mapk, phk, okk)
        CALL symrot_kmap(ikq, i0q, ngsq, trq, mapq, phq, okq)
        kmapped = okk .AND. okq
        IF (kmapped) kmapped = (i0k <= sth_engine_rwf_count() .AND. i0q <= sth_engine_rwf_count())
      ENDIF
      IF (.NOT. kmapped) THEN
        CALL get_buffer(evc, lrwfc, iuwfc, ikk)
        IF (.NOT. lgamma) CALL get_buffer(evq, lrwfc, iuwfc, ikq)   ! lgamma: evq => evc, ikq = ikk
      ENDIF
      IF (sth_timing) THEN
        CALL SYSTEM_CLOCK(cb); tk(1) = tk(1) + REAL(cb - ca, DP) / crate; ca = cb
      ENDIF
      IF (.NOT. kdev) CALL init_us_2(npwq, igk_k(1, ikq), xk(1, ikq), vkb, .true.)
      IF (kdev) THEN
        CONTINUE     ! vkb already in W (chunks above)
      ELSE IF (wdev) THEN
        !$acc host_data use_device(vkb)
        CALL sth_engine_set_wdev(ik, npwq, vkb)
        !$acc end host_data
        ! the host copy is left as the default path leaves it (vkb of the last k+q)
        IF (ik == nksq) THEN
          !$acc update host(vkb)
        ENDIF
      ELSE
        !$acc update host(vkb)
      ENDIF
      IF (sth_timing) THEN
        CALL SYSTEM_CLOCK(cb); tk(2) = tk(2) + REAL(cb - ca, DP) / crate; ca = cb
      ENDIF
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
      IF (sth_timing) THEN
        CALL SYSTEM_CLOCK(cb); tk(3) = tk(3) + REAL(cb - ca, DP) / crate; ca = cb
      ENDIF
      CALL sth_engine_set_k(ik, npw, npwq, nock, nockq, wk(ikk), idxk(1:npw), idxkq(1:npwq), evc, evq, &
                            et(:, ikk), g2kin, vkb, wwg(:, 1:nock), wg1(1:nock), et_kq=et(:, ikq), nowf=kmapped)
      IF (kdev .AND. kdv == 2) THEN
        IF (kmapped) THEN
          CALL sth_engine_kdev_op(ik, 1, i0k, trk, sk, g0k)
          CALL sth_engine_kdev_op(ik, 2, i0q, trq, sq, g0q)
          nkd = nkd + 1
        ELSE
          CALL sth_engine_kdev_op(ik, 1, 0, 1, sk, g0k)
          CALL sth_engine_kdev_op(ik, 2, 0, 1, sq, g0q)
        ENDIF
      ELSE IF (kdev) THEN
        IF (kmapped) THEN
          ftk = 0; ftq = 0
          IF (ANY(phk(1:npw) /= (1.d0, 0.d0))) ftk = 1
          IF (ANY(phq(1:npwq) /= (1.d0, 0.d0))) ftq = 1
          CALL sth_engine_kdev_map(ik, 1, i0k, trk, ftk, mapk, phk)
          CALL sth_engine_kdev_map(ik, 2, i0q, trq, ftq, mapq, phq)
          nkd = nkd + 1
        ELSE
          CALL sth_engine_kdev_map(ik, 1, 0, 1, 0, mapk, phk)
          CALL sth_engine_kdev_map(ik, 2, 0, 1, 0, mapq, phq)
        ENDIF
      ENDIF
      IF (dvb_on) CALL sth_engine_set_nlk(ik, becp1(ik)%k, dvb_al(1, ik)%k, dvb_al(2, ik)%k, dvb_al(3, ik)%k)
      IF (sth_timing) THEN
        CALL SYSTEM_CLOCK(cb); tk(4) = tk(4) + REAL(cb - ca, DP) / crate; ca = cb
      ENDIF
      DO ipert = 1, npert
        nrec = (ipert - 1) * nksq + ik
        IF (dvb_on) THEN
          IF (.NOT. first_iter) THEN
            CALL get_buffer(dpsi, lrdwf, iudwf, nrec)
            CALL sth_engine_set_dpsi(ik, ipert, dpsi)
          ENDIF
          CYCLE
        ENDIF
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
        IF (.NOT. first_iter) THEN
          ! restart in the middle of an irrep: warm start from iudwf, as vanilla
          CALL get_buffer(dpsi, lrdwf, iudwf, nrec)
          CALL sth_engine_set_dpsi(ik, ipert, dpsi)
        ENDIF
      ENDDO
      IF (sth_timing) THEN
        CALL SYSTEM_CLOCK(cb); tk(5) = tk(5) + REAL(cb - ca, DP) / crate
      ENDIF
    ENDDO
    DEALLOCATE(idxk, idxkq, wwg, wg1)
    IF (kdev) THEN
      ! leave the host state as the default loop leaves it: evc / evq / vkb of the last k, k+q
      ikk = ikks(nksq); ikq = ikqs(nksq)
      CALL get_buffer(evc, lrwfc, iuwfc, ikk)
      CALL get_buffer(evq, lrwfc, iuwfc, ikq)
      CALL init_us_2(ngk(ikq), igk_k(1, ikq), xk(1, ikq), vkb, .true.)
      !$acc update host(vkb)
      DEALLOCATE(mapk, mapq, phk, phq)
      IF (sth_timing) WRITE(stdout, '(5x,"rtsc kdev: ",I0," of ",I0," k blocks with both wavefunctions rotated on the device")') &
           nkd, nksq
    ENDIF
    IF (lgamma) THEN
      !$acc update device(evc)
    ELSE
      !$acc update device(evc, evq)
    ENDIF
    IF (sth_timing) THEN
      CALL SYSTEM_CLOCK(c1, crate)
      WRITE(*, '(A,F8.3,A)') '     rtsc sth_init: host per-k loop ', REAL(c1 - c0, DP) / REAL(crate, DP), ' s'
      WRITE(stdout, '(5x,"rtsc sth_init k-loop: get_buffer ",F7.3," init_us_2+vkb ",F7.3," g2kin+weights ",F7.3, &
           &" set_k ",F7.3," dvbare/dpsi ",F7.3)') tk(1:5)
    ENDIF
    ! CBSR (RTSC_STH_CBSR=1): which RTSC_NSCF_SYMROT cache entry, and which G map, every k+q was rotated from
    IF (sth_cbsr) THEN
    BLOCK
      INTEGER :: i0, ngs, tr, nmap, ios
      LOGICAL :: ok
      INTEGER, ALLOCATABLE :: imap(:)
      COMPLEX(DP), ALLOCATABLE :: ph(:)
      CHARACTER(LEN=8) :: sv
      ! q = Gamma reads the scf psi (no SYMROT map): every k needs its own basis, which costs more than Gamma's CG
      ! (|K_irr| eigendecompositions for one q); CG is kept there unless RTSC_STH_CBSR_GAMMA=1
      sth_cbsr_skip = .FALSE.
      IF (lgamma) THEN
        CALL get_environment_variable('RTSC_STH_CBSR_GAMMA', sv, STATUS=ios)
        sth_cbsr_skip = .NOT. (ios == 0 .AND. TRIM(sv) == '1')
      ENDIF
      ALLOCATE(imap(npwx), ph(npwx))
      nmap = 0
      DO ik = 1, nksq
        ok = .FALSE.
        IF (.NOT. lgamma) CALL symrot_kmap(ikqs(ik), i0, ngs, tr, imap, ph, ok)
        IF (ok) THEN
          nmap = nmap + 1
        ELSE
          i0 = 0; ngs = 0; tr = 1
        ENDIF
        CALL sth_engine_cbsr_map(ik, i0, ngs, tr, imap, ph)
      ENDDO
      WRITE(stdout, '(5x,"rtsc cbsr: ",I0," of ",I0," k+q mapped onto RTSC_NSCF_SYMROT cache entries")') nmap, nksq
      DEALLOCATE(imap, ph)
    END BLOCK
    ENDIF
    CALL sth_engine_setup(ierr)
    IF (ierr == 0 .AND. dvb_on) CALL dvb_build(npert, lrdvpsi, iudvpsi)
    IF (sth_timing) THEN
      CALL SYSTEM_CLOCK(c2)
      WRITE(*, '(A,F8.3,A)') '     rtsc sth_init: setup + dvb ', REAL(c2 - c1, DP) / REAL(crate, DP), ' s'
    ENDIF
    CALL stop_clock('sth_init')
  END SUBROUTINE sth_batched_init
  !
  !--------------------------------------------------------------------------------------------
  SUBROUTINE kqg_table(kqg)
    !--------------------------------------------------------------------------------------------
    !! kqg(npwx, 3, nksq) = k+q+G for the engine (dvb_build, becs). RTSC_STH_QCACHE: built once per q and kept on
    !! the device by the engine; then kqg is returned with a dummy size (the engine does not read it).
    USE gvect,            ONLY : g
    USE klist,            ONLY : xk, ngk, igk_k
    USE wvfct,            ONLY : npwx
    USE qpoint,           ONLY : nksq, ikqs
    USE sth_engine,       ONLY : sth_engine_kqg_ready, sth_engine_set_kqg
    REAL(DP), ALLOCATABLE, INTENT(OUT) :: kqg(:,:,:)
    INTEGER :: ik, ikq, ig, a
    IF (sth_engine_kqg_ready()) THEN
      ALLOCATE(kqg(npwx, 3, 1))
      RETURN
    ENDIF
    ALLOCATE(kqg(npwx, 3, nksq))
    kqg = 0.d0
    DO ik = 1, nksq
      ikq = ikqs(ik)
      DO a = 1, 3
        DO ig = 1, ngk(ikq)
          kqg(ig, a, ik) = xk(a, ikq) + g(a, igk_k(ig, ikq))
        ENDDO
      ENDDO
    ENDDO
    CALL sth_engine_set_kqg(kqg)      ! no-op unless RTSC_STH_QCACHE (and it fits)
  END SUBROUTINE kqg_table
  !
  !--------------------------------------------------------------------------------------------
  SUBROUTINE dvb_build(npert, lrdvpsi, iudvpsi)
    !--------------------------------------------------------------------------------------------
    !! dV_bare psi of the current irrep on the device (after the engine's irrep commit)
    USE ions_base,        ONLY : nat, ityp
    USE uspp,             ONLY : nkb, ofsbeta
    USE uspp_param,       ONLY : nh
    USE cell_base,        ONLY : tpiba
    USE gvect,            ONLY : g
    USE klist,            ONLY : xk, ngk, igk_k
    USE wvfct,            ONLY : npwx
    USE qpoint,           ONLY : nksq, ikqs
    USE buffers,          ONLY : get_buffer
    USE eqv,              ONLY : dvpsi
    USE io_global,        ONLY : stdout
    USE sth_engine,       ONLY : sth_engine_build_dvb, sth_engine_dvb_check, sth_engine_kqg_ready
    INTEGER, INTENT(IN) :: npert, lrdvpsi, iudvpsi
    INTEGER, ALLOCATABLE :: kbat(:)
    REAL(DP), ALLOCATABLE :: kqg(:,:,:)
    INTEGER :: na, ih, ik, ikq, ig, a, ipert, nrec
    REAL(DP) :: dmax, ymax
    INTEGER(8) :: c0, c1, crate
    CALL start_clock('sth_dvb')
    CALL SYSTEM_CLOCK(c0, crate)
    ALLOCATE(kbat(MAX(nkb, 1)))
    DO na = 1, nat
      DO ih = 1, nh(ityp(na))
        kbat(ofsbeta(na) + ih) = na
      ENDDO
    ENDDO
    CALL kqg_table(kqg)
    CALL SYSTEM_CLOCK(c1)
    t_kqg = REAL(c1 - c0, DP) / REAL(crate, DP)
    CALL sth_engine_build_dvb(dvb_vl, dvb_u, 3 * nat, kbat, tpiba, kqg)
    DEALLOCATE(kbat, kqg)
    IF (sth_env_flag('RTSC_STH_DVB_CHECK')) THEN
      dmax = 0.d0; ymax = 0.d0
      DO ik = 1, nksq
        DO ipert = 1, npert
          nrec = (ipert - 1) * nksq + ik
          CALL get_buffer(dvpsi, lrdvpsi, iudvpsi, nrec)
          CALL sth_engine_dvb_check(ik, ipert, dvpsi, dmax, ymax)
        ENDDO
      ENDDO
      WRITE(stdout, '(5x,"rtsc dvb check: max|engine - vanilla| / max|vanilla| ",ES10.3," (max ",ES10.3,")")') &
           dmax / MAX(ymax, TINY(1.d0)), ymax
    ENDIF
    CALL stop_clock('sth_dvb')
  END SUBROUTINE dvb_build
#endif
  !
END MODULE sth_batched
