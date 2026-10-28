!
! rtsc: offline acceptance harness for the batched Sternheimer engine (STH-ENGINE.md).
!
!   sth_harness.x <dump dir> [nrep] [bmax]
!
! Reads one sternheimer_kernel call written by LR_Modules/sth_dump.f90, runs sth_engine on it,
! and compares with the dumped vanilla outputs:
!   - drho relative L2 against drho_out (target <= 1e-8);
!   - per-column dpsi relative error, columns (k, pert, n <= nocc_k), rows 1..npwq
!     (target max <= 1e-6, median <= 1e-8);
!   - mean CG iterations against tot_num_iter / tot_cg_calls (target within 5 %);
!   - h_diag against the dumped hdiag (sanity);
! and reports the engine wall time (nrep repeated solves from the same warm start).
!
! Diagnostics (defaults leave the engine untouched):
!   sth_harness.x <dir> <nrep> <bmax> <k> <band>   print the CG history of column (k, band, pert 1)
!   STH_RB=<32..256>      threads per column reduction (changes only the summation order)
!   STH_OPVAR=1           kinetic term as (g2kin - e) x (same operator, other rounding)
!   STH_DUMP_HDIAG=1      use the dumped h_diag instead of the engine's
!   STH_SAVE=<file> / STH_REF=<file>   save the engine dpsi / compare against a saved engine run
! The last step re-solves with dvscf * (1 + 1e-14) and reports the per-column response (sensitivity).
!
PROGRAM sth_harness
  USE sth_engine
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp = sth_dp
  CHARACTER(LEN=512) :: dir, arg, line
  CHARACTER(LEN=64) :: key, s1, s2, s3, s4, s5
  INTEGER :: nksq, npwx, nbnd, nkb, npert, nnr, nr1, nr2, nr3, nr1x, nr2x, nr3x, ngauss
  INTEGER :: tot_num_iter_v, tot_cg_calls_v, u, ios, ik, ip, n, nslot, ierr, nrep, irep, ncol, nq
  LOGICAL :: first_iter, lgauss, lgamma
  REAL(dp) :: thresh, alpha_pv, ef, degauss, omega, tpiba2
  INTEGER, ALLOCATABLE :: npw(:), npwq(:), nock(:), nockq(:), idx(:)
  REAL(dp), ALLOCATABLE :: wk(:), vrs(:), etk(:), etkq(:), g2(:), wwg(:,:), wg1(:), hd(:,:), hd_e(:,:)
  REAL(dp), ALLOCATABLE :: colerr(:), colnrm(:), colgap(:), colsrt(:)
  INTEGER, ALLOCATABLE :: colk(:), coln(:), colp(:), ordr(:)
  COMPLEX(dp), ALLOCATABLE :: dmat(:,:), evc(:,:), evq(:,:), vkb(:,:), dvb(:,:), dpsi(:,:), dpsi_e(:,:)
  COMPLEX(dp), ALLOCATABLE :: dvscf(:,:), drho_in(:,:), drho_out(:,:), drho(:,:)
  INTEGER :: tot_num_iter, tot_cg_calls, nfail, dbg_k = 0, dbg_n = 0, tot_num_iter_iter1
  REAL(dp) :: t0, t1, num, den, e, hderr, drho_err, tbest, med
  !
  CALL get_command_argument(1, dir)
  nrep = 3
  IF (command_argument_count() >= 2) THEN
    CALL get_command_argument(2, arg)
    READ(arg, *) nrep
  ENDIF
  IF (command_argument_count() >= 3) THEN
    CALL get_command_argument(3, arg)
    READ(arg, *) sth_bmax
  ENDIF
  CALL get_environment_variable('STH_RB', arg, STATUS=ios)
  IF (ios == 0) READ(arg, *) sth_rb
  CALL get_environment_variable('STH_OPVAR', arg, STATUS=ios)
  IF (ios == 0) READ(arg, *) sth_opvar
  IF (command_argument_count() >= 5) THEN
    ! debug: print the CG history of column (k, band) for pert 1
    CALL get_command_argument(4, arg)
    READ(arg, *) dbg_k
    CALL get_command_argument(5, arg)
    READ(arg, *) dbg_n
  ENDIF
  !
  ! ---- meta.txt
  OPEN(NEWUNIT=u, FILE=TRIM(dir)//'/meta.txt', STATUS='old', ACTION='read')
  tot_num_iter_v = -1; tot_cg_calls_v = -1
  DO
    READ(u, '(A)', IOSTAT=ios) line
    IF (ios /= 0) EXIT
    READ(line, *) key
    SELECT CASE (TRIM(key))
    CASE ('nksq');  READ(line, *) key, nksq
    CASE ('npwx');  READ(line, *) key, npwx
    CASE ('nbnd');  READ(line, *) key, nbnd
    CASE ('nkb');   READ(line, *) key, nkb
    CASE ('npert'); READ(line, *) key, npert
    CASE ('nnr');   READ(line, *) key, nnr
    CASE ('nr1');   READ(line, *) key, nr1, s1, nr2, s2, nr3
    CASE ('nr1x');  READ(line, *) key, nr1x, s1, nr2x, s2, nr3x
    CASE ('first_iter'); READ(line, *) key, first_iter
    CASE ('thresh');   READ(line, *) key, thresh
    CASE ('alpha_pv'); READ(line, *) key, alpha_pv
    CASE ('ef');       READ(line, *) key, ef
    CASE ('degauss');  READ(line, *) key, degauss
    CASE ('ngauss');   READ(line, *) key, ngauss
    CASE ('lgauss');   READ(line, *) key, lgauss
    CASE ('lgamma');   READ(line, *) key, lgamma
    CASE ('omega');    READ(line, *) key, omega
    CASE ('tpiba2');   READ(line, *) key, tpiba2
    CASE ('k')
      IF (.NOT. ALLOCATED(npw)) ALLOCATE(npw(nksq), npwq(nksq), nock(nksq), nockq(nksq), wk(nksq))
      READ(line, *) key, ik, s1, npw(ik), s2, npwq(ik), s3, nock(ik), s4, nockq(ik), s5, wk(ik)
    CASE ('tot_num_iter'); READ(line, *) key, tot_num_iter_v
    CASE ('tot_cg_calls'); READ(line, *) key, tot_cg_calls_v
    END SELECT
  ENDDO
  CLOSE(u)
  IF (.NOT. lgauss .OR. lgamma) STOP 'harness: dump is not an lgauss, q /= 0 case'
  nslot = MAXVAL(nock)
  WRITE(*, '(A,A)') 'dump: ', TRIM(dir)
  WRITE(*, '(A,I5,A,I5,A,I3,A,I3,A,I2,A,I3,A,ES10.3,A,L1)') ' nksq', nksq, ' npwx', npwx, ' nbnd', nbnd, &
       ' nkb', nkb, ' npert', npert, ' nslot', nslot, ' thresh', thresh, ' first_iter ', first_iter
  !
  ! ---- engine setup
  t0 = wtime()
  CALL sth_engine_init(nksq, npwx, nbnd, nkb, npert, nslot, nr1, nr2, nr3, nr1x, nr2x, nr3x, nnr, &
                       alpha_pv, omega, ierr)
  IF (ierr /= 0) STOP 'harness: sth_engine_init failed'
  ALLOCATE(vrs(nnr), dmat(nkb, nkb))
  CALL rd_r('vrs.bin', vrs, nnr)
  CALL rd_c('dmat.bin', dmat, nkb * nkb)
  CALL sth_engine_set_global(vrs, dmat)
  ALLOCATE(idx(npwx), evc(npwx, nbnd), evq(npwx, nbnd), vkb(npwx, nkb), etk(nbnd), etkq(nbnd), g2(npwx))
  ALLOCATE(dvb(npwx, nbnd), dpsi(npwx, nbnd), dpsi_e(npwx, nbnd))
  ALLOCATE(wwg(nbnd, nslot), wg1(nslot))
  DO ik = 1, nksq
    ! idx_k is only consumed through set_k (copied), so read both maps in turn
    BLOCK
      INTEGER :: idxk(npw(ik)), idxkq(npwq(ik))
      CALL rd_i(kt(ik, 0)//'igk_k.bin', idxk, npw(ik))
      CALL rd_i(kt(ik, 0)//'igk_kq.bin', idxkq, npwq(ik))
      CALL rd_c(kt(ik, 0)//'evc.bin', evc, npwx * nbnd)
      CALL rd_c(kt(ik, 0)//'evq.bin', evq, npwx * nbnd)
      CALL rd_r(kt(ik, 0)//'et_k.bin', etk, nbnd)
      CALL rd_r(kt(ik, 0)//'et_kq.bin', etkq, nbnd)
      CALL rd_r(kt(ik, 0)//'g2kin.bin', g2, npwx)
      CALL rd_c(kt(ik, 0)//'vkb.bin', vkb, npwx * nkb)
      CALL sth_weights(nbnd, nock(ik), nockq(ik), etk, etkq, ef, degauss, ngauss, alpha_pv, wwg, wg1)
      CALL sth_engine_set_k(ik, npw(ik), npwq(ik), nock(ik), nockq(ik), wk(ik), idxk, idxkq, evc, evq, &
                            etk, g2, vkb, wwg(:, 1:nock(ik)), wg1(1:nock(ik)))
    END BLOCK
    DO ip = 1, npert
      CALL rd_c(kt(ik, ip)//'dvbare.bin', dvb, npwx * nbnd)
      CALL sth_engine_set_dvbare(ik, ip, dvb)
      IF (.NOT. first_iter) THEN
        CALL rd_c(kt(ik, ip)//'dpsi_in.bin', dpsi, npwx * nbnd)
        CALL sth_engine_set_dpsi(ik, ip, dpsi)
      ENDIF
    ENDDO
  ENDDO
  t1 = wtime()
  WRITE(*, '(A,F8.3,A)') ' read + stage: ', t1 - t0, ' s'
  sth_timing = .TRUE.
  IF (dbg_k > 0) THEN
    sth_debug_slot = dbg_n + nslot * npert * (dbg_k - 1)
    CALL rd_r(kt(dbg_k, 0)//'et_k.bin', etk, nbnd)
    CALL rd_r(kt(dbg_k, 0)//'et_kq.bin', etkq, nbnd)
    WRITE(*, '(A,I4,A,I2,A,I2,A,I2)') ' debug column k', dbg_k, ' band', dbg_n, ' nocc_k', nock(dbg_k), ' nocc_kq', nockq(dbg_k)
    WRITE(*, '(A,12F10.5)') '   et_k  ', etk
    WRITE(*, '(A,12F10.5)') '   et_kq ', etkq
    CALL sth_weights(nbnd, nock(dbg_k), nockq(dbg_k), etk, etkq, ef, degauss, ngauss, alpha_pv, wwg, wg1)
    WRITE(*, '(A,F10.5,A,ES12.4)') '   ef ', ef, '  wg1 ', wg1(dbg_n)
    WRITE(*, '(A,12ES10.2)') '   wwg(:,n) ', wwg(:, dbg_n)
  ENDIF
  CALL sth_engine_setup(ierr)
  IF (ierr /= 0) STOP 'harness: sth_engine_setup failed'
  WRITE(*, '(A,F8.3,A,I0)') ' engine setup:  ', sth_t_setup, ' s   bmax ', sth_bmax
  !
  ! ---- h_diag sanity
  ALLOCATE(hd(npwx, nslot), hd_e(npwx, nbnd))
  hderr = 0
  DO ik = 1, nksq
    CALL sth_engine_get_hdiag(ik, hd)
    CALL rd_r(kt(ik, 0)//'hdiag.bin', hd_e, npwx * nbnd)
    nq = npwq(ik)
    DO n = 1, nock(ik)
      hderr = MAX(hderr, MAXVAL(ABS(hd(1:nq, n) - hd_e(1:nq, n))))
    ENDDO
  ENDDO
  WRITE(*, '(A,ES10.3)') ' h_diag max abs diff vs dump: ', hderr
  CALL get_environment_variable('STH_DUMP_HDIAG', arg, STATUS=ios)
  IF (ios == 0) THEN
    WRITE(*, '(A)') ' (diagnostic) using the dumped h_diag'
    DO ik = 1, nksq
      CALL rd_r(kt(ik, 0)//'hdiag.bin', hd_e, npwx * nbnd)
      CALL sth_engine_set_hdiag(ik, hd_e)
    ENDDO
  ENDIF
  !
  ! ---- solve
  ALLOCATE(dvscf(nnr, npert), drho_in(nnr, npert), drho_out(nnr, npert), drho(nnr, npert))
  CALL rd_c('dvscf.bin', dvscf, nnr * npert)
  CALL rd_c('drho_in.bin', drho_in, nnr * npert)
  CALL rd_c('drho_out.bin', drho_out, nnr * npert)
  tbest = HUGE(1.0_dp)
  DO irep = 1, MAX(1, nrep)
    IF (irep > 1) CALL sth_engine_reset_dpsi()
    drho = drho_in
    t0 = wtime()
    CALL sth_engine_solve(first_iter, thresh, dvscf, drho, tot_num_iter, tot_cg_calls, nfail)
    t1 = wtime()
    tbest = MIN(tbest, t1 - t0)
    WRITE(*, '(A,I2,A,F8.4,A,4F8.4,A,I4,A,I9)') ' solve ', irep, ': wall ', t1 - t0, &
         ' s  [dpot ortho cg drho] ', sth_t_dpot, sth_t_ortho, sth_t_cg, sth_t_drho, '  steps ', sth_nsteps, &
         '  col-applies ', sth_ncolapp
  ENDDO
  sth_timing = .FALSE.
  ! untimed repeat: the timers add syncs, so report a run without them too
  CALL sth_engine_reset_dpsi()
  drho = drho_in
  t0 = wtime()
  CALL sth_engine_solve(first_iter, thresh, dvscf, drho, tot_num_iter, tot_cg_calls, nfail)
  t1 = wtime()
  WRITE(*, '(A,F8.4,A,F8.4,A)') ' solve (no phase timers): ', t1 - t0, ' s ; best timed ', tbest, ' s'
  !
  ! ---- compare
  num = SUM(ABS(drho - drho_out)**2)
  den = SUM(ABS(drho_out)**2)
  drho_err = SQRT(num / den)
  WRITE(*, '(A,ES10.3,A)') ' drho rel L2 vs drho_out:  ', drho_err, '   (target <= 1e-8)'
  WRITE(*, '(A,ES10.3)') '   (|drho_out - drho_in| / |drho_out|: ', SQRT(SUM(ABS(drho_out - drho_in)**2) / den)
  ncol = SUM(nock) * npert
  ALLOCATE(colerr(ncol), colnrm(ncol), colgap(ncol), colk(ncol), coln(ncol), colp(ncol), colsrt(ncol), ordr(ncol))
  ncol = 0
  DO ik = 1, nksq
    nq = npwq(ik)
    DO ip = 1, npert
      CALL sth_engine_get_dpsi(ik, ip, dpsi_e)
      CALL rd_c(kt(ik, ip)//'dpsi_out.bin', dpsi, npwx * nbnd)
      DO n = 1, nock(ik)
        ncol = ncol + 1
        num = SUM(ABS(dpsi_e(1:nq, n) - dpsi(1:nq, n))**2)
        den = SUM(ABS(dpsi(1:nq, n))**2)
        colerr(ncol) = SQRT(num / MAX(den, TINY(1.0_dp)))
        colnrm(ncol) = SQRT(den); colk(ncol) = ik; coln(ncol) = n; colp(ncol) = ip
        CALL rd_r(kt(ik, 0)//'et_k.bin', etk, nbnd)
        CALL rd_r(kt(ik, 0)//'et_kq.bin', etkq, nbnd)
        ! smallest |e_kq(m) - e_k(n)| over the k+q bands not projected by P_v (m > nocc_kq)
        colgap(ncol) = HUGE(1.0_dp)
        IF (nockq(ik) < nbnd) colgap(ncol) = MINVAL(ABS(etkq(nockq(ik) + 1:nbnd) - etk(n)))
      ENDDO
    ENDDO
  ENDDO
  colsrt(1:ncol) = colerr(1:ncol)
  CALL sort(colsrt, ncol)
  ! worst 8 columns
  DO irep = 1, ncol
    ordr(irep) = irep
  ENDDO
  DO irep = 1, MIN(8, ncol)
    n = MAXLOC(colerr(1:ncol), 1)
    WRITE(*, '(A,I4,A,I2,A,I2,A,ES10.3,A,ES10.3,A,ES10.3,A)') '   worst: k ', colk(n), ' pert ', colp(n), &
         ' band ', coln(n), '  err ', colerr(n), '  |dpsi| ', colnrm(n), '  min gap to unprojected k+q band ', &
         colgap(n), ' Ry'
    colerr(n) = -1.0_dp
  ENDDO
  colerr(1:ncol) = colsrt(1:ncol)
  med = colerr((ncol + 1) / 2)
  WRITE(*, '(A,I6,A,ES10.3,A,ES10.3,A,ES10.3,A)') ' dpsi per-column rel err (', ncol, ' cols): max ', &
       colerr(ncol), '  median ', med, '  p99 ', colerr(MAX(1, (99 * ncol) / 100)), &
       '   (targets 1e-6 / 1e-8)'
  e = REAL(tot_num_iter, dp) / REAL(tot_cg_calls, dp)
  WRITE(*, '(A,I7,A,I5,A,F8.4,A,F8.4,A,F7.3,A)') ' CG iterations: engine ', tot_num_iter, ' /', tot_cg_calls, &
       ' = ', e, '   vanilla ', REAL(tot_num_iter_v, dp) / REAL(tot_cg_calls_v, dp), '   (', &
       100.0_dp * (e / (REAL(tot_num_iter_v, dp) / REAL(tot_cg_calls_v, dp)) - 1.0_dp), ' %)'
  WRITE(*, '(A,I0)') ' unconverged (k,pert) solves: ', nfail
  tot_num_iter_iter1 = tot_num_iter
  !
  ! ---- engine against a saved engine run (STH_SAVE=file writes, STH_REF=file compares)
  CALL get_environment_variable('STH_SAVE', arg, STATUS=ios)
  IF (ios == 0) THEN
    OPEN(NEWUNIT=u, FILE=TRIM(arg), ACCESS='stream', FORM='unformatted', STATUS='replace')
    DO ik = 1, nksq
      DO ip = 1, npert
        CALL sth_engine_get_dpsi(ik, ip, dpsi_e)
        WRITE(u) dpsi_e
      ENDDO
    ENDDO
    CLOSE(u)
  ENDIF
  CALL get_environment_variable('STH_REF', arg, STATUS=ios)
  IF (ios == 0) THEN
    OPEN(NEWUNIT=u, FILE=TRIM(arg), ACCESS='stream', FORM='unformatted', STATUS='old')
    num = 0; ncol = 0
    DO ik = 1, nksq
      nq = npwq(ik)
      DO ip = 1, npert
        CALL sth_engine_get_dpsi(ik, ip, dpsi_e)
        READ(u) dpsi
        DO n = 1, nock(ik)
          ncol = ncol + 1
          colsrt(ncol) = SQRT(SUM(ABS(dpsi_e(1:nq, n) - dpsi(1:nq, n))**2) / SUM(ABS(dpsi(1:nq, n))**2))
          IF (colsrt(ncol) > num) THEN
            num = colsrt(ncol); colk(1) = ik; coln(1) = n
          ENDIF
        ENDDO
      ENDDO
    ENDDO
    CLOSE(u)
    CALL sort(colsrt, ncol)
    WRITE(*, '(A,ES10.3,A,I4,A,I2,A,ES10.3)') ' engine vs saved engine run: max col diff ', num, ' at k ', colk(1), &
         ' band ', coln(1), '  median ', colsrt((ncol + 1) / 2)
  ENDIF
  !
  ! ---- rounding-sensitivity probe: dvscf * (1 + 1e-14), engine against itself, per column
  BLOCK
    COMPLEX(dp), ALLOCATABLE :: ref(:,:,:)
    REAL(dp) :: smax
    INTEGER :: kmax, nmax, i
    ALLOCATE(ref(npwx, nbnd, nksq * npert))
    DO ik = 1, nksq
      DO ip = 1, npert
        CALL sth_engine_get_dpsi(ik, ip, ref(:, :, ip + npert * (ik - 1)))
      ENDDO
    ENDDO
    CALL sth_engine_reset_dpsi()
    drho = drho_in
    CALL sth_engine_solve(first_iter, thresh, dvscf * (1.0_dp + 1.0d-14), drho, tot_num_iter, tot_cg_calls, nfail)
    smax = 0; kmax = 0; nmax = 0; i = 0
    DO ik = 1, nksq
      nq = npwq(ik)
      DO ip = 1, npert
        CALL sth_engine_get_dpsi(ik, ip, dpsi_e)
        DO n = 1, nock(ik)
          i = i + 1
          num = SQRT(SUM(ABS(dpsi_e(1:nq, n) - ref(1:nq, n, ip + npert * (ik - 1)))**2) / &
                     SUM(ABS(ref(1:nq, n, ip + npert * (ik - 1)))**2))
          colsrt(i) = num
          IF (num > smax) THEN
            smax = num; kmax = ik; nmax = n
          ENDIF
        ENDDO
      ENDDO
    ENDDO
    CALL sort(colsrt, i)
    WRITE(*, '(A,ES10.3,A,I4,A,I2,A,ES10.3,A,I6)') ' sensitivity (dvscf*(1+1e-14)): max col diff ', smax, &
         ' at k ', kmax, ' band ', nmax, '  median ', colsrt((i + 1) / 2), '  iters ', tot_num_iter
  END BLOCK
  e = REAL(tot_num_iter_v, dp) / REAL(tot_cg_calls_v, dp)
  WRITE(*, '(A,L1,A,L1,A,L1,A,L1)') ' criteria: drho<=1e-8 ', drho_err <= 1.0d-8, '  max<=1e-6 ', &
       colerr(ncol) <= 1.0d-6, '  median<=1e-8 ', med <= 1.0d-8, '  iters within 5% ', &
       ABS(REAL(tot_num_iter_iter1, dp) / REAL(tot_cg_calls, dp) / e - 1.0_dp) <= 0.05_dp
  IF (drho_err <= 1.0d-8 .AND. colerr(ncol) <= 1.0d-6 .AND. med <= 1.0d-8 .AND. &
      ABS(REAL(tot_num_iter_iter1, dp) / REAL(tot_cg_calls, dp) / e - 1.0_dp) <= 0.05_dp) THEN
    WRITE(*, '(A)') ' ACCEPTANCE: PASS'
  ELSE
    WRITE(*, '(A)') ' ACCEPTANCE: FAIL'
  ENDIF
  CALL sth_engine_shutdown()
  !
CONTAINS
  !
  FUNCTION kt(ik, ip) RESULT(t)
    INTEGER, INTENT(IN) :: ik, ip
    CHARACTER(LEN=:), ALLOCATABLE :: t
    CHARACTER(LEN=16) :: b
    IF (ip > 0) THEN
      WRITE(b, '("k",I4.4,"_p",I2.2,"_")') ik, ip
    ELSE
      WRITE(b, '("k",I4.4,"_")') ik
    ENDIF
    t = TRIM(b)
  END FUNCTION kt
  !
  SUBROUTINE rd_c(f, a, n)
    CHARACTER(LEN=*), INTENT(IN) :: f
    INTEGER, INTENT(IN) :: n
    COMPLEX(dp), INTENT(OUT) :: a(n)
    INTEGER :: uu
    OPEN(NEWUNIT=uu, FILE=TRIM(dir)//'/'//f, ACCESS='stream', FORM='unformatted', STATUS='old', ACTION='read')
    READ(uu) a
    CLOSE(uu)
  END SUBROUTINE rd_c
  !
  SUBROUTINE rd_r(f, a, n)
    CHARACTER(LEN=*), INTENT(IN) :: f
    INTEGER, INTENT(IN) :: n
    REAL(dp), INTENT(OUT) :: a(n)
    INTEGER :: uu
    OPEN(NEWUNIT=uu, FILE=TRIM(dir)//'/'//f, ACCESS='stream', FORM='unformatted', STATUS='old', ACTION='read')
    READ(uu) a
    CLOSE(uu)
  END SUBROUTINE rd_r
  !
  SUBROUTINE rd_i(f, a, n)
    CHARACTER(LEN=*), INTENT(IN) :: f
    INTEGER, INTENT(IN) :: n
    INTEGER, INTENT(OUT) :: a(n)
    INTEGER :: uu
    OPEN(NEWUNIT=uu, FILE=TRIM(dir)//'/'//f, ACCESS='stream', FORM='unformatted', STATUS='old', ACTION='read')
    READ(uu) a
    CLOSE(uu)
  END SUBROUTINE rd_i
  !
  REAL(dp) FUNCTION wtime()
    INTEGER(8) :: c, r
    CALL SYSTEM_CLOCK(c, r)
    wtime = REAL(c, dp) / REAL(r, dp)
  END FUNCTION wtime
  !
  SUBROUTINE sort(a, n)
    ! insertion sort is enough for a few thousand columns
    INTEGER, INTENT(IN) :: n
    REAL(dp), INTENT(INOUT) :: a(n)
    INTEGER :: i, j
    REAL(dp) :: x
    DO i = 2, n
      x = a(i)
      j = i - 1
      DO WHILE (j >= 1)
        IF (a(j) <= x) EXIT
        a(j + 1) = a(j)
        j = j - 1
      ENDDO
      a(j + 1) = x
    ENDDO
  END SUBROUTINE sort
  !
END PROGRAM sth_harness
