!
! rtsc: one-shot DFPT pass from an interpolated dvscf (groups/compute/DQ-LOG.md).
!
! RTSC_DQ=1, RTSC_DQ_WPOT=<dir>: at every q /= Gamma, solve_linter calls dq_solve_irrep instead of the
! self-consistent dfpt_kernel. The induced potential dV_in of the irrep is Fourier-interpolated
! (dvscf_r2q, from the w_pot that dvscf_q2r.x wrote for a coarser q-grid), symmetrized (psymdvscf),
! and one non-self-consistent Sternheimer solve gives dpsi and dn_out (on the batched engine with
! RTSC_STH=1: one engine call per irrep, CG from dpsi = 0, no mixing). Then:
!  - dvscf written for EPW: dV_in (the interpolated, symmetrized potential);
!  - dynamical matrix: QE's drhodv/drhodvloc from dpsi and dn_out (Hellmann-Feynman form, error linear
!    in dV_in - dV*), plus the stationary correction
!       C(i,j) = - int conj(dn_out_i) (dV_in_j - dV_out_j),   dV_out = Hxc response to dn_out,
!    which makes the irrep block variational: its error is bilinear in the errors of dn_out
!    (RTSC_DQ_VAR=0 leaves it out). Hermitized downstream (symdynph_gq).
!  - printed per irrep: |dV_out - dV_in| / |dV_in| (the interpolation-quality meter) and max |C|.
! Gamma runs the normal SCF (metals need the Fermi shift there; Gamma is on every subgrid).
! RTSC_DQ_THRESH: Sternheimer threshold of the pass (default 1e-8 = 0.1 sqrt(tr2_ph = 1e-14)).
!
MODULE rtsc_dq
  USE kinds, ONLY : DP
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: dq_active, dq_solve_irrep, dq_add_dyn, dq_end_q
  !
  LOGICAL, SAVE :: have_q = .FALSE.
  REAL(DP), SAVE :: dq_xq(3) = 1.d10
  COMPLEX(DP), ALLOCATABLE, SAVE :: dv_all(:,:,:)  ! (dfftp%nnr, nspin_mag, 3 nat): interpolated dV_ind, u basis
  COMPLEX(DP), ALLOCATABLE, SAVE :: corr(:,:)      ! (npe, npe) stationary correction of the last irrep
  INTEGER, SAVE :: corr_i0 = -1, corr_npe = 0
  !
CONTAINS
  !
  !------------------------------------------------------------------------------------------------
  LOGICAL FUNCTION env_is(name, val)
    CHARACTER(LEN=*), INTENT(IN) :: name, val
    CHARACTER(LEN=16) :: s
    INTEGER :: ios
    env_is = .FALSE.
    CALL get_environment_variable(name, s, STATUS=ios)
    IF (ios == 0) env_is = (TRIM(s) == val)
  END FUNCTION env_is
  !
  !------------------------------------------------------------------------------------------------
  LOGICAL FUNCTION dq_active()
    USE control_lr, ONLY : lgamma
    dq_active = env_is('RTSC_DQ', '1') .AND. .NOT. lgamma
  END FUNCTION dq_active
  !
  !------------------------------------------------------------------------------------------------
  SUBROUTINE dq_interpolate()
    !------------------------------------------------------------------------------------------------
    !! dV_ind(q) in the u basis for all modes of the current q, from w_pot (as elphon does with
    !! ldvscf_interpolate; the w_pot units are opened and closed here, once per q)
    USE io_global,         ONLY : stdout
    USE io_files,          ONLY : prefix
    USE mp_pools,          ONLY : me_pool, root_pool
    USE fft_base,          ONLY : dfftp
    USE noncollin_module,  ONLY : nspin_mag
    USE ions_base,         ONLY : nat
    USE modes,             ONLY : u
    USE qpoint,            ONLY : xq
    USE dvscf_interpolate, ONLY : dvscf_interpol_setup, dvscf_r2q, dvscf_interpol_close, wpot_dir, &
                                  do_long_range, nrlocal, nrbase, iunwpot, lrwpot
    CHARACTER(LEN=256) :: dir, filwpot
    CHARACTER(LEN=256), EXTERNAL :: trimcheck
    INTEGER :: ios, irlocal, ir, direct_io_factor
    INTEGER(8) :: unf_lrwpot
    REAL(DP) :: dummy
    !
    CALL start_clock('dq_interp')
    CALL get_environment_variable('RTSC_DQ_WPOT', dir, STATUS=ios)
    IF (ios /= 0 .OR. LEN_TRIM(dir) == 0) CALL errore('rtsc_dq', 'RTSC_DQ=1 needs RTSC_DQ_WPOT (w_pot dir)', 1)
    wpot_dir = trimcheck(dir)
    do_long_range = .FALSE.     ! metals (B1, B2): no Born charges; dvscf_q2r.x ran without it too
    CALL dvscf_interpol_setup()
    lrwpot = 2 * dfftp%nr1x * dfftp%nr2x * dfftp%nr3x * nspin_mag
    INQUIRE (IOLENGTH=direct_io_factor) dummy
    unf_lrwpot = direct_io_factor * INT(lrwpot, KIND=8)
    IF (me_pool == root_pool) THEN
      DO irlocal = 1, nrlocal
        ir = irlocal + nrbase
#if defined(__MPI)
        WRITE(filwpot, '(a,I0,a)') TRIM(wpot_dir) // TRIM(prefix) // '.wpot.irc', ir, '.dat1'
#else
        WRITE(filwpot, '(a,I0,a)') TRIM(wpot_dir) // TRIM(prefix) // '.wpot.irc', ir, '.dat'
#endif
        iunwpot(irlocal) = 1000 + irlocal
        OPEN(iunwpot(irlocal), FILE=TRIM(filwpot), RECL=unf_lrwpot, FORM='unformatted', STATUS='old', &
             ACCESS='direct', IOSTAT=ios)
        IF (ios /= 0) CALL errore('rtsc_dq', 'cannot open w_pot file ' // TRIM(filwpot), 1)
      ENDDO
    ENDIF
    IF (ALLOCATED(dv_all)) DEALLOCATE(dv_all)
    ALLOCATE(dv_all(dfftp%nnr, nspin_mag, 3 * nat))
    CALL dvscf_r2q(xq, u, dv_all)
    CALL dvscf_interpol_close()
    dq_xq = xq
    have_q = .TRUE.
    WRITE(stdout, '(5x,"rtsc dq: dV_ind interpolated from ",a)') TRIM(wpot_dir)
    CALL stop_clock('dq_interp')
  END SUBROUTINE dq_interpolate
  !
  !------------------------------------------------------------------------------------------------
  SUBROUTINE dq_solve_irrep(npe, imode0, lrbar, iubar, dr2, dfpt_data)
    !------------------------------------------------------------------------------------------------
    !! replaces dfpt_kernel for one irrep: one Sternheimer solve with dV_in, no SCF loop
    USE io_global,         ONLY : stdout
    USE mp,                ONLY : mp_sum
    USE mp_bands,          ONLY : intra_bgrp_comm
    USE fft_base,          ONLY : dfftp, dffts
    USE cell_base,         ONLY : omega
    USE noncollin_module,  ONLY : noncolin, nspin_mag
    USE lsda_mod,          ONLY : lsda
    USE uspp,              ONLY : okvan
    USE paw_variables,     ONLY : okpaw
    USE ldaU,              ONLY : lda_plus_u
    USE two_chem,          ONLY : twochem
    USE buffers,           ONLY : save_buffer
    USE units_lr,          ONLY : iudwf, lrdwf
    USE eqv,               ONLY : dpsi
    USE qpoint,            ONLY : nksq, xq
    USE control_lr,        ONLY : convt, lnoloc
    USE dv_of_drho_lr,     ONLY : dv_of_drho
    USE apply_dpot_mod,    ONLY : apply_dpot_allocate, apply_dpot_deallocate
    USE dfpt_type,         ONLY : dfpt_data_type, dfpt_dvscfp_to_dvscfs
    USE response_kernels,  ONLY : sternheimer_kernel, sternheimer_postprocess
    USE sth_batched,       ONLY : sth_batched_cold, sth_batched_flush_end, sth_env_flag
    !
    INTEGER, INTENT(IN) :: npe, imode0, lrbar, iubar
    REAL(DP), INTENT(OUT) :: dr2
    TYPE(dfpt_data_type), INTENT(INOUT) :: dfpt_data
    !
    COMPLEX(DP), ALLOCATABLE :: vin(:,:,:), vout(:,:,:)
    REAL(DP) :: thresh, averlt, nin, nres, rel, cmax
    REAL(DP), EXTERNAL :: get_clock
    LOGICAL :: all_conv
    INTEGER :: ipert, jpert, ik, ios
    CHARACTER(LEN=32) :: s
    !
    IF (okvan .OR. okpaw .OR. noncolin .OR. lsda .OR. nspin_mag /= 1 .OR. lda_plus_u .OR. twochem .OR. lnoloc) &
      CALL errore('rtsc_dq', 'RTSC_DQ: only NC, collinear, non-magnetic, no +U/twochem/lnoloc', 1)
    CALL start_clock('dq_irrep')
    IF (.NOT. have_q .OR. ANY(ABS(xq - dq_xq) > 1.d-10)) CALL dq_interpolate()
    !
    thresh = 1.d-8
    CALL get_environment_variable('RTSC_DQ_THRESH', s, STATUS=ios)
    IF (ios == 0 .AND. LEN_TRIM(s) > 0) READ(s, *) thresh
    !
    ! dV_in: the irrep's slice of the interpolated potential, projected on the irrep (psymdvscf)
    ALLOCATE(vin(dfftp%nnr, nspin_mag, npe), vout(dfftp%nnr, nspin_mag, npe))
    DO ipert = 1, npe
      dfpt_data%dvscfp(:, :, ipert) = dv_all(:, :, imode0 + ipert)
    ENDDO
    CALL psymdvscf(dfpt_data%dvscfp, dfftp)
    vin = dfpt_data%dvscfp
    CALL dfpt_dvscfp_to_dvscfs(dfpt_data)
    !
    ! one Sternheimer solve, dpsi from 0 (engine: cold start; vanilla loop: zero iudwf records)
    CALL apply_dpot_allocate()
    dfpt_data%drhos = (0.d0, 0.d0)
    dfpt_data%dbecsum = (0.d0, 0.d0)
    IF (sth_env_flag('RTSC_STH')) THEN
      CALL sth_batched_cold(.TRUE.)
    ELSE
      dpsi = (0.d0, 0.d0)
      DO ipert = 1, npe
        DO ik = 1, nksq
          CALL save_buffer(dpsi, lrdwf, iudwf, (ipert - 1) * nksq + ik)
        ENDDO
      ENDDO
    ENDIF
    CALL sternheimer_kernel(.FALSE., .FALSE., npe, lrbar, iubar, thresh, dfpt_data%dvscfs, all_conv, &
                            averlt, dfpt_data%drhos, dfpt_data%dbecsum)
    CALL sth_batched_cold(.FALSE.)
    CALL sternheimer_postprocess(1, dfpt_data)
    CALL psymdvscf(dfpt_data%drhos, dffts)
    CALL psymdvscf(dfpt_data%drhop, dfftp)
    !
    ! dV_out = Hxc response to dn_out
    DO ipert = 1, npe
      vout(:, :, ipert) = dfpt_data%drhop(:, :, ipert)
      IF (ALLOCATED(dfpt_data%drhoc)) THEN
        CALL dv_of_drho(vout(:, :, ipert), drhoc = dfpt_data%drhoc(:, ipert))
      ELSE
        CALL dv_of_drho(vout(:, :, ipert))
      ENDIF
    ENDDO
    !
    ! residual meter and the stationary correction C(i,j) = - int conj(dn_out_i) (dV_in_j - dV_out_j)
    IF (ALLOCATED(corr)) DEALLOCATE(corr)
    ALLOCATE(corr(npe, npe))
    dr2 = 0.d0
    DO jpert = 1, npe
      nin = REAL(DOT_PRODUCT(vin(:, 1, jpert), vin(:, 1, jpert)), DP)
      nres = REAL(DOT_PRODUCT(vin(:, 1, jpert) - vout(:, 1, jpert), vin(:, 1, jpert) - vout(:, 1, jpert)), DP)
      CALL mp_sum(nin, intra_bgrp_comm)
      CALL mp_sum(nres, intra_bgrp_comm)
      rel = SQRT(nres / MAX(nin, 1.d-300))
      dr2 = dr2 + nres
      WRITE(stdout, '(5x,"rtsc dq: mode ",I4,"  |dV_out - dV_in| / |dV_in| = ",ES10.3)') imode0 + jpert, rel
      DO ipert = 1, npe
        corr(ipert, jpert) = - DOT_PRODUCT(dfpt_data%drhop(:, 1, ipert), vin(:, 1, jpert) - vout(:, 1, jpert)) &
                             * omega / REAL(dfftp%nr1 * dfftp%nr2 * dfftp%nr3, DP)
      ENDDO
    ENDDO
    CALL mp_sum(corr, intra_bgrp_comm)
    ! as mix_potential + dfpt_kernel print it: (|res| / ndim)^2 / npert, ndim = 2 nnr npert
    dr2 = dr2 / (2.d0 * dfftp%nr1 * dfftp%nr2 * dfftp%nr3 * npe)**2 / npe
    IF (env_is('RTSC_DQ_VAR', '0')) corr = (0.d0, 0.d0)
    corr_i0 = imode0
    corr_npe = npe
    cmax = MAXVAL(ABS(corr))
    WRITE(stdout, '(/,5x," iter #   1 total cpu time :",f8.1," secs   av.it.: ",f5.1)') get_clock('PHONON'), averlt
    WRITE(stdout, '(5x," thresh=",es10.3," one-shot (rtsc dq) |ddv_scf|^2 = ",es10.3,"  max|C| = ",es10.3)') &
         thresh, dr2, cmax
    IF (.NOT. all_conv) WRITE(stdout, '(5x,"rtsc dq: Sternheimer not converged at all k")')
    !
    ! dvscf of this irrep (written by solve_linter for EPW, and used nowhere else): dV_in
    dfpt_data%dvscfp = vin
    CALL dfpt_dvscfp_to_dvscfs(dfpt_data)
    convt = .TRUE.
    !
    CALL sth_batched_flush_end()
    CALL apply_dpot_deallocate()
    DEALLOCATE(vin, vout)
    CALL stop_clock('dq_irrep')
  END SUBROUTINE dq_solve_irrep
  !
  !------------------------------------------------------------------------------------------------
  SUBROUTINE dq_add_dyn(imode0, npe)
    !------------------------------------------------------------------------------------------------
    !! add the stationary correction of this irrep to dyn and dyn_rec (after drhodvus, which resets
    !! dyn_rec; before drhodv, which adds the Hellmann-Feynman part)
    USE dynmat, ONLY : dyn, dyn_rec
    INTEGER, INTENT(IN) :: imode0, npe
    INTEGER :: i, j
    IF (.NOT. ALLOCATED(corr) .OR. corr_i0 /= imode0 .OR. corr_npe /= npe) &
      CALL errore('rtsc_dq', 'dq_add_dyn: no correction for this irrep', 1)
    DO j = 1, npe
      DO i = 1, npe
        dyn(imode0 + i, imode0 + j) = dyn(imode0 + i, imode0 + j) + corr(i, j)
        dyn_rec(imode0 + i, imode0 + j) = dyn_rec(imode0 + i, imode0 + j) + corr(i, j)
      ENDDO
    ENDDO
    DEALLOCATE(corr)
    corr_i0 = -1
  END SUBROUTINE dq_add_dyn
  !
  !------------------------------------------------------------------------------------------------
  SUBROUTINE dq_end_q()
    IF (ALLOCATED(dv_all)) DEALLOCATE(dv_all)
    IF (ALLOCATED(corr)) DEALLOCATE(corr)
    have_q = .FALSE.
    dq_xq = 1.d10
  END SUBROUTINE dq_end_q
  !
END MODULE rtsc_dq
