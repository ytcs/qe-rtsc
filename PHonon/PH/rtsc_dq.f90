!
! rtsc: one-shot DFPT pass from an interpolated dvscf (groups/compute/DQ-LOG.md).
!
! RTSC_DQ=1, RTSC_DQ_WPOT=<dir>: at every q /= Gamma, solve_linter calls dq_solve_irrep instead of the
! self-consistent dfpt_kernel. The induced potential dV_in of the irrep is Fourier-interpolated
! (dvscf_r2q, from the w_pot that dvscf_q2r.x wrote for a coarser q-grid), its G-sphere part symmetrized,
! and one non-self-consistent Sternheimer solve gives dpsi and dn_out (on the batched engine with
! RTSC_STH=1: one engine call per irrep, CG from dpsi = 0, no mixing). Then:
!  - dvscf written for EPW: dV_in (the interpolated, symmetrized potential);
!  - dynamical matrix: QE's drhodv/drhodvloc from dpsi and dn_out (Hellmann-Feynman form, error linear
!    in dV_in - dV*), plus the stationary correction
!       C(i,j) = - int conj(dn_out_i) (dV_in_j - dV_out_j),   dV_out = Hxc response to dn_out,
!    for every pair of modes of the q (dq_finish_dyn, after phqscf: irreps of the same symmetry
!    type are coupled in dyn), which makes dyn stationary: its error is bilinear in the errors of
!    dn_out (RTSC_DQ_VAR=0 leaves it out). Hermitized downstream (symdynph_gq).
!  - printed per irrep: |dV_out - dV_in| / |dV_in| (the interpolation-quality meter) and max |C|.
! Gamma runs the normal SCF (metals need the Fermi shift there; Gamma is on every subgrid).
! RTSC_DQ_THRESH: Sternheimer threshold of the pass (default 1e-8 = 0.1 sqrt(tr2_ph = 1e-14)).
! RTSC_DQ_NITER=N (default 1): N passes, Broyden-mixed as dfpt_kernel, the last one as above.
!
MODULE rtsc_dq
  USE kinds, ONLY : DP
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: dq_active, dq_solve_irrep, dq_finish_dyn, dq_end_q
  !
  LOGICAL, SAVE :: have_q = .FALSE.
  REAL(DP), SAVE :: dq_xq(3) = 1.d10
  COMPLEX(DP), ALLOCATABLE, SAVE :: dv_all(:,:,:)  ! (dfftp%nnr, nspin_mag, 3 nat): interpolated dV_ind, u basis
  COMPLEX(DP), ALLOCATABLE, SAVE :: dn_all(:,:)    ! (dfftp%nnr, 3 nat): dn_out of every mode (last pass)
  COMPLEX(DP), ALLOCATABLE, SAVE :: r_all(:,:)     ! (dfftp%nnr, 3 nat): dV_in - dV_out of every mode
  LOGICAL, ALLOCATABLE, SAVE :: done_mode(:)
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
    USE ions_base,         ONLY : nat
    USE control_lr,        ONLY : convt, lnoloc, alpha_mix, nmix_ph, tr2_ph
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
    COMPLEX(DP), ALLOCATABLE :: vin(:,:,:), vout(:,:,:), vmix(:,:,:)
    REAL(DP) :: thresh, averlt, nin, nres, nout, dr2m
    LOGICAL :: conv
    REAL(DP), EXTERNAL :: get_clock
    LOGICAL :: all_conv
    INTEGER :: ipert, jpert, ik, ios, it, nit
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
    ALLOCATE(vin(dfftp%nnr, nspin_mag, npe), vout(dfftp%nnr, nspin_mag, npe), vmix(dfftp%nnr, nspin_mag, npe))
    DO ipert = 1, npe
      dfpt_data%dvscfp(:, :, ipert) = dv_all(:, :, imode0 + ipert)
    ENDDO
    IF (.NOT. env_is('RTSC_DQ_NOSYM', '1')) CALL psym_sphere(dfpt_data%dvscfp, npe)   ! NOSYM: diagnostic
    CALL dfpt_dvscfp_to_dvscfs(dfpt_data)
    !
    ! RTSC_DQ_NITER passes (default 1): pass 1 solves from dpsi = 0 (engine: cold start; vanilla loop:
    ! zero iudwf records), later passes warm-start from the previous dpsi after a Broyden step
    ! (mix_potential, as dfpt_kernel). The last pass gives dpsi, dn_out, the correction and the dvscf.
    nit = 1
    CALL get_environment_variable('RTSC_DQ_NITER', s, STATUS=ios)
    IF (ios == 0 .AND. LEN_TRIM(s) > 0) READ(s, *) nit
    CALL apply_dpot_allocate()
    IF (.NOT. sth_env_flag('RTSC_STH')) THEN
      dpsi = (0.d0, 0.d0)
      DO ipert = 1, npe
        DO ik = 1, nksq
          CALL save_buffer(dpsi, lrdwf, iudwf, (ipert - 1) * nksq + ik)
        ENDDO
      ENDDO
    ENDIF
    DO it = 1, nit
      vin = dfpt_data%dvscfp
      dfpt_data%drhos = (0.d0, 0.d0)
      dfpt_data%dbecsum = (0.d0, 0.d0)
      IF (it == 1 .AND. sth_env_flag('RTSC_STH')) CALL sth_batched_cold(.TRUE.)
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
      ! residual meter (per mode) and dr2 as mix_potential + dfpt_kernel print it
      dr2 = 0.d0
      DO jpert = 1, npe
        nin = REAL(DOT_PRODUCT(vin(:, 1, jpert), vin(:, 1, jpert)), DP)
        nres = REAL(DOT_PRODUCT(vin(:, 1, jpert) - vout(:, 1, jpert), vin(:, 1, jpert) - vout(:, 1, jpert)), DP)
        CALL mp_sum(nin, intra_bgrp_comm)
        CALL mp_sum(nres, intra_bgrp_comm)
        dr2 = dr2 + nres
        nout = REAL(DOT_PRODUCT(vout(:, 1, jpert), vout(:, 1, jpert)), DP)
        CALL mp_sum(nout, intra_bgrp_comm)
        WRITE(stdout, '(5x,"rtsc dq: pass ",I0," mode ",I4,"  |dV_out - dV_in| / |dV_in| = ",ES10.3, &
             &"  |dV_out| / |dV_in| = ",F8.4)') it, imode0 + jpert, SQRT(nres / MAX(nin, 1.d-300)), &
             SQRT(nout / MAX(nin, 1.d-300))
      ENDDO
      dr2 = dr2 / (2.d0 * dfftp%nr1 * dfftp%nr2 * dfftp%nr3 * npe)**2 / npe
      WRITE(stdout, '(/,5x," iter # ",i3," total cpu time :",f8.1," secs   av.it.: ",f5.1)') it, &
           get_clock('PHONON'), averlt
      WRITE(stdout, '(5x," thresh=",es10.3," rtsc dq pass |ddv_scf|^2 = ",es10.3)') thresh, dr2
      IF (.NOT. all_conv) WRITE(stdout, '(5x,"rtsc dq: Sternheimer not converged at all k")')
      IF (env_is('RTSC_DQ_SYMTEST', '1')) CALL symtest(vout, 'psym(dV_out) vs dV_out')
      IF (env_is('RTSC_DQ_SYMTEST', '1')) CALL symtest(vin, 'psym(dV_in) vs dV_in')
      IF (env_is('RTSC_DQ_SYMTEST', '1')) CALL symtest(dfpt_data%drhop, 'psym(dn_out) vs dn_out')
      IF (it == nit) EXIT
      vmix = vout
      CALL mix_potential(2 * dfftp%nnr * nspin_mag * npe, vmix, dfpt_data%dvscfp, alpha_mix(it), dr2m, &
                         npe * tr2_ph, it, nmix_ph, 'dqmix', conv)
      CALL dfpt_dvscfp_to_dvscfs(dfpt_data)
    ENDDO
    !
    ! keep dn_out and the residual dV_in - dV_out of this irrep's modes for the stationary correction,
    ! which couples every pair of modes of the q (irreps of the same symmetry type are coupled in dyn)
    IF (.NOT. ALLOCATED(dn_all)) THEN
      ALLOCATE(dn_all(dfftp%nnr, 3 * nat), r_all(dfftp%nnr, 3 * nat), done_mode(3 * nat))
      done_mode = .FALSE.
    ENDIF
    DO ipert = 1, npe
      dn_all(:, imode0 + ipert) = dfpt_data%drhop(:, 1, ipert)
      r_all(:, imode0 + ipert) = vin(:, 1, ipert) - vout(:, 1, ipert)
      done_mode(imode0 + ipert) = .TRUE.
    ENDDO
    WRITE(stdout, '(5x,"rtsc dq: ",I0," pass(es)")') nit
    !
    ! dvscf of this irrep (written by solve_linter for EPW, and used nowhere else): the dV_in of
    ! the last pass
    dfpt_data%dvscfp = vin
    CALL dfpt_dvscfp_to_dvscfs(dfpt_data)
    convt = .TRUE.
    !
    CALL sth_batched_flush_end()
    CALL apply_dpot_deallocate()
    DEALLOCATE(vin, vout, vmix)
    CALL stop_clock('dq_irrep')
  END SUBROUTINE dq_solve_irrep
  !
  !------------------------------------------------------------------------------------------------
  SUBROUTINE psym_sphere(v, npe)
    !------------------------------------------------------------------------------------------------
    !! psymdvscf on the G-sphere part of v only. psymdvscf multiplies by exp(i G r) for the
    !! operations with S q = q + G (G /= 0, e.g. at K); on the FFT grid that wraps components near the
    !! box edge, so it is a projector only for functions inside the sphere (as dn is). dV_ind has
    !! components outside it (xc in real space; B1 K point: psymdvscf moved the exact dV_ind by
    !! 1-10 %). The outside part is kept as interpolated.
    USE fft_base,         ONLY : dfftp
    USE fft_interfaces,   ONLY : fwfft, invfft
    USE gvect,            ONLY : ngm
    USE noncollin_module, ONLY : nspin_mag
    INTEGER, INTENT(IN) :: npe
    COMPLEX(DP), INTENT(INOUT) :: v(dfftp%nnr, nspin_mag, npe)
    COMPLEX(DP), ALLOCATABLE :: a(:,:,:), aux(:)
    INTEGER :: ip, is
    ALLOCATE(a(dfftp%nnr, nspin_mag, npe), aux(dfftp%nnr))
    DO ip = 1, npe
      DO is = 1, nspin_mag
        aux = v(:, is, ip)
        CALL fwfft('Rho', aux, dfftp)
        a(:, is, ip) = (0.d0, 0.d0)
        a(dfftp%nl(1:ngm), is, ip) = aux(dfftp%nl(1:ngm))
        CALL invfft('Rho', a(:, is, ip), dfftp)
      ENDDO
    ENDDO
    v = v - a                  ! outside the sphere
    CALL psymdvscf(a, dfftp)   ! inside: symmetrize
    v = v + a
    DEALLOCATE(a, aux)
  END SUBROUTINE psym_sphere
  !
  !------------------------------------------------------------------------------------------------
  SUBROUTINE symtest(v, what)
    !! diagnostic (RTSC_DQ_SYMTEST=1): how much psymdvscf changes v, per perturbation
    USE io_global,        ONLY : stdout
    USE fft_base,         ONLY : dfftp
    USE noncollin_module, ONLY : nspin_mag
    USE lr_symm_base,     ONLY : lr_npert
    COMPLEX(DP), INTENT(IN) :: v(dfftp%nnr, nspin_mag, lr_npert)
    CHARACTER(LEN=*), INTENT(IN) :: what
    COMPLEX(DP), ALLOCATABLE :: w(:,:,:)
    INTEGER :: ip
    ALLOCATE(w(dfftp%nnr, nspin_mag, lr_npert))
    w = v
    CALL psymdvscf(w, dfftp)
    DO ip = 1, lr_npert
      WRITE(stdout, '(5x,"rtsc dq symtest: ",a," pert ",I0,": ",ES10.3)') what, ip, &
           SQRT(SUM(ABS(w(:, 1, ip) - v(:, 1, ip))**2) / SUM(ABS(v(:, 1, ip))**2))
    ENDDO
    DEALLOCATE(w)
  END SUBROUTINE symtest
  !
  !------------------------------------------------------------------------------------------------
  SUBROUTINE dq_finish_dyn()
    !------------------------------------------------------------------------------------------------
    !! after phqscf (all irreps of this q): add the stationary correction to dyn (pattern basis),
    !!    C(i,j) = - int conj(dn_out_i) (dV_in_j - dV_out_j)   for all modes i, j,
    !! so that the Hermitized dyn is stationary (error bilinear in the dn_out errors). The
    !! Hellmann-Feynman part (drhodv) fills the same rows i against all columns j.
    USE io_global, ONLY : stdout
    USE mp,        ONLY : mp_sum
    USE mp_bands,  ONLY : intra_bgrp_comm
    USE fft_base,  ONLY : dfftp
    USE cell_base, ONLY : omega
    USE ions_base, ONLY : nat
    USE dynmat,    ONLY : dyn
    COMPLEX(DP), ALLOCATABLE :: c(:,:)
    INTEGER :: i, j
    IF (.NOT. dq_active()) RETURN
    IF (.NOT. ALLOCATED(dn_all)) RETURN
    IF (.NOT. ALL(done_mode)) CALL errore('rtsc_dq', 'dq_finish_dyn: not every mode of this q was solved', 1)
    ALLOCATE(c(3 * nat, 3 * nat))
    DO j = 1, 3 * nat
      DO i = 1, 3 * nat
        c(i, j) = - DOT_PRODUCT(dn_all(:, i), r_all(:, j)) * omega / REAL(dfftp%nr1 * dfftp%nr2 * dfftp%nr3, DP)
      ENDDO
    ENDDO
    CALL mp_sum(c, intra_bgrp_comm)
    IF (env_is('RTSC_DQ_VAR', '0')) c = (0.d0, 0.d0)
    dyn = dyn + c
    WRITE(stdout, '(5x,"rtsc dq: stationary correction added to dyn, max|C| = ",ES10.3)') MAXVAL(ABS(c))
    DEALLOCATE(c)
  END SUBROUTINE dq_finish_dyn
  !
  !------------------------------------------------------------------------------------------------
  SUBROUTINE dq_end_q()
    IF (ALLOCATED(dv_all)) DEALLOCATE(dv_all)
    IF (ALLOCATED(dn_all)) DEALLOCATE(dn_all, r_all, done_mode)
    have_q = .FALSE.
    dq_xq = 1.d10
  END SUBROUTINE dq_end_q
  !
END MODULE rtsc_dq
