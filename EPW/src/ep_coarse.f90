  !
  ! Copyright (C) 2023-2026 EPW-Collaboration
  ! Copyright (C) 2016-2023 EPW-Collaboration
  ! Copyright (C) 2010-2016 Samuel Ponce', Roxana Margine, Carla Verdi, Feliciano Giustino
  ! Copyright (C) 2007-2009 Jesse Noffsinger, Brad Malone, Feliciano Giustino
  !
  ! This file is distributed under the terms of the GNU General Public
  ! License. See the file `LICENSE' in the root directory of the
  ! present distribution, or http://www.gnu.org/copyleft.gpl.txt .
  !
  !
  !----------------------------------------------------------------------
  MODULE ep_coarse
  !----------------------------------------------------------------------
  !!
  !! This module contains routines for the EPC computed on the coarse grid.
  !!
  USE kinds, ONLY : DP
  IMPLICIT NONE
  !
  ! rtsc epw3 (RTSC_EPW_STAR=1; groups/compute/EPW3-LOG.md): star-of-q shortcut. The coarse g of the first q0 of a
  ! star (identity op) is kept for every k (star_g0, pattern basis, all modes). For another member S q0, EPW evaluates
  ! g = <R psi_{k+Sq0}| dV_q0 |R psi_k> with R psi_k at k~ = S k (and R psi_{k+Sq0} at k~ + q0); this is
  ! g = D_kq^H g0(k~) D_k with D_k = <psi_stored(k~)| R psi_k> (kept bands), exact when D is unitary. A k whose D
  ! misses unitarity by more than star_tol (degenerate multiplet cut by the band set) is computed directly.
  ! RTSC_EPW_STAR_CHECK=1: the first 2 k of every member are also computed directly and compared.
  LOGICAL, SAVE, PRIVATE :: star_read = .FALSE., star_req = .FALSE., star_chk = .FALSE.
  LOGICAL, SAVE, PRIVATE :: star_now = .FALSE., star_have = .FALSE.
  COMPLEX(KIND = DP), ALLOCATABLE, SAVE, PRIVATE :: star_g0(:, :, :, :)
  INTEGER, ALLOCATABLE, SAVE, PRIVATE :: star_pos(:)
  INTEGER, SAVE, PRIVATE :: star_nsc = 0, star_nfb = 0, star_nck = 0
  REAL(KIND = DP), SAVE, PRIVATE :: star_umax = 0.0_DP, star_cmax = 0.0_DP
  REAL(KIND = DP), PARAMETER, PRIVATE :: star_tol = 1.0E-8_DP
  INTEGER, SAVE, PRIVATE :: star_iqirr = 0, star_nqcirr = 0
  !
  CONTAINS
    !
    !-----------------------------------------------------------------------
    LOGICAL FUNCTION star_enabled()
    !-----------------------------------------------------------------------
    !! rtsc epw3: RTSC_EPW_STAR=1 and the all-modes GPU path of ep_coarse_compute applies
    CHARACTER(LEN = 16) :: ev
    IF (.NOT. star_read) THEN
      CALL GET_ENVIRONMENT_VARIABLE('RTSC_EPW_STAR', ev)
      star_req = TRIM(ev) == '1'
      CALL GET_ENVIRONMENT_VARIABLE('RTSC_EPW_STAR_CHECK', ev)
      star_chk = star_req .AND. TRIM(ev) == '1'
      star_read = .TRUE.
    ENDIF
    star_enabled = star_req
    IF (star_enabled) star_enabled = ep_coarse_gpu_ok()
    END FUNCTION star_enabled
    !
    !-----------------------------------------------------------------------
    SUBROUTINE star_begin(first, use_now, iq_irr, nqc_irr)
    !-----------------------------------------------------------------------
    !! before ep_coarse_compute of a star member: first = the base member (q0, identity op) of a new star;
    !! use_now = the shortcut may be used in this call; iq_irr, nqc_irr: for a deferred dvscf_read
    LOGICAL, INTENT(IN) :: first, use_now
    INTEGER, INTENT(IN) :: iq_irr, nqc_irr
    IF (first) star_have = .FALSE.
    star_now = use_now .AND. star_have
    star_iqirr = iq_irr
    star_nqcirr = nqc_irr
    END SUBROUTINE star_begin
    !
    !-----------------------------------------------------------------------
    LOGICAL FUNCTION star_lazy()
    !-----------------------------------------------------------------------
    !! the member's dvscf is read (and the GPU potentials set up) only if some k needs the direct form
    !! (RTSC_EPW_STAR_CHECK=1 computes 2 k directly, so it reads up front)
    star_lazy = star_now .AND. .NOT. star_chk
    END FUNCTION star_lazy
    !
    !-----------------------------------------------------------------------
    SUBROUTINE star_store(el_ph_mat, nm)
    !-----------------------------------------------------------------------
    !! after the base member: keep its g (pattern basis) for every k of the grid (all pools)
    USE global_var,  ONLY : nbndep, nk_loc, nkpts
    USE mp_global,   ONLY : inter_pool_comm
    USE mp,          ONLY : mp_sum
    USE parallelism, ONLY : kpointdivision
    INTEGER, INTENT(IN) :: nm
    COMPLEX(KIND = DP), INTENT(IN) :: el_ph_mat(nbndep, nbndep, nk_loc, nm)
    INTEGER :: ik0, ik, nu
    star_now = .FALSE.
    IF (ALLOCATED(star_g0)) THEN
      IF (SIZE(star_g0, 1) /= nbndep .OR. SIZE(star_g0, 3) /= nm .OR. SIZE(star_g0, 4) /= nkpts) DEALLOCATE(star_g0)
    ENDIF
    IF (.NOT. ALLOCATED(star_g0)) ALLOCATE(star_g0(nbndep, nbndep, nm, nkpts))
    star_g0 = (0.0_DP, 0.0_DP)
    CALL kpointdivision(ik0)
    DO ik = 1, nk_loc
      DO nu = 1, nm
        star_g0(:, :, nu, ik + ik0) = el_ph_mat(:, :, ik, nu)
      ENDDO
    ENDDO
    CALL mp_sum(star_g0, inter_pool_comm)
    star_have = .TRUE.
    END SUBROUTINE star_store
    !
    !-----------------------------------------------------------------------
    SUBROUTINE star_end()
    !-----------------------------------------------------------------------
    star_now = .FALSE.
    END SUBROUTINE star_end
    !
    !-----------------------------------------------------------------------
    INTEGER FUNCTION star_lin(m)
    !-----------------------------------------------------------------------
    USE fft_base, ONLY : dffts
    INTEGER, INTENT(IN) :: m(3)
    star_lin = 1 + MODULO(m(1), dffts%nr1) + MODULO(m(2), dffts%nr2) * dffts%nr1 + &
               MODULO(m(3), dffts%nr3) * dffts%nr1 * dffts%nr2
    END FUNCTION star_lin
    !
    !-----------------------------------------------------------------------
    SUBROUTINE star_d(kv, n, igl, a, d, nabs, ok)
    !-----------------------------------------------------------------------
    !! d = <psi_stored(kv)| a> over the kept bands; a: nbndep rotated states at the k-vector kv (cart.) with
    !! plane waves igl(1:n); nabs = absolute index of the stored grid point kv mod G; ok = D unitary
    USE global_var,       ONLY : nbndep, igk_k_all, ngk_all, ibndkept
    USE input,            ONLY : xk_all
    USE kfold,            ONLY : ktokpmq
    USE io,               ONLY : readwfc
    USE wvfct,            ONLY : npwx
    USE pwcom,            ONLY : nbnd
    USE noncollin_module, ONLY : npol
    USE gvect,            ONLY : mill
    USE fft_base,         ONLY : dffts
    USE cell_base,        ONLY : at
    USE ep_constants,     ONLY : czero, cone
    REAL(KIND = DP), INTENT(IN) :: kv(3)
    INTEGER, INTENT(IN) :: n, igl(n)
    COMPLEX(KIND = DP), INTENT(IN) :: a(npwx * npol, nbndep)
    COMPLEX(KIND = DP), INTENT(OUT) :: d(nbndep, nbndep)
    INTEGER, INTENT(OUT) :: nabs
    LOGICAL, INTENT(OUT) :: ok
    REAL(KIND = DP) :: z3(3), v(3), e
    INTEGER :: ip, nl, ng, j, ig, ib, m3(3), g1(3)
    COMPLEX(KIND = DP), ALLOCATABLE :: evt(:, :), p(:, :), u(:, :)
    ok = .FALSE.
    z3 = 0.0_DP
    CALL ktokpmq(z3, kv, +1, ip, nl, nabs)
    v = kv - xk_all(:, nabs)
    CALL cryst_to_cart(1, v, at, -1)
    g1 = NINT(v)
    IF (ANY(ABS(v - g1) > 1.0E-6_DP)) RETURN
    ng = ngk_all(nabs)
    IF (ng /= n) RETURN
    IF (.NOT. ALLOCATED(star_pos)) THEN
      ALLOCATE(star_pos(dffts%nr1 * dffts%nr2 * dffts%nr3))
      star_pos = 0
    ENDIF
    ALLOCATE(evt(npwx * npol, nbnd), p(n, nbndep))
    CALL readwfc(ip, nl, evt)
    DO j = 1, ng
      star_pos(star_lin(mill(:, igk_k_all(j, nabs)))) = j
    ENDDO
    ok = .TRUE.
    DO ig = 1, n
      m3 = mill(:, igl(ig)) + g1
      j = star_pos(star_lin(m3))
      IF (j == 0) THEN
        ok = .FALSE.
        EXIT
      ENDIF
      IF (ANY(mill(:, igk_k_all(j, nabs)) /= m3)) THEN
        ok = .FALSE.
        EXIT
      ENDIF
      DO ib = 1, nbndep
        p(ig, ib) = evt(j, ibndkept(ib))
      ENDDO
    ENDDO
    DO j = 1, ng
      star_pos(star_lin(mill(:, igk_k_all(j, nabs)))) = 0
    ENDDO
    IF (ok) THEN
      CALL ZGEMM('C', 'N', nbndep, nbndep, n, cone, p, n, a, npwx * npol, czero, d, nbndep)
      ALLOCATE(u(nbndep, nbndep))
      CALL ZGEMM('C', 'N', nbndep, nbndep, nbndep, cone, d, nbndep, d, nbndep, czero, u, nbndep)
      DO ib = 1, nbndep
        u(ib, ib) = u(ib, ib) - cone
      ENDDO
      e = MAXVAL(ABS(u))
      star_umax = MAX(star_umax, e)
      ok = e <= star_tol
      DEALLOCATE(u)
    ENDIF
    DEALLOCATE(evt, p)
    END SUBROUTINE star_d
    !
    !-----------------------------------------------------------------------
    SUBROUTINE star_g(sxk, xkqv, npw, igk, a5, npwq, igkq, a4, nm, gout, ok)
    !-----------------------------------------------------------------------
    !! gout(:, :, nu) = D_kq^H g0(:, :, nu, k~) D_k for the current member (see the module header)
    USE global_var,       ONLY : nbndep
    USE wvfct,            ONLY : npwx
    USE noncollin_module, ONLY : npol
    USE ep_constants,     ONLY : czero, cone
    REAL(KIND = DP), INTENT(IN) :: sxk(3), xkqv(3)
    INTEGER, INTENT(IN) :: npw, npwq, igk(npw), igkq(npwq), nm
    COMPLEX(KIND = DP), INTENT(IN) :: a5(npwx * npol, nbndep), a4(npwx * npol, nbndep)
    COMPLEX(KIND = DP), INTENT(OUT) :: gout(nbndep, nbndep, nm)
    LOGICAL, INTENT(OUT) :: ok
    COMPLEX(KIND = DP), ALLOCATABLE :: dk(:, :), dq(:, :), t(:, :)
    INTEGER :: nkt, nkqt, nu
    LOGICAL :: okq
    ok = .FALSE.
    IF (.NOT. star_have) RETURN
    ALLOCATE(dk(nbndep, nbndep), dq(nbndep, nbndep), t(nbndep, nbndep))
    CALL star_d(sxk, npw, igk, a5, dk, nkt, ok)
    IF (ok) CALL star_d(xkqv, npwq, igkq, a4, dq, nkqt, okq)
    ok = ok .AND. okq
    IF (ok) THEN
      DO nu = 1, nm
        CALL ZGEMM('N', 'N', nbndep, nbndep, nbndep, cone, star_g0(1, 1, nu, nkt), nbndep, dk, nbndep, czero, &
                   t, nbndep)
        CALL ZGEMM('C', 'N', nbndep, nbndep, nbndep, cone, dq, nbndep, t, nbndep, czero, gout(1, 1, nu), nbndep)
      ENDDO
    ENDIF
    DEALLOCATE(dk, dq, t)
    END SUBROUTINE star_g
    !
    !-----------------------------------------------------------------------
    SUBROUTINE dvscf_read(imode0, iq_irr, nqc_irr, npe, xq0, dvscfins, timerev)
    !-----------------------------------------------------------------------
    !!
    !! Electron-phonon calculation from data saved in fildvscf
    !! Shuffle2 mode (shuffle on electrons + load all phonon q's)
    !!
    !! When double_grid == .TRUE. (ecutrho > ecutwfc * 4), the dvscf file
    !! contains data on the dense grid (dfftp). Here we read the data and use it. 
    !! But we return only the smooth part (dffts) by applying fft_interpolate, 
    !! because only the smooth part is needed afterwards in ep_coarse_compute.
    !!
    !! RM - Nov/Dec 2014
    !! Imported the noncolinear case implemented by xlzhang
    !!
    !! Roxana Margine - Jan 2019: Updated based on QE 6.3 for US
    !! HL - Mar 2020: PAW added based on QE v6.5
    !! SP - This routine cannot be placed in io/ due to cyclic dependence.
    !!
    !! Wooil Yang - Nov 2025: Added reading of dnsscf when using DFT+U.
    !!
    !-----------------------------------------------------------------------
    !
    USE kinds,            ONLY : DP
    USE mp,               ONLY : mp_barrier, mp_bcast
    USE mp_global,        ONLY : ionode_id
    USE mp_pools,         ONLY : my_pool_id, inter_pool_comm, root_pool
    USE ions_base,        ONLY : nat
    USE gvecs,            ONLY : doublegrid
    USE lrus,             ONLY : int3, int3_nc, int3_paw
    USE uspp,             ONLY : okvan
    USE paw_variables,    ONLY : okpaw
    USE lsda_mod,         ONLY : nspin
    USE fft_base,         ONLY : dfftp, dffts
    USE modes,            ONLY : nmodes
    USE io,               ONLY : readdvscf, readint3paw, readdnsscf
    USE ldaU_lr,          ONLY : dnsscf
    USE ldaU_ph,          ONLY : dnsscf_all_modes
    USE ldaU,             ONLY : lda_plus_u, Hubbard_lmax
    USE uspp_param,       ONLY : nhm
    USE ep_constants,     ONLY : czero, cone
    USE fft_interfaces,   ONLY : fft_interpolate
    USE noncollin_module, ONLY : nspin_mag, noncolin
    USE dvqpsi,           ONLY : newdq2
    !
    IMPLICIT NONE
    !
    INTEGER, INTENT(in) :: imode0
    !! Current mode number
    INTEGER, INTENT(in) :: iq_irr
    !! Current ireducible q-point
    INTEGER, INTENT(in) :: nqc_irr
    !! Total number of irreducible q-points in the list
    INTEGER, INTENT(in) :: npe
    !! Number of perturbations for this irr representation
    REAL(KIND = DP), INTENT(in) :: xq0(3)
    !! The first q-point in the star (cartesian coords.)
    COMPLEX(KIND = DP), INTENT(inout) :: dvscfins(dffts%nnr, nspin_mag, npe)
    !! Change of the scf potential on the smooth FFT grid
    LOGICAL, INTENT(in) :: timerev
    !!  true if we are using time reversal
    !
    ! Local variables
    INTEGER :: ipert
    !! Change of Vscf due to perturbations
    INTEGER :: is
    !! Counter on spin
    INTEGER :: ierr
    !! Error status
    COMPLEX(KIND = DP), ALLOCATABLE :: dvscfin(:, :, :)
    !! Change of the scf potential on the dense FFT grid
    !
    CALL start_clock('dvscf_read')
    !
    IF (okvan) THEN
      ALLOCATE(int3(nhm, nhm, nat, nspin_mag, npe), STAT = ierr)
      IF (ierr /= 0) CALL errore('dvscf_read', 'Error allocating int3', 1)
      IF (okpaw) THEN
        ALLOCATE(int3_paw(nhm, nhm, nat, nspin_mag, npe), STAT = ierr)
        IF (ierr /= 0) CALL errore('dvscf_read', 'Error allocating int3_paw', 1)
      ENDIF
      IF (noncolin) THEN
        ALLOCATE(int3_nc(nhm, nhm, nat, nspin, npe), STAT = ierr)
        IF (ierr /= 0) CALL errore('dvscf_read', 'Error allocating int3_nc', 1)
      ENDIF
    ENDIF
    !
    IF (lda_plus_u) THEN
      ALLOCATE(dnsscf_all_modes(2 * Hubbard_lmax + 1, 2 * Hubbard_lmax + 1, nspin_mag, nat, nmodes), STAT = ierr)
      IF (ierr /= 0) CALL errore('dvscf_read', 'Error allocating dnsscf_all_modes', 1)
      ALLOCATE(dnsscf(2 * Hubbard_lmax + 1, 2 * Hubbard_lmax + 1, nspin_mag, nat, npe), STAT = ierr)
      IF (ierr /= 0) CALL errore('dvscf_read', 'Error allocating dnsscf', 1)
    ENDIF
    !
    ! read the <prefix>.dvscf_q[iq] files
    !
    ALLOCATE(dvscfin(dfftp%nnr, nspin_mag, npe), STAT = ierr)
    IF (ierr /= 0) CALL errore('dvscf_read', 'Error allocating dvscfin', 1)
    !
    dvscfin = czero
    IF (okpaw) int3_paw = czero
    IF (lda_plus_u) THEN
      dnsscf_all_modes = czero
      dnsscf = czero
    ENDIF
    !
    !! 2020.03 HL
    !! The part below should be changed accordingly when parallelization over G vectors is implemented.
    !
    IF (my_pool_id == ionode_id) THEN
      DO ipert = 1, npe
        CALL readdvscf(dvscfin(:, :, ipert), imode0 + ipert, iq_irr, nqc_irr)
        IF (okpaw) CALL readint3paw(int3_paw(:, :, :, :, ipert), imode0 + ipert, iq_irr, nqc_irr)
      ENDDO
      IF (lda_plus_u) THEN
        !! The dnsscf file contains the data for all perturbations, and data for 
        !! a single perturbation cannot be read individually.
        !! Therefore, we read the whole file and only keep the relevant slices,
        !! even though we only need the part corresponding to the npe perturbations 
        !! in the irrep. 
        CALL readdnsscf(dnsscf_all_modes(:, :, :, :, :), nmodes, iq_irr, nqc_irr, Hubbard_lmax)
        DO ipert = 1, npe
          dnsscf(:, :, :, :, ipert) = dnsscf_all_modes(:, :, :, :, imode0 + ipert)
        ENDDO
      ENDIF
    ENDIF
    !
    !! 2020.03 HL
    !! Below root_pool should be interpreted as the index of root pool group.
    !! Currently, there is no root_pool_id like root_bgrp_id in bands groups.
    !
    CALL mp_bcast(dvscfin, ionode_id, inter_pool_comm)
    IF (okpaw) CALL mp_bcast(int3_paw, ionode_id, inter_pool_comm)
    IF (lda_plus_u) CALL mp_bcast(dnsscf, ionode_id, inter_pool_comm)
    !
    CALL newdq2(dvscfin, npe, xq0, timerev)
    !
    IF (doublegrid) THEN
      DO is = 1, nspin_mag
        DO ipert = 1, npe
          CALL fft_interpolate(dfftp, dvscfin(:, is, ipert), dffts, dvscfins(:, is, ipert))
        ENDDO
      ENDDO
    ELSE
      dvscfins(:, :, :) = dvscfin(:, :, :)
    ENDIF
    !
    DEALLOCATE(dvscfin, STAT = ierr)
    IF (ierr /= 0) CALL errore('dvscf_read', 'Error deallocating dvscfin', 1)
    !
    CALL stop_clock('dvscf_read')  ! rtsc: was never stopped
    !
    !-----------------------------------------------------------------------
    END SUBROUTINE dvscf_read
    !-----------------------------------------------------------------------
    LOGICAL FUNCTION ep_coarse_gpu_ok()
    !-----------------------------------------------------------------------
    !! rtsc: .TRUE. when the GPU path of ep_coarse_compute applies: CUDA build, norm-conserving,
    !! collinear, no spin polarisation, no DFT+U, and RTSC_EPW_COARSE is not '0'.
    USE uspp,             ONLY : okvan
    USE noncollin_module, ONLY : noncolin, npol
    USE ldaU,             ONLY : lda_plus_u
    USE input,            ONLY : lsda
    CHARACTER(LEN = 16) :: envval
    ep_coarse_gpu_ok = .FALSE.
#if defined (__CUDA)
    CALL GET_ENVIRONMENT_VARIABLE('RTSC_EPW_COARSE', envval)
    ep_coarse_gpu_ok = (TRIM(envval) /= '0') .AND. (.NOT. okvan) .AND. (.NOT. noncolin) .AND. &
                       (.NOT. lda_plus_u) .AND. (TRIM(lsda) == 'none') .AND. (npol == 1)
#endif
    END FUNCTION ep_coarse_gpu_ok
    !
    !---------------------------------------------------------------------
    SUBROUTINE ep_coarse_compute(irr, npe, imode0, dvscfins, gmapsym, eigv, &
                                 isym, xq0, nqc, el_ph_mat, epmatq, timerev)
    !---------------------------------------------------------------------
    !!
    !! Calculation of the electron-phonon matrix elements el_ph_mat
    !! <\psi(k+q)|dV_{SCF}/du^q_{i a}|\psi(k)>
    !!
    !! Written by Feliciano Giustino based on the routine PH/elphon.f90/elphel.
    !! Main difference w.r.t. to original routine is gauge fixing,
    !! shuffle (umklapp) mode and all-q implementation.
    !!
    !! Shuffle mode implemented on may 7 2006
    !!
    !! Nota Bene: this SUBROUTINE is intended only for one proc per pool,
    !! i.e. with no G-vector parallelization (some work on the igkq is
    !! required for that in the g-mapping)
    !!
    !! In order to allow a pool reading the wfc file of another
    !! pool, I had to modify the bound npwx in PW/n_plane_waves.f90
    !! which is now the max across all pools. In this way lrwfc is
    !! the same for all pools.
    !!
    !! RM - Nov/Dec 2014
    !! Imported the noncolinear case implemented by xlzhang
    !!
    !! SP - Nov 2015
    !! We want g(k,Sq) = < k+S(q)(r) | V_S(q)(r) | k(r) >
    !!                 = < k+S(q)(r) | V_q({S|v}^-1 r) | k(r) >
    !!                 = < k+S(q)({S|v}r) | V_q (r) | k({S|v}r) >
    !!
    !! It is important to note that the KB projectors that are applied to the V need
    !! to be computed at (r) and not ({S|v}r). Therefore, for the KB proj (computed in
    !! init_us_2, we need to provide the < Sk+q (r)| and |Sk (r)>.
    !! See Eq. 11.40 and 11.41 of the R. Martin Electronic Structure book.
    !!
    !! Note that in QE Sq is defined as S^-1(q)
    !!
    !! In case of time-reversal
    !! ------------------------
    !!  g(k,-Sq) = < k-S(q)({S|v}r) | V^loc_-q (r) + (V^nloc_q)* | k({S|v}r) >
    !!  where V^loc_{-q} is obtained with setlocq and V^nloc_q = CONGJ(u_pattern)*dvscfins*u_pattern.
    !!  We have to do this splitting because we do not have V^nloc_-q and
    !!  V^loc has to be computed at -q to be mappable with the vkb of the wavefunctions
    !!  computed in init_us_2.
    !!
    !! Roxana Margine - Jan 2019: Updated based on QE 6.3 for US
    !! SP - Sept. 2019: Cleaning
    !!
    !! JL - Feb 2020
    !! Roation of the <k| and <k+q| with SU(2) in the case of noncolin wavefunctions
    !!
    !---------------------------------------------------------------------
    !
    USE kinds,            ONLY : DP
    USE mp_global,        ONLY : my_pool_id, nproc_pool, intra_pool_comm, &
                                 inter_pool_comm, inter_image_comm, world_comm
    USE mp,               ONLY : mp_barrier, mp_bcast, mp_sum, mp_max
    USE wavefunctions,    ONLY : evc
    USE io_files,         ONLY : diropn
    USE wvfct,            ONLY : npwx
    USE pwcom,            ONLY : current_spin, nbnd
    USE cell_base,        ONLY : at, bg
    USE input,            ONLY : xk_loc, xk_all, isk_loc, et_all, isk_all, lsda
    USE cell_base,        ONLY : tpiba
    USE gvect,            ONLY : g
    USE uspp,             ONLY : vkb
    USE symm_base,        ONLY : s, ft
    USE modes,            ONLY : u, nirr, nmodes
    USE ions_base,        ONLY : nat
    USE qpoint,           ONLY : xq, npwq
    USE eqv,              ONLY : dvpsi
    USE phcom,            ONLY : evq
    USE units_lr,         ONLY : lrwfc, iuwfc
    USE phus,             ONLY : alphap
    USE lrus,             ONLY : becp1
    USE becmod,           ONLY : calbec
    USE global_var,       ONLY : igk_k_all, xkq, etq, ngk_all, lower_band,  &
                                 upper_band, ibndkept, nbndep, ngxx, ngxxf, &
                                 ng0vec, shift, gmap, g0vec_all_r, nk_loc,  &
                                 nkpts
    USE fft_base,         ONLY : dffts
    USE ep_constants,     ONLY : czero, cone, ci, zero, eps8, eps4
    USE klist,            ONLY : nkstot
    USE parallelism,      ONLY : kpointdivision, fkbounds, fkbounds_bnd
    USE kfold,            ONLY : ktokpmq
    USE low_lvl,          ONLY : fractrasl, rotate_cart, s_crystocart
    USE io,               ONLY : readwfc, readkmap
    USE noncollin_module, ONLY : noncolin, npol, nspin_mag
    USE dvqpsi,           ONLY : dvqpsi_us3, adddvscf2
    USE uspp_init,        ONLY : init_us_2
    USE input,            ONLY : nqc1, nqc2, nqc3, ngk_loc, igk_k_loc
    USE lrus,             ONLY : int3, int3_nc, int3_paw
    USE uspp,             ONLY : okvan
    USE paw_variables,    ONLY : okpaw
    USE io_global,        ONLY : stdout
    USE dvqhub,           ONLY : dvqhub_barepsi_us3, adddvhubscf2, lr_orthoUwfc2
    USE ldaU,             ONLY : lda_plus_u, nwfcU
    USE ldaU_lr,          ONLY : dnsscf
    USE ldaU_ph,          ONLY : dnsscf_all_modes
#if defined (__ONEMKL)
    USE epw_onemkl
#endif
#if defined (__CUDA)
    USE epw_gpu_coarse,   ONLY : egc_init, egc_set_v, egc_local, egc_nl_set, egc_local_nl, egc_prof_on, &
                                 egc_prof_add, egc_prof_print, egc_wall, egc_becs_set
#endif
    USE ions_base,        ONLY : ityp, ntyp => nsp
    USE uspp_param,       ONLY : nh, nhm
    USE uspp,             ONLY : nkb
    USE input,            ONLY : et_loc
    !
    IMPLICIT NONE
    !
    INTEGER, INTENT(in) :: irr
    !! Current irr perturbation
    INTEGER, INTENT(in) :: npe
    !! Number of perturbations for this irr representation
    INTEGER, INTENT(in) :: imode0
    !! Current mode number
    INTEGER, INTENT(in) :: gmapsym(ngxxf)
    !! Correspondence  G->S(G)
    INTEGER, INTENT(in) :: isym
    !! The symmetry which generates the current q in the star
    INTEGER, INTENT(in) :: nqc
    !! Current q-point index
    REAL(KIND = DP), INTENT(in) :: xq0(3)
    !! The first q-point in the star (cartesian coords.)
    COMPLEX(KIND = DP), INTENT(inout) :: dvscfins(dffts%nnr, nspin_mag, npe)
    !! rtsc epw3: inout, filled here when a RTSC_EPW_STAR member defers its read (star_lazy)
    !! Delta scf potential
    COMPLEX(KIND = DP), INTENT(in) :: eigv(ngxxf)
    !! $e^{iGv}$ for 1...nsym (v the fractional translation)
    COMPLEX(KIND = DP), INTENT(inout) :: el_ph_mat(nbndep, nbndep, nk_loc, 3 * nat)
    !! ep matrix element for this q-point
    COMPLEX(KIND = DP), INTENT(inout) :: epmatq(nbndep, nbndep, nk_loc, nmodes, nqc1 * nqc2 * nqc3)
    !! ep-mat on full BZ
    LOGICAL, INTENT(in) :: timerev
    !!  true if we are using time reversal
    !
    ! Local variables
    LOGICAL :: exst
    !! logical variable to check file exists
    INTEGER :: ik
    !! Counter on k-points in the pool
    INTEGER :: ik0
    !! Index of the first k-point block in this pool - 1
    INTEGER :: ibnd
    !! Counter on bands
    INTEGER :: jbnd
    !! Counter on bands
    INTEGER :: ipert
    !! Counter on change of Vscf due to perturbations
    INTEGER :: mode
    !! Counter on modes plus pertubations
    INTEGER :: ig
    !! Counter on G-vectors
    INTEGER :: ipooltmp
    !! Index of pool for k
    INTEGER :: ipool
    !! Index of pool for k+q
    INTEGER :: igkq_tmp(npwx)
    !! Correspondence k+q+G <-> G
    INTEGER :: imap
    !! Index in gmap for G-sphere igkq translation with G_0
    INTEGER :: ipol
    !! Counter on polarizations
    INTEGER :: npw
    !! Number of k+G-vectors inside 'ecut sphere'
    INTEGER :: lower_bnd
    !! Lower bounds index after k paral
    INTEGER :: upper_bnd
    !! Upper bounds index after k paral
    INTEGER :: nbnd_loc
    !! Local number of bands for band parallelization
    INTEGER :: nkq
    !! Index of k+q-point in the pool
    INTEGER :: nkq_abs
    !! Absolute index of k+q-point
    INTEGER :: ierr
    !! Error status
    INTEGER, ALLOCATABLE :: igk(:)
    !! Index for k+G
    INTEGER, ALLOCATABLE :: igkq(:)
    !! Index for k+q+G
    ! Local variables for rotating the wavefunctions (in order to use q in the irr wedge)
    REAL(KIND = DP) :: xkqtmp(3)
    !! Temporary k+q vector for KB projectors
    REAL(KIND = DP) :: sxk(3)
    !! Rotated k-point xk
    REAL(KIND = DP) :: zero_vect(3)
    !! Temporary zero vector
    REAL(KIND = DP) :: s_cart(3, 3)
    !! Symmetry matrix in Cartesian basis.
    COMPLEX(KIND = DP) :: su2(2, 2)
    !! SU2 rotation matrix to act WFs
    COMPLEX(KIND = DP) :: su2_no_dagger(2, 2)
    !! SU2 that comes out of find_u, we need the dagger of this, which we assign to su2
    COMPLEX(KIND = DP), ALLOCATABLE :: aux1(:, :)
    !! Auxillary wavefunction
    COMPLEX(KIND = DP), ALLOCATABLE :: aux2(:, :)
    !! Auxillary wavefunction
    COMPLEX(KIND = DP), ALLOCATABLE :: aux4(:, :)
    !! Rotated psi_m,k+q WF by SU(2)
    COMPLEX(KIND = DP), ALLOCATABLE :: aux5(:, :)
    !! Rotated psi_nk WF by SU(2)
    COMPLEX(KIND = DP), ALLOCATABLE :: elphmat(:, :, :)
    !! arrays for e-ph matrix elements
    COMPLEX(KIND = DP), ALLOCATABLE :: wfcatomk_(:, :)
    !! projector for atomic wavefunctions at k-point
    COMPLEX(KIND = DP), ALLOCATABLE :: wfcatomkpq_(:, :)
    !! projector for atomic wavefunctions at k+q-point
    COMPLEX(KIND = DP), ALLOCATABLE :: swfcatomk_(:, :)
    !! projector for S*atomic wavefunctions at k-point
    COMPLEX(KIND = DP), ALLOCATABLE :: swfcatomkpq_(:, :)
    !! projector for S*atomic wavefunctions at k+q-point
    ! rtsc: GPU path (local part by Parseval on the GPU, nonlocal part by projections)
    LOGICAL :: use_gpu, nl_gpu
    LOGICAL :: bec_gpu
    !! rtsc amdahl: RTSC_EPW_BEC (becp1 / alphap on the device)
    REAL(KIND = DP), ALLOCATABLE :: kgk(:, :)
    INTEGER :: nt, na, mu, ih, jh, ikb, jkb, ijkb0
    INTEGER, ALLOCATABLE :: kbatom(:)
    REAL(KIND = DP), ALLOCATABLE :: kqg(:, :)
    COMPLEX(KIND = DP), ALLOCATABLE :: dmat(:, :), uqm(:, :)
    INTEGER, ALLOCATABLE :: indk(:), indq(:)
    REAL(KIND = DP) :: xq_bak(3)
    REAL(KIND = DP), ALLOCATABLE :: deff(:, :, :)
    COMPLEX(KIND = DP), ALLOCATABLE :: vtot(:, :), becq(:, :, :), auxv(:, :), ps1(:, :), ps2(:, :, :)
    ! rtsc epw3: RTSC_EPW_STAR
    LOGICAL :: star_done, star_cmp, lazy, use_star
    LOGICAL, ALLOCATABLE :: kdefer(:)
    INTEGER :: kpass, ndefer
    COMPLEX(KIND = DP), ALLOCATABLE :: gstar(:, :, :)
    !
    ALLOCATE(elphmat(nbndep, nbndep, npe), STAT = ierr)
    IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error allocating elphmat', 1)
    ALLOCATE(aux1(dffts%nnr, npol), STAT = ierr)
    IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error allocating aux1', 1)
    elphmat(:, :, :) = czero
    aux1(:, :) = czero
    zero_vect = zero
    xkq(:, :) = zero
    !
    IF (nproc_pool > 1) THEN
      CALL errore('ep_coarse_compute', 'only one proc per pool in shuffle mode', 1)
    ENDIF
    !
    ! find the bounds of k-dependent arrays in the parallel case in each pool
    CALL fkbounds(nkpts, lower_bnd, upper_bnd)
    !
    ! S.Tiwari: Commented since v6.1 
    ! SP: Bound for band parallelism
    !CALL fkbounds_bnd(nbndep, lower_band, upper_band)
    upper_band = nbndep
    lower_band = 1
    nbnd_loc = upper_band - lower_band + 1
    !
    ALLOCATE(aux2(npwx * npol, nbndep), STAT = ierr)
    IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error allocating aux2', 1)
    ALLOCATE(aux4(npwx * npol, nbndep), STAT = ierr)
    IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error allocating aux4', 1)
    ALLOCATE(aux5(npwx * npol, nbndep), STAT = ierr)
    IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error allocating aux5', 1)
    ALLOCATE(dvpsi(npwx * npol, lower_band:upper_band), STAT = ierr)
    IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error allocating dvpsi', 1)
    dvpsi(:, :) = czero
    !
    ! setup for k+q folding
    !
    CALL kpointdivision(ik0)
    !
    CALL readkmap(nkpts)
    !
    ! close all sequential files in order to re-open them as direct access
    ! close all .wfc files in order to prepare shuffled read
    !
    CLOSE(iuwfc, STATUS = 'keep')
    ! never remove this barrier
    CALL mp_barrier(inter_pool_comm)
    !
    ALLOCATE(etq(nbnd, nk_loc), STAT = ierr)
    IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error allocating etq', 1)
    IF (lda_plus_u) THEN
      ALLOCATE(wfcatomk_(npwx * npol, nwfcU), STAT = ierr)
      IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error allocating wfcatomk', 1)
      ALLOCATE(wfcatomkpq_(npwx * npol, nwfcU), STAT = ierr)
      IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error allocating wfcatomkpq', 1)
      ALLOCATE(swfcatomk_(npwx * npol, nwfcU), STAT = ierr)
      IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error allocating swfcatomk', 1)
      ALLOCATE(swfcatomkpq_(npwx * npol, nwfcU), STAT = ierr)
      IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error allocating swfcatomkpq', 1)
      !
      wfcatomk_(:, :) = czero
      wfcatomkpq_(:, :) = czero
      swfcatomk_(:, :) = czero
      swfcatomkpq_(:, :) = czero
    ENDIF
    !
    ! rtsc: decide on the GPU path (NC, collinear, no spin, no DFT+U); RTSC_EPW_COARSE=0 disables it
    use_gpu = .FALSE.
    nl_gpu = .FALSE.
    bec_gpu = .FALSE.
#if defined (__CUDA)
    use_gpu = ep_coarse_gpu_ok()
#endif
    ! rtsc epw3 (RTSC_EPW_STAR): a member using the shortcut defers dvscf and the GPU potentials to pass 2,
    ! which runs only if some k of some pool needs the direct form
    lazy = star_lazy()
    IF (.NOT. lazy) CALL gpu_setup()
    ALLOCATE(kdefer(nk_loc))
    kdefer(:) = .FALSE.
    !
    pass_loop: DO kpass = 1, 2
    IF (kpass == 2) THEN
      IF (.NOT. lazy) EXIT pass_loop
      ndefer = COUNT(kdefer)
      CALL mp_max(ndefer, inter_pool_comm)
      IF (ndefer == 0) EXIT pass_loop
      CALL dvscf_read(imode0, star_iqirr, star_nqcirr, npe, xq0, dvscfins, timerev)
      CALL gpu_setup()
    ENDIF
    use_star = star_now .AND. kpass == 1
    !
    DO ik = 1, nk_loc
      IF (kpass == 2) THEN
        IF (.NOT. kdefer(ik)) CYCLE
      ENDIF
      !
      etq(:, :) = zero
      elphmat(:, :, :) = czero
      current_spin = 1
      IF (TRIM(lsda) /= 'none') THEN
        current_spin = isk_loc(ik)
      ENDIF
      !DBSP
      !b = zero
      !c = zero
      !d = zero
      !
      ! find index, and possibly pool, of k+q
      ! the index nkq (nkq_abs) takes into account the even/odd ordering
      ! of the nscf calc
      ! we also redefine the ikq points and the corresponding energies
      ! (we need to make sure that xk(:,ikq) is really k+q for the KB projectors
      ! below and also that the eigenvalues are taken correctly in ephwann)
      ! read unperturbed wavefunctions psi(k) and psi(k+q)
      !
      CALL ktokpmq(xk_loc(:, ik), zero_vect, +1, ipool, nkq, nkq_abs)
      !
      ipooltmp = ipool
      !
      CALL start_clock('ep_wfcread')
      CALL readwfc(ipooltmp, nkq, evc)
      !
      !
      CALL ktokpmq(xk_loc(:, ik), xq, +1, ipool, nkq, nkq_abs)
      !
      !   we define xkq(:,ik) and etq(:,ik) for the current xq
      !
      xkq(:, ik) = xk_all(:, nkq_abs - (current_spin - 1) * nkpts)
      etq(:, ik) = et_all(:, nkq_abs - (current_spin - 1) * nkpts)
      !
      ! in serial execution ipool is not used in the called subroutines,
      ! in parallel ipooltmp is for k and ipool is for k+q
      !
      CALL readwfc(ipool, nkq, evq)
      CALL stop_clock('ep_wfcread')
      CALL start_clock('ep_prolog')
      !
      ! --------------------------------------------------
      !  Calculate SO(3) in cartesian from s_axis_to_cart
      !  then calculate SU(2) transformation matrix from
      !  find_u.
      ! --------------------------------------------------
      s_cart(:, :) = zero
      su2_no_dagger(:, :) = czero
      su2(:, :) = czero
      CALL s_crystocart(s(:,:,isym), s_cart, at, bg)
      CALL find_u(s_cart, su2_no_dagger)
      su2 = TRANSPOSE(CONJG(su2_no_dagger))
      !
      ! Now we define the igk and igkq from the global igk_k_all
      !
      npw  = ngk_loc(ik)
      npwq = ngk_all(nkq_abs - (current_spin - 1) * nkpts)
      !
      ALLOCATE(igk(npw), STAT = ierr)
      IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error allocating igk', 1)
      ALLOCATE(igkq(npwq), STAT = ierr)
      IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error allocating igkq', 1)
      !
      igk = igk_k_loc(1:npw, ik)
      igkq = igk_k_all(1:npwq, nkq_abs - (current_spin - 1) * nkpts )
      !
      IF (nk_loc > 1 .AND. MAXVAL(igkq(1:npwq)) > ngxx) THEN
        CALL errore('ep_coarse_compute', 'ngxx too small', 1)
      ENDIF
      !
      !------------------------------------------------------------
      ! Rotate evc --> axu5 and evq --> aux4 by SU2^{dagger}
      !------------------------------------------------------------
      aux4(:, :) = czero
      aux5(:, :) = czero
      DO ibnd = 1, nbndep
        jbnd = ibndkept(ibnd)
        IF (noncolin) THEN
          DO ig = 1, npw
            aux5(ig, ibnd) = su2(1, 1) * evc(ig, jbnd) + su2(1, 2) * evc(ig + npwx, jbnd)
            aux5(ig + npwx, ibnd) = su2(2, 1) * evc(ig, jbnd) + su2(2, 2) * evc(ig + npwx, jbnd)
          ENDDO
          DO ig = 1, npwq
            aux4(ig, ibnd) = su2(1, 1) * evq(ig, jbnd) + su2(1, 2) * evq(ig + npwx, jbnd)
            aux4(ig + npwx, ibnd) = su2(2, 1) * evq(ig, jbnd) + su2(2, 2) * evq(ig + npwx, jbnd)
          ENDDO
        ELSE
          aux5(1:npw,  ibnd) = evc(1:npw,  jbnd)
          aux4(1:npwq, ibnd) = evq(1:npwq, jbnd)
        ENDIF
      ENDDO
      !
      ! ----------------------------------------------------------------
      ! Set the gauge for the eigenstates: unitary transform and phases
      ! ----------------------------------------------------------------
      !
      ! With this option, different compilers and different machines
      ! should always give the same wavefunctions.
      !
!      CALL ktokpmq(xk_loc(:, ik),  zero_vect, +1, ipool, nkk, nkk_abs)
!      CALL ktokpmq(xkq(:, ik), zero_vect, +1, ipool, nkk, nkq_abs)
      !
!      umat(:, :, ik)  = umat_all(:, :, nkk_abs)
!      umatq(:, :, ik) = umat_all(:, :, nkq_abs)
      !
      ! the k-vector needed for the KB projectors
      xkqtmp = xkq(:, ik)
      !
      ! --------------------------------------------------
      !   Fourier translation of the G-sphere igkq
      ! --------------------------------------------------
      !
      !  Translate by G_0 the G-sphere where evq is defined,
      !  none of the G-points are lost.
      !
      IF (ANY( ABS(g0vec_all_r(:, shift(ik + ik0))) > eps8 )) THEN
        DO ig = 1, npwq
          imap = ng0vec * (igkq(ig) - 1) + shift(ik + ik0)
          igkq_tmp(ig) = gmap(imap)
          !  the old matrix version...
          !  igkq_tmp(ig) = gmap( igkq(ig), shift(ik+ik0) )
        ENDDO
        igkq = igkq_tmp
      ENDIF
      !  find k+q from k+q+G_0
      !  (this is needed in the calculation of the KB terms
      !  for nonlocal pseudos)
      !
      xkqtmp = xkq(:, ik) - g0vec_all_r(:, shift(ik + ik0))
      !
      ! ---------------------------------------------------------------------
      ! phase factor arising from fractional traslations
      ! ---------------------------------------------------------------------
      !
      !  u_{k+q+G_0} carries an additional factor e^{i G_0 v}
      !
      IF (ANY( ABS(ft(:, isym)) > eps8 )) THEN
        CALL fractrasl(nbndep, npw,  igk,  aux5, eigv, cone)
        CALL fractrasl(nbndep, npwq, igkq, aux4, eigv, cone)
      ENDIF
      !
      ! ---------------------------------------------------------------------
      ! wave function rotation to generate matrix elements for the star of q
      ! ---------------------------------------------------------------------
      !
      ! ps. don't use npwx instead of npw, npwq since the unused elements
      ! may be large and blow up gmapsym (personal experience)
      !
      IF (isym /= 1) THEN
         igk(1:npw) = gmapsym(igk(1:npw))
         igkq(1:npwq) = gmapsym(igkq(1:npwq))
      ENDIF
      !
      ! In dvqpsi_us_only3 we need becp1 and alphap for the rotated wfs.
      ! The other quantities (deeq and qq) do not depend on the wfs, in
      ! particular in the KB case (not ultrasoft), the deeq's are the
      ! unscreened coefficients, and the qq's are zero.
      !
      ! For the KB part, remember dV_NL[q_0] ~ |S^-1(k)+q_0> <S^-1(k)|
      ! the total momentum transfer must be q_0 and the rotation
      ! tranforms k+Sq_0 into S^-1(k)+q_0, k into S^-1(k)
      ! [see Eqs. (A9),(A14) Baroni et al. RMP]
      !
      ! Since in QE a normal rotation s is defined as S^-1 we have here
      ! sxk = S(k).
      !
      CALL rotate_cart(xk_loc(:, ik), s(:, :, isym), sxk)
      !
      ! rtsc epw3 (RTSC_EPW_STAR=1): g from the base member of the star and the band overlaps D (module header)
      star_done = .FALSE.
      star_cmp = .FALSE.
      IF (use_star) THEN
        CALL stop_clock('ep_prolog')
        CALL start_clock('ep_star')
        IF (.NOT. ALLOCATED(gstar)) ALLOCATE(gstar(nbndep, nbndep, npe))
        CALL star_g(sxk, sxk + xq0, npw, igk, aux5, npwq, igkq, aux4, npe, gstar, star_done)
        CALL stop_clock('ep_star')
        IF (star_done) THEN
          star_nsc = star_nsc + 1
          elphmat(:, :, 1:npe) = gstar(:, :, 1:npe)
          IF (star_chk .AND. ik <= 2) THEN
            star_done = .FALSE.
            star_cmp = .TRUE.
          ENDIF
        ELSE
          star_nfb = star_nfb + 1
          IF (lazy) THEN
            ! direct form in pass 2, after the deferred dvscf read
            kdefer(ik) = .TRUE.
            DEALLOCATE(igk, igkq)
            CYCLE
          ENDIF
        ENDIF
        IF (.NOT. star_done) CALL start_clock('ep_prolog')
      ENDIF
      IF (.NOT. star_done) THEN
      !
      ! here we generate vkb on the igk() set and for k ...
      CALL init_us_2(npw, igk, sxk, vkb)
      !
      ! rtsc amdahl (RTSC_EPW_BEC=1, GPU NL path): becp1 and alphap are formed on the device in egc_local_nl
      IF (bec_gpu) THEN
        DO ipol = 1, 3
          DO ig = 1, npw
            kgk(ig, ipol) = sxk(ipol) + g(ipol, igk(ig))
          ENDDO
        ENDDO
        CALL egc_becs_set(npw, nkb, npwx, vkb, kgk, tpiba)
      ELSE
      !
      ! ... and we recompute the becp terms with the wfs (rotated through igk)
      !
      CALL calbec(npw, vkb, aux5, becp1(ik), nbndep)
      !
      ! we also recompute the derivative of the becp terms with the (rotated) wfs
      !
      DO ipol = 1, 3
        aux2(:, :) = czero
        DO ibnd = lower_band, upper_band
          DO ig = 1, npw
            aux2(ig, ibnd) = aux5(ig, ibnd) * tpiba * ci * (sxk(ipol) + g(ipol,igk(ig)))
          END DO
          IF (noncolin) THEN
            DO ig = 1, npw
              aux2(ig + npwx, ibnd) = aux5(ig + npwx, ibnd) * tpiba * ci * (sxk(ipol) + g(ipol, igk(ig)))
            ENDDO
          ENDIF
        ENDDO
        CALL calbec(npw, vkb, aux2(:, lower_band:upper_band), alphap(ipol, ik), upper_band - lower_band + 1)
      ENDDO
      ENDIF   ! rtsc bec_gpu
      !
      ! now we generate vkb on the igkq() set because dvpsi is needed on that set
      ! we need S(k)+q_0 in the KB projector: total momentum transfer must be q_0
      !
      xkqtmp = sxk + xq0
      CALL init_us_2(npwq, igkq, xkqtmp, vkb)
      !
      !! Defining \phi at S^-1(k) and S^-1(k)+q_0
      !! In a similar way to the KB part
      IF (lda_plus_u) THEN
         CALL lr_orthoUwfc2(sxk, xkqtmp, igk, igkq, npw, npwq, wfcatomk_, &
                            wfcatomkpq_, swfcatomk_, swfcatomkpq_, .FALSE.)      
      ENDIF
      !
      ! --------------------------------------------------
      !   Calculation of the matrix element
      ! --------------------------------------------------
      !
      CALL stop_clock('ep_prolog')
      IF (use_gpu) THEN
        CALL start_clock('ep_gpu_k')
        CALL ep_gpu_k()
        CALL stop_clock('ep_gpu_k')
      ELSE
      DO ipert = 1, npe
        !
        !  recalculate dvbare_q*psi_k
        !  the call to dvqpsi_us3 differs from the old one to dvqpsi_us
        !  only the xkqtmp passed.
        !
        !  The bare potential from nonlinear core correction is not compute in dvqpsi_us3
        !  (because of the .FALSE. parameter) because it is part of the induced potential
        !  dvscfins written by ph.x. This is done so because the bare potential is spin
        !  independent but the potential due to core correction is not.
        !
        !  we have to use the first q in the star in the dvqpsi_us3 call below (xq0)
        !
        mode = imode0 + ipert
        IF (timerev) THEN
          CALL dvqpsi_us3(ik, CONJG(u(:, mode)), .FALSE., xkqtmp, xq0, igk, igkq, npw, npwq, &
                          aux5, CONJG(dvscfins(:, :, ipert)))
          ! DFPT+U: calculate the bare derivative of the Hubbard potential in el-ph
          IF (lda_plus_u) CALL dvqhub_barepsi_us3(ik, CONJG(u(:, mode)), xkqtmp, sxk, igk, &
              igkq, npw, npwq, aux5, wfcatomk_, wfcatomkpq_, swfcatomk_, swfcatomkpq_, nwfcU) 
        ELSE
          CALL dvqpsi_us3(ik, u(:, mode), .FALSE., xkqtmp, xq0, igk, igkq, npw, npwq, &
                          aux5, dvscfins(:, :, ipert))
          ! DFPT+U: calculate the bare derivative of the Hubbard potential in el-ph
          IF (lda_plus_u) CALL dvqhub_barepsi_us3(ik, u(:, mode), xkqtmp, sxk, igk, igkq, &
                npw, npwq, aux5, wfcatomk_, wfcatomkpq_, swfcatomk_, swfcatomkpq_, nwfcU) 
        ENDIF
        !DBSP
        ! b = b+SUM((REAL(REAL(dvpsi(:, :))))**2)+SUM((REAL(AIMAG(dvpsi(:, :))))**2)
        !
        CALL adddvscf2(ipert, ik)
        ! DFPT+U: add to dvpsi the scf part of the response
        ! Hubbard potential dV_hub
        !
        IF (lda_plus_u) CALL adddvhubscf2(ipert, ik, npw, npwq, aux5, swfcatomk_, swfcatomkpq_, nwfcU)
        ! DBPS
        ! c = b+SUM((REAL(REAL(dvpsi(:, :))))**2)+SUM((REAL(AIMAG(dvpsi(:, :))))**2)
        !
        ! calculate elphmat(j,i)=<psi_{k+q,j}|dvscf_q*psi_{k,i}> for this pertur
        !
#if defined (__ONEMKL)
        !$omp target data map(aux4,dvpsi(1:npwx*npol,lower_band:upper_band),elphmat)
        !$omp dispatch
#endif
        CALL ZGEMM('C', 'N', nbndep, nbnd_loc, npwx * npol, &
                  (1.d0, 0.d0), aux4(1, 1), npwx * npol, dvpsi(1, lower_band), npwx * npol, &
                  (0.d0, 0.d0), elphmat(1, 1, ipert), nbndep)
#if defined (__ONEMKL)
        !$omp end target data
#endif
        !
      ENDDO
      ENDIF ! use_gpu
      !
      IF (star_cmp) THEN
        star_nck = star_nck + 1
        star_cmax = MAX(star_cmax, MAXVAL(ABS(elphmat(:, :, 1:npe) - gstar(:, :, 1:npe))) / &
                                   MAXVAL(ABS(elphmat(:, :, 1:npe))))
      ENDIF
      ENDIF ! rtsc epw3: .NOT. star_done
      !
      CALL mp_sum(elphmat, intra_pool_comm)
      !
      !DBSP
      !IF (ik==2) THEN
      !  write(*,*)'SUM dvpsi b ', b
      !  write(*,*)'SUM dvpsi c ', c
      !  write(*,*)'elphmat(:, :, :)**2', SUM((REAL(REAL(elphmat(:, :, :))))**2)+SUM((REAL(AIMAG(elphmat(:, :, :))))**2)
      !ENDIF
      !
      ! 02/2020
      ! The part below is commented since umat and umatq originating from umat_all in setphases_wrap
      ! are identity matrices.
      !
      ! Rotate elphmat with the gauge matrices (this should be equivalent
      ! to calculate elphmat with the truely rotated eigenstates)
      !
      ! DO ipert = 1, npe
      !   !
      !   ! the two zgemm call perform the following ops:
      !   !  elphmat = umat(k+q)^\dagger * [ elphmat * umat(k) ]
      !   !
      !   CALL ZGEMM('n', 'n', nbnd, nbnd, nbnd, cone, elphmat(:, :, ipert), &
      !              nbnd, umat(:, :, ik), nbnd, czero, eptmp, nbnd)
      !   CALL ZGEMM('c', 'n', nbnd, nbnd, nbnd, cone, umatq(:, :, ik), &
      !              nbnd, eptmp, nbnd, czero, elphmat(:, :, ipert), nbnd)
      !   !
      ! ENDDO
      !
      !  save eph matrix elements into el_ph_mat
      !
      DO ipert = 1, npe
        DO jbnd = 1, nbndep
          DO ibnd = 1, nbndep
            el_ph_mat(ibnd, jbnd, ik, ipert + imode0) = elphmat(ibnd, jbnd, ipert)
          ENDDO
        ENDDO
      ENDDO
      !
      DEALLOCATE(igk, STAT = ierr)
      IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error deallocating igk', 1)
      DEALLOCATE(igkq, STAT = ierr)
      IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error deallocating igkq', 1)
      !
    ENDDO ! ik
    END DO pass_loop
    DEALLOCATE(kdefer)
    !
    ! When all irr perturbations have been done, rotate from pattern to Cartesian basis
    IF (irr == nirr) THEN
      ! epmat_{CART} = conjg ( U ) * epmat_{PATTERN}
      ! note it is not U^\dagger but u_pattern!
      ! Have a look to symdyn_munu.f90 for comparison
      !
      DO ibnd = 1, nbndep
        DO jbnd = 1, nbndep
          DO ik = 1, nk_loc
            !
            ! Here is where we calculate epmatq, it appears to be
            ! epmatq = cone * conjug(u) * el_ph_mat + czero
            IF (timerev) THEN
              CALL ZGEMV('n', nmodes, nmodes, cone, u, nmodes, &
                el_ph_mat(ibnd, jbnd, ik, :), 1, czero, epmatq(ibnd, jbnd, ik, :, nqc), 1)
            ELSE
              CALL ZGEMV('n', nmodes, nmodes, cone, CONJG(u), nmodes, &
                el_ph_mat(ibnd, jbnd, ik, :), 1, czero, epmatq(ibnd, jbnd, ik, :, nqc), 1)
            ENDIF
            !
          ENDDO ! ik
        ENDDO ! jbnd
      ENDDO ! ibnd
      !
    ENDIF ! irr == nirr
    !
    !  restore original configuration of files
    !
    CALL diropn(iuwfc, 'wfc', lrwfc, exst)
    ! never remove this barrier - > insures that wfcs are restored to each pool before moving on
    CALL mp_barrier(inter_pool_comm)
    !
    IF (okvan) THEN
      DEALLOCATE(int3, STAT = ierr)
      IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error deallocating int3', 1)
      IF (okpaw) THEN
        DEALLOCATE(int3_paw, STAT = ierr)
        IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error deallocating int3_paw', 1)
      ENDIF
      IF (noncolin) THEN
        DEALLOCATE(int3_nc, STAT = ierr)
        IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error deallocating int3_nc', 1)
      ENDIF
    ENDIF
    !
    IF (use_gpu) CALL egc_prof_print()
    IF (star_now) WRITE(stdout, '(5x, "rtsc epw star (cumulative): ", I0, " k from the base member, ", I0, &
         &" computed directly (D not unitary), max |D^H D - 1| ", ES9.2, "; check: ", I0, " k, max |dg|/max|g| ", &
         &ES9.2)') star_nsc, star_nfb, star_umax, star_nck, star_cmax
    IF (ALLOCATED(gstar)) DEALLOCATE(gstar)
    IF (ALLOCATED(vtot)) DEALLOCATE(vtot, deff, becq, auxv, ps1, ps2, indk, indq)
    IF (ALLOCATED(kbatom)) DEALLOCATE(kbatom, dmat, uqm, kqg)
    IF (ALLOCATED(kgk)) DEALLOCATE(kgk)
    DEALLOCATE(elphmat, STAT = ierr)
    IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error deallocating elphmat', 1)
    DEALLOCATE(aux1, STAT = ierr)
    IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error deallocating aux1', 1)
    DEALLOCATE(aux2, STAT = ierr)
    IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error deallocating aux2', 1)
    DEALLOCATE(aux4, STAT = ierr)
    IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error deallocating aux4', 1)
    DEALLOCATE(aux5, STAT = ierr)
    IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error deallocating aux5', 1)
    DEALLOCATE(dvpsi, STAT = ierr)
    IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error deallocating dvpsi', 1)
    DEALLOCATE(etq, STAT = ierr)
    IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error deallocating etq', 1)
    IF (lda_plus_u) THEN
      DEALLOCATE(wfcatomk_, STAT = ierr)
      IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error deallocating wfcatomk', 1)
      DEALLOCATE(wfcatomkpq_, STAT = ierr)
      IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error deallocating wfcatomkpq', 1)
      DEALLOCATE(swfcatomk_, STAT = ierr)
      IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error deallocating swfcatomk', 1)
      DEALLOCATE(swfcatomkpq_, STAT = ierr)
      IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error deallocating swfcatomkpq', 1)
      DEALLOCATE(dnsscf_all_modes, STAT = ierr)
      IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error deallocating dnsscf_all_modes', 1)
      DEALLOCATE(dnsscf, STAT = ierr)
      IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error deallocating dnsscf', 1)
    ENDIF
    !
    CONTAINS
      !
      SUBROUTINE gpu_setup()
      !! rtsc: GPU-path setup of ep_coarse_compute (total local potentials, device buffers, nonlocal data)
#if defined (__CUDA)
      IF (use_gpu) THEN
        ! total local perturbing potential per perturbation: dvscf + dV_loc^bare (k-independent)
        ALLOCATE(vtot(dffts%nnr, npe), deff(nhm, nhm, nat), becq(nkb, nbndep, 0:3), auxv(npwx, nkb), &
                 ps1(nkb, nbndep), ps2(nkb, nbndep, 3), indk(npwx), indq(npwx), STAT = ierr)
        IF (ierr /= 0) CALL errore('ep_coarse_compute', 'Error allocating GPU-path arrays', 1)
        xq_bak = xq
        xq = xq0
        DO ipert = 1, npe
          mode = imode0 + ipert
          IF (timerev) THEN
            CALL compute_dvloc(CONJG(u(:, mode)), .FALSE., vtot(:, ipert))
            vtot(:, ipert) = vtot(:, ipert) + CONJG(dvscfins(:, 1, ipert))
          ELSE
            CALL compute_dvloc(u(:, mode), .FALSE., vtot(:, ipert))
            vtot(:, ipert) = vtot(:, ipert) + dvscfins(:, 1, ipert)
          ENDIF
        ENDDO
        xq = xq_bak
        CALL egc_init(dffts%nr1, dffts%nr2, dffts%nr3, dffts%nr1x, dffts%nr2x, dffts%nr3x, &
                      dffts%nnr, nbndep, 3 * nat, npwx)
        CALL egc_set_v(npe, vtot)
        ! nonlocal part on the GPU too (RTSC_EPW_NL=0 keeps the host code below)
        BLOCK
          CHARACTER(LEN = 16) :: ev
          CALL GET_ENVIRONMENT_VARIABLE('RTSC_EPW_NL', ev)
          nl_gpu = (TRIM(ev) /= '0') .AND. nkb > 0
          CALL GET_ENVIRONMENT_VARIABLE('RTSC_EPW_BEC', ev)
          bec_gpu = nl_gpu .AND. TRIM(ev) == '1' .AND. .NOT. noncolin .AND. lower_band == 1 .AND. upper_band == nbndep
          IF (bec_gpu) ALLOCATE(kgk(npwx, 3))
        END BLOCK
        IF (nl_gpu) THEN
          ALLOCATE(kbatom(nkb), dmat(nkb, nkb), uqm(3 * nat, npe), kqg(npwx, 3))
          ijkb0 = 0
          DO nt = 1, ntyp
            DO na = 1, nat
              IF (ityp(na) == nt) THEN
                kbatom(ijkb0 + 1:ijkb0 + nh(nt)) = na
                ijkb0 = ijkb0 + nh(nt)
              ENDIF
            ENDDO
          ENDDO
          DO ipert = 1, npe
            mode = imode0 + ipert
            IF (timerev) THEN
              uqm(:, ipert) = CONJG(u(:, mode))
            ELSE
              uqm(:, ipert) = u(:, mode)
            ENDIF
          ENDDO
        ENDIF
      ENDIF
#endif
      END SUBROUTINE gpu_setup
      !
      SUBROUTINE ep_gpu_k()
      !! rtsc: elphmat(:, :, 1:npe) for the current k, all perturbations of this irrep at once.
      !! Local part: (1/N) sum_r conj(psi_k+q(r)) V(r) psi_k(r) on the GPU (Parseval form of
      !! dvqpsi_us3 + ZGEMM). Nonlocal part: vanilla dvqpsi_us_only3 builds
      !! dvpsi_nl = vkb ps1 + sum_a [(k+q+G)_a vkb] ps2_a; projecting on aux4 gives
      !! becq0^H ps1 + sum_a becq_a^H ps2_a with becq computed once per k (NC, no US terms).
      INTEGER :: ipol_, ib
      REAL(DP) :: tpa
#if defined (__CUDA)
      IF (egc_prof_on()) tpa = egc_wall(.FALSE.)
      DO ig = 1, npw
        indk(ig) = dffts%nl(igk(ig))
      ENDDO
      DO ig = 1, npwq
        indq(ig) = dffts%nl(igkq(ig))
      ENDDO
      IF (nl_gpu) THEN
        ! NC: deff = deeq does not depend on the band energy; D is its block-diagonal dense form
        CALL compute_deff(deff, et_loc(ibndkept(1), ik))
        dmat = czero
        ijkb0 = 0
        DO nt = 1, ntyp
          DO na = 1, nat
            IF (ityp(na) == nt) THEN
              DO jh = 1, nh(nt)
                DO ih = 1, nh(nt)
                  dmat(ijkb0 + ih, ijkb0 + jh) = deff(ih, jh, na)
                ENDDO
              ENDDO
              ijkb0 = ijkb0 + nh(nt)
            ENDIF
          ENDDO
        ENDDO
        CALL egc_nl_set(nkb, 3 * nat, npe, dmat, kbatom, uqm)
        DO ipol_ = 1, 3
          DO ig = 1, npwq
            kqg(ig, ipol_) = xkqtmp(ipol_) + g(ipol_, igkq(ig))
          ENDDO
        ENDDO
        IF (egc_prof_on()) CALL egc_prof_add(6, egc_wall(.FALSE.) - tpa)
        CALL egc_local_nl(npw, indk, aux5, npwq, indq, aux4, vkb, npwx, kqg, becp1(ik)%k, nkb, &
                          alphap(1, ik)%k, alphap(2, ik)%k, alphap(3, ik)%k, tpiba, elphmat)
        RETURN
      ENDIF
      CALL egc_local(npw, indk, aux5, npwq, indq, aux4, elphmat)
#endif
      IF (nkb == 0) RETURN
      ! projections of psi_k+q on vkb(k+q) and on (k+q+G)_a vkb(k+q)
      CALL ZGEMM('C', 'N', nkb, nbndep, npwq, cone, vkb, npwx, aux4, npwx, czero, becq(1, 1, 0), nkb)
      DO ipol_ = 1, 3
        DO ikb = 1, nkb
          DO ig = 1, npwq
            auxv(ig, ikb) = vkb(ig, ikb) * (xkqtmp(ipol_) + g(ipol_, igkq(ig)))
          ENDDO
        ENDDO
        CALL ZGEMM('C', 'N', nkb, nbndep, npwq, cone, auxv, npwx, aux4, npwx, czero, &
                   becq(1, 1, ipol_), nkb)
      ENDDO
      ! NC (no okvan): deff = deeq does not depend on the band energy, so compute it once per k
      CALL compute_deff(deff, et_loc(ibndkept(1), ik))
      DO ipert = 1, npe
        mode = imode0 + ipert
        ps1(:, :) = czero
        ps2(:, :, :) = czero
        DO ib = 1, nbndep
          ijkb0 = 0
          DO nt = 1, ntyp
            DO na = 1, nat
              IF (ityp(na) == nt) THEN
                mu = 3 * (na - 1)
                DO ih = 1, nh(nt)
                  ikb = ijkb0 + ih
                  DO jh = 1, nh(nt)
                    jkb = ijkb0 + jh
                    DO ipol_ = 1, 3
                      IF (ABS(uu(mu + 1)) + ABS(uu(mu + 2)) + ABS(uu(mu + 3)) > 1.0d-12) THEN
                        ps1(ikb, ib) = ps1(ikb, ib) + deff(ih, jh, na) * &
                                       alphap(ipol_, ik)%k(jkb, ib) * uu(mu + ipol_)
                        ps2(ikb, ib, ipol_) = ps2(ikb, ib, ipol_) + deff(ih, jh, na) * &
                                       becp1(ik)%k(jkb, ib) * (0.d0, -1.d0) * uu(mu + ipol_) * tpiba
                      ENDIF
                    ENDDO
                  ENDDO
                ENDDO
                ijkb0 = ijkb0 + nh(nt)
              ENDIF
            ENDDO
          ENDDO
        ENDDO
        CALL ZGEMM('C', 'N', nbndep, nbndep, nkb, cone, becq(1, 1, 0), nkb, ps1, nkb, cone, &
                   elphmat(1, 1, ipert), nbndep)
        DO ipol_ = 1, 3
          CALL ZGEMM('C', 'N', nbndep, nbndep, nkb, cone, becq(1, 1, ipol_), nkb, ps2(1, 1, ipol_), nkb, &
                     cone, elphmat(1, 1, ipert), nbndep)
        ENDDO
      ENDDO
      END SUBROUTINE ep_gpu_k
      !
      COMPLEX(KIND = DP) FUNCTION uu(i)
      !! displacement pattern of the current mode, conjugated under time reversal (as vanilla)
      INTEGER, INTENT(IN) :: i
      IF (timerev) THEN
        uu = CONJG(u(i, mode))
      ELSE
        uu = u(i, mode)
      ENDIF
      END FUNCTION uu
    !
    !------------------------------------------------------------
    END SUBROUTINE ep_coarse_compute
    !------------------------------------------------------------
  !-----------------------------------------------------------------------------
  END MODULE ep_coarse
  !-----------------------------------------------------------------------------
