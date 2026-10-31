!
! rtsc: batched Sternheimer engine (groups/compute/STH-ENGINE.md, spec v1).
!
! Replaces one call of sternheimer_kernel (LR_Modules/response_kernels.f90) for the
! norm-conserving, collinear, nspin=1, lgauss, q /= 0 case: every column c = (k, pert, band
! n <= nocc_k) is solved at once on the GPU, with the algorithms of
!   apply_dpot_bands      dvpsi = dvbare + gather_kq( FFT( dvscf(r) * psi_kn(r) ) )
!   orthogonalize (lgauss) ps = evq^H dvpsi ; ps *= wwg ; dvpsi *= wg1 ; dvpsi = evq ps - dvpsi
!   cgsolve_all           per-column preconditioned CG, same update formulas and convergence
!                         test (sqrt(<h_diag g, g>) < ethr), same iteration count kter_eff
!   ch_psi_all (k)        (H_kq - e) x + alpha_pv P_v x, P_v over nocc_kq bands of k+q
!   h_psi                 g2kin x + Vloc x (FFT) + vkb D vkb^H x
!   cg_psi / h_prec       diagonal preconditioner
!   incdrhoscf            drho(r,p) += 2 wk/omega sum_n conj(psi_kn(r)) dpsi_c(r)
!
! No QE modules: plain arrays in, plain arrays out. CUDA Fortran, cuFFT, cuBLAS. All work runs
! on one CUDA stream; the only per-CG-step host round trip is the 4-byte active-column count.
!
! Column storage ("slots"): every column array is (npwx, nsl), nsl = nslot*npert*nksq, with
!   slot s = n + nslot*((p-1) + npert*(k-1)),  n = 1..nslot (nslot = max nocc_k), p, k.
! Slots with n > nocc_k are invalid and are kept at zero. Rows G > npwq(k) are kept at zero.
! Projections onto W_k = [vkb_k | evq_k] (npwx x (nkb+nbnd), zero-padded rows) are strided-batched
! ZGEMMs over k on all slots; FFT work runs only on the active columns.
!
MODULE sth_engine
  USE cudafor
  USE cufft
  USE cublas_v2
  USE cusolverDn
  USE sth_pfft, ONLY : pfft_setup, pfft_free, pfft_local, pfft_ok, pfft_info
  USE sth_pfft_s, ONLY : pfs_setup, pfs_free, pfs_local, pfs_ok
  USE iso_c_binding, ONLY : c_int, c_double, c_float
  IMPLICIT NONE
  PRIVATE
  !
  INTEGER, PARAMETER, PUBLIC :: sth_dp = SELECTED_REAL_KIND(14, 200)
  INTEGER, PARAMETER :: dp = sth_dp
  INTEGER, PARAMETER :: TPB = 256      ! threads per block, element-wise kernels
  INTEGER, PARAMETER :: RB = 256       ! max threads per block, per-column reductions (power of 2)
  INTEGER, PARAMETER :: MAXPERT = 6
  INTEGER, PARAMETER :: MAXPLAN = 12
  INTEGER, PARAMETER :: MAXE = 64      ! RTSC_STH_DEFL: at most this many known empty bands per k   ! FFT batch sizes 2**0 .. 2**(MAXPLAN-1)
  INTEGER, PARAMETER :: maxter = 400   ! as cgsolve_all
  !
  PUBLIC :: sth_engine_init, sth_engine_set_global, sth_engine_set_k, sth_engine_set_dvbare, &
            sth_engine_set_dpsi, sth_engine_setup, sth_engine_solve, sth_engine_get_dpsi, &
            sth_engine_get_hdiag, sth_engine_set_hdiag, sth_engine_reset_dpsi, sth_engine_free, sth_engine_shutdown, &
            sth_weights, sth_engine_ready, sth_engine_irrep, sth_engine_irrep_commit, sth_engine_qready, &
            sth_engine_becs, sth_mem_trim, sth_mem_sample, sth_mem_take_hw, sth_engine_xp_set, &
            sth_engine_set_nlk, sth_engine_build_dvb, sth_engine_dvb_check, &
            sth_engine_kqg_ready, sth_engine_set_kqg, sth_engine_set_wdev, sth_engine_wdev, sth_engine_cbsr_map, &
            sth_engine_rwf_count, sth_engine_rwf_put, sth_engine_kdev_map, sth_engine_set_wdev_n, sth_engine_kdev_op, sth_engine_cbsr_op, &
            sth_engine_set_npmax
  !
  ! Device memory. The NVHPC runtime (CUDA Fortran and OpenACC, NVCOMPILER_ACC_MEM_MANAGE, on by
  ! default) does not return a DEALLOCATEd device block to the driver: it keeps it on a free list and
  ! reuses it only for a later allocation of the same size. The engine and the nscf engine allocate
  ! sizes that change with every q (nksq) and irrep (npert), so the free lists grew q after q until
  ! cudaMemGetInfo saw almost nothing free (B1 frozen, q7: 8.75 of 79 GiB free, engine declined).
  ! sth_mem_trim returns the free lists to the driver (acc_clear_freelists, an NVHPC extension); it
  ! runs before every memory check and at every q-point boundary. sth_mem_sample records the
  ! high-water mark of device memory in use (driver view, total - free).
  INTERFACE
    SUBROUTINE acc_clear_freelists() BIND(C, name='acc_clear_freelists')
    END SUBROUTINE acc_clear_freelists
  END INTERFACE
  !
  ! cuFFTDx pruned fused local operator (rtsc/sth/fdx/sth_fftdx.cu; groups/compute/FFT-LOG.md), RTSC_STH_FFTDX=1.
  ! Built when CMake finds MathDx (RTSC_MATHDX_INC, RTSC_CUTLASS_INC); otherwise a stub whose fdx_setup returns
  ! "unsupported" and the engine keeps pfft / cuFFT.
  INTERFACE
    INTEGER(c_int) FUNCTION fdx_setup(nr, nrx, nksq, npwx, npwq, idxkq, chunk) BIND(C, name='fdx_setup')
      IMPORT :: c_int
      INTEGER(c_int) :: nr(3), nrx(3), npwq(*), idxkq(*)
      INTEGER(c_int), VALUE :: nksq, npwx, chunk
    END FUNCTION fdx_setup
    SUBROUTINE fdx_free() BIND(C, name='fdx_free')
    END SUBROUTINE fdx_free
    SUBROUTINE fdx_info(nst, nxa, chunk) BIND(C, name='fdx_info')
      IMPORT :: c_int
      INTEGER(c_int) :: nst, nxa, chunk
    END SUBROUTINE fdx_info
    INTEGER(c_int) FUNCTION fdx_local_d(x, y, list, nl, nslot, npert, npwx, nksq, nbnd, vrs, npwq, g2, etk, &
                                        scale, strm) BIND(C, name='fdx_local_d')
      IMPORT :: c_int, c_double, cuda_stream_kind
      COMPLEX(c_double), DEVICE :: x(*), y(*)
      INTEGER(c_int), DEVICE :: list(*), npwq(*)
      INTEGER(c_int), VALUE :: nl, nslot, npert, npwx, nksq, nbnd
      REAL(c_double), DEVICE :: vrs(*), g2(*), etk(*)
      REAL(c_double), VALUE :: scale
      INTEGER(cuda_stream_kind), VALUE :: strm
    END FUNCTION fdx_local_d
    INTEGER(c_int) FUNCTION fdx_local_s(x, y, list, nl, nslot, npert, npwx, nksq, nbnd, vrs, npwq, g2, etk, &
                                        scale, strm) BIND(C, name='fdx_local_s')
      IMPORT :: c_int, c_float, c_double, cuda_stream_kind
      COMPLEX(c_float), DEVICE :: x(*), y(*)
      INTEGER(c_int), DEVICE :: list(*), npwq(*)
      INTEGER(c_int), VALUE :: nl, nslot, npert, npwx, nksq, nbnd
      REAL(c_float), DEVICE :: vrs(*), g2(*)
      REAL(c_double), DEVICE :: etk(*)
      REAL(c_float), VALUE :: scale
      INTEGER(cuda_stream_kind), VALUE :: strm
    END FUNCTION fdx_local_s
    ! RTSC_STH_FDXDP (groups/compute/AMDAHL-LOG.md): apply_dpot and the inverse transform of incdrhoscf on the
    ! pruned path (FP64)
    INTEGER(c_int) FUNCTION fdx_dpot_d(d, psir, dv, list, s0, nb, nslot, npert, npwx, nnr, pj, scale, strm) &
                                       BIND(C, name='fdx_dpot_d')
      IMPORT :: c_int, c_double, cuda_stream_kind
      COMPLEX(c_double), DEVICE :: d(*), psir(*), dv(*)
      INTEGER(c_int), DEVICE :: list(*)
      INTEGER(c_int), VALUE :: s0, nb, nslot, npert, npwx, nnr, pj
      REAL(c_double), VALUE :: scale
      INTEGER(cuda_stream_kind), VALUE :: strm
    END FUNCTION fdx_dpot_d
    INTEGER(c_int) FUNCTION fdx_inv_d(box, x, list, s0, nb, nslot, npert, npwx, nnr, strm) BIND(C, name='fdx_inv_d')
      IMPORT :: c_int, c_double, cuda_stream_kind
      COMPLEX(c_double), DEVICE :: box(*), x(*)
      INTEGER(c_int), DEVICE :: list(*)
      INTEGER(c_int), VALUE :: s0, nb, nslot, npert, npwx, nnr
      INTEGER(cuda_stream_kind), VALUE :: strm
    END FUNCTION fdx_inv_d
  END INTERFACE
  INTEGER(8), SAVE :: mem_hw = 0
  !
  ! run statistics of the last solve (seconds; timers need sth_timing = .true.)
  LOGICAL, PUBLIC, SAVE :: sth_timing = .FALSE.
  REAL(dp), PUBLIC, SAVE :: sth_t_dpot = 0, sth_t_ortho = 0, sth_t_cg = 0, sth_t_drho = 0, &
                            sth_t_total = 0, sth_t_setup = 0
  INTEGER, PUBLIC, SAVE :: sth_nsteps = 0
  INTEGER(8), PUBLIC, SAVE :: sth_ncolapp = 0      ! column-operator applications in the CG
  ! rtsc solve-work profile (sth_timing): valid columns, columns active at the first test, initial
  ! residual sqrt(<g|h_diag|g>)/thresh histogram over the active columns (bins of 1/2 decade from
  ! 10**0, last bin open), and the time of the apply_op parts (projector GEMMs vs FFT loop)
  INTEGER, PUBLIC, SAVE :: sth_nvalid = 0, sth_nact0 = 0
  INTEGER, PUBLIC, SAVE :: sth_r0hist(0:11) = 0
  ! CG tests per column by distance of the band from the top occupied band of its k (0..5, 6 = more)
  REAL(dp), PUBLIC, SAVE :: sth_stband(0:6) = 0
  ! same, mean log10(r0/thresh) per band bin; and tests / log10(r0/thresh) by the smallest known gap
  ! e_m(k+q) - e_n(k), m = nockq+1..nbnd: < 0.02, 0.02-0.1, 0.1-0.5, > 0.5 Ry, none known (bins 1..5)
  REAL(dp), PUBLIC, SAVE :: sth_r0band(0:6) = 0, sth_gapst(5) = 0, sth_gapdec(5) = 0, sth_gapn(5) = 0
  INTEGER, DEVICE, ALLOCATABLE :: d_nst(:)
  REAL(dp), PUBLIC, SAVE :: sth_r0med = 0, sth_t_gemm = 0, sth_t_fft = 0
  LOGICAL, PUBLIC, SAVE :: sth_prof = .FALSE.      ! sync inside apply_op to time GEMMs vs FFTs
  !
  ! rtsc dpsi extrapolation across the SCF iterations of one irrep (RTSC_STH_XP, SOLVE-LOG.md).
  ! dpsi(dV_in) is affine in dV_in (one fixed operator per column), so when the adapter finds
  ! dV_in(k) - dV_in(k-1) ~ sum_j c_j [dV_in(j) - dV_in(j-1)], the solution moves by about
  ! sum_j c_j [dpsi(j) - dpsi(j-1)]. The call starts CG from dpsi(k-1) + that sum and solves to the
  ! unchanged thresh, so only the starting point changes. History of dpsi increments: pinned host
  ! memory (sth_xp_m slots); one extra device column array holds dpsi at the start of a call.
  ! relative stopping test for non-final SCF calls (RTSC_STH_ETA, set per call by the adapter): a column
  ! also stops when its residual has fallen below sth_eta x its residual at the first test; 0 = off
  REAL(dp), PUBLIC, SAVE :: sth_eta = 0.0_dp
  REAL(dp), DEVICE, ALLOCATABLE :: d_r0(:)
  ! RTSC_STH_DEFL: additive spectral preconditioner with the known empty bands at k+q (k_check_defl)
  INTEGER, PUBLIC, SAVE :: sth_defl = 0
  ! rtsc b2 diagnostic (RTSC_STH_TRUERES=1, DEFL path only): stop on the plain residual norm, and when every column
  ! has stopped, recompute the true residual A dpsi - rhs once and restart the CG on the columns that fail (<= 3 times)
  INTEGER, SAVE :: sth_trn = -1
  INTEGER(8), SAVE :: trn_rechk = 0, trn_fail = 0
  REAL(dp), PUBLIC, SAVE :: sth_dfloor = 1.0e-4_dp
  REAL(dp), DEVICE, ALLOCATABLE :: d_etkq(:,:)
  INTEGER, PUBLIC, SAVE :: sth_xp_m = 0            ! history slots; 0 = off (set before irrep commit)
  INTEGER, SAVE :: xp_nc = 0, xp_rec = 0, xp_slot(16) = 0
  COMPLEX(dp), SAVE :: xp_c(16) = (0.0_dp, 0.0_dp)
  COMPLEX(dp), PINNED, ALLOCATABLE, SAVE :: h_xh(:,:,:)
  COMPLEX(dp), DEVICE, ALLOCATABLE :: d_xp(:,:)
  INTEGER, PUBLIC, SAVE :: sth_bmax = 256          ! FFT chunk (largest batch); set before init
  INTEGER, PUBLIC, SAVE :: sth_debug_slot = 0      ! >0: print the CG history of this slot
  INTEGER, PUBLIC, SAVE :: sth_opvar = 0           ! testing: 1 = (g2 - e) x + Vx (same operator, other rounding)
  INTEGER, PUBLIC, SAVE :: sth_rb = 128            ! threads per column reduction (power of 2, <= 256)
  ! pruned fused local operator (rtsc/sth/sth_pfft.f90; KERNEL-LOG lever 1), set before setup (per q).
  ! sth_pfft_check: at the first operator application of each q, run the cuFFT path too and print the
  ! difference of the local part (diagnostic; costs one extra application).
  LOGICAL, PUBLIC, SAVE :: sth_pfft_req = .FALSE., sth_pfft_check = .FALSE.
  LOGICAL, SAVE :: pfft_checked = .FALSE.
  ! cuFFTDx variant (takes precedence over pfft when both are requested and the grid is supported);
  ! sth_fftdx_check: as sth_pfft_check, for fftdx against the cuFFT path
  LOGICAL, PUBLIC, SAVE :: sth_fftdx_req = .FALSE., sth_fftdx_check = .FALSE.
  LOGICAL, SAVE :: fdx_on = .FALSE.
  ! rtsc amdahl (groups/compute/AMDAHL-LOG.md), set before setup (per q):
  ! sth_fdxdp_req (RTSC_STH_FDXDP): apply_dpot, the DVB local part and incdrhoscf's inverse FFT on the pruned fftdx path
  !   (needs fdx_on); sth_fdxdp_check: at the first non-trivial call of each q, run the box path too and print the
  !   differences; sth_obatch (RTSC_STH_OBATCH): orthogonalize with two strided-batched ZGEMMs (ortho_batched) in FP64
  LOGICAL, PUBLIC, SAVE :: sth_fdxdp_req = .FALSE., sth_fdxdp_check = .FALSE., sth_obatch = .FALSE.
  LOGICAL, SAVE :: dp_on = .FALSE., dp_checked = .FALSE., dr_checked = .FALSE.
  REAL(dp), PUBLIC, SAVE :: sth_t_fdxsetup = 0
  ! dV_bare psi built in the engine (KERNEL-LOG lever 4; RTSC_STH_DVB=1), set before init (per q):
  ! becp1 / alphap of every k (staged by sth_engine_set_nlk), D becp1 and D alphap on the device
  LOGICAL, PUBLIC, SAVE :: sth_dvb_req = .FALSE.
  LOGICAL, SAVE :: dvb_q = .FALSE.
  COMPLEX(dp), ALLOCATABLE, SAVE :: h_bp(:,:,:), h_al(:,:,:,:)
  COMPLEX(dp), DEVICE, ALLOCATABLE :: d_db(:,:,:), d_da(:,:,:,:)
  !
  ! rtsc setup (groups/compute/SETUP-LOG.md; RTSC_STH_QCACHE=1, set before init, per q). Same values, less setup work:
  ! - host stages h_dvb / h_dpsi0 allocated only when written (DVB on and a first iteration write neither: the device
  !   arrays are zeroed instead of uploading zeros);
  ! - the irrep work arrays kept across the irreps of one q when their shape (npert) is unchanged;
  ! - the k+q+G table (npwx, 3, nksq) cached on the device per q (dvb_build, becs), if it fits (else per call);
  ! - W (vkb at k+q, from the device; evq) and evc written straight into d_W / d_evc per k (no host h_W / h_evc),
  !   and d_inv built on the device.
  ! All of it is per-q state, freed by sth_engine_free at the q boundary.
  LOGICAL, PUBLIC, SAVE :: sth_qcache = .FALSE.
  LOGICAL, SAVE :: qc_q = .FALSE., kqg_q = .FALSE., wdev_q = .FALSE., kqg_said = .FALSE.
  REAL(dp), DEVICE, ALLOCATABLE :: d_kqgq(:,:,:)
  ! setup clocks (sth_timing): commit = free+trim+check, alloc, uploads, valid list + zeroing; dvb = local, nonlocal
  REAL(dp), PUBLIC, SAVE :: sth_t_commit(4) = 0, sth_t_dvb(2) = 0
  !
  LOGICAL, SAVE :: ready = .FALSE., staged = .FALSE., qready = .FALSE.
  INTEGER, SAVE :: nksq, npwx, nbnd, nkb, npert, nslot, nproj, nsl, ngrp, nnr, nvalid, nbox
  INTEGER, SAVE :: nr(3), nrx(3)
  REAL(dp), SAVE :: alpha_pv, omega
  !
  ! host staging (filled by set_* calls, uploaded by setup)
  INTEGER, ALLOCATABLE, SAVE :: h_npw(:), h_npwq(:), h_nock(:), h_nockq(:), h_idxk(:,:), h_idxkq(:,:)
  REAL(dp), ALLOCATABLE, SAVE :: h_wk(:), h_g2(:,:), h_etk(:,:), h_wwg(:,:,:), h_wg1(:,:), h_etkq(:,:)
  COMPLEX(dp), ALLOCATABLE, SAVE :: h_W(:,:,:), h_evc(:,:), h_dvb(:,:), h_dpsi0(:,:), h_dpsi(:,:)
  LOGICAL, SAVE :: have_dpsi0 = .FALSE., h_dpsi_valid = .FALSE.
  INTEGER, ALLOCATABLE, SAVE :: h_valid(:)
  !
  ! device state
  INTEGER, DEVICE, ALLOCATABLE :: d_npw(:), d_npwq(:), d_nock(:), d_nockq(:), d_idxk(:,:), d_idxkq(:,:)
  INTEGER, DEVICE, ALLOCATABLE :: d_inv(:,:)          ! FFT index -> k+q plane-wave index (0: outside)
  INTEGER, DEVICE, ALLOCATABLE :: d_valid(:), d_act(:), d_act2(:), d_conv(:), d_cnt(:), d_nact(:)
  REAL(dp), DEVICE, ALLOCATABLE :: d_g2(:,:), d_etk(:,:), d_hd(:,:,:), d_wwg(:,:,:), d_wg1(:,:)
  REAL(dp), DEVICE, ALLOCATABLE :: d_vrs(:), d_wgt(:), d_rho(:), d_kter(:), d_av(:), d_cv(:)
  COMPLEX(dp), DEVICE, ALLOCATABLE :: d_W(:,:,:), d_dmat(:,:), d_psir(:,:), d_dvb(:,:), d_d0(:,:)
  COMPLEX(dp), DEVICE, ALLOCATABLE :: d_dpsi(:,:), d_g(:,:), d_h(:,:), d_t(:,:), d_cf(:,:), d_cf2(:,:)
  COMPLEX(dp), DEVICE, ALLOCATABLE :: d_box(:,:), d_dv(:,:), d_drho(:,:), d_evc(:,:), d_ps(:,:)
  !
  INTEGER(KIND=cuda_stream_kind), SAVE :: strm
  TYPE(cublasHandle), SAVE :: hblas
  LOGICAL, SAVE :: have_handles = .FALSE.
  INTEGER, SAVE :: plans(0:MAXPLAN-1) = 0, nplan = 0, plan_nnr = 0, plan_nr(3) = 0
  ! psi_k(r) cache (nnr x nslot*nksq) or, when it does not fit (or RTSC_STH_NOPSIR=1), psi_k(r)
  ! rebuilt per column chunk from evc into d_box2 (one extra batched inverse FFT per chunk)
  LOGICAL, SAVE :: psir_cached = .TRUE.
  ! rtsc b2r (P5 memory, RTSC_STH_PSIR_PART=1; off by default): a partial psi(r) cache. psi_kn(r) is cached for the first
  ! psir_kc k-points (as many as fit after everything else of the q, CBSR bases included) and recomputed per chunk for the
  ! rest. psir_nv = the valid slots (k-major) whose k is cached. Full cache: psir_kc = nksq; none: psir_kc = 0.
  LOGICAL, SAVE :: psir_part = .FALSE.
  INTEGER, SAVE :: psir_kc = 0, psir_nv = 0, sth_npmax = 3
  COMPLEX(dp), DEVICE, ALLOCATABLE :: d_box2(:,:)
  INTEGER, PINNED, ALLOCATABLE, SAVE :: h_nact(:)
  !
  ! Mixed precision (sth_mp, RTSC_STH_MP=1; groups/compute/MIXED-LOG.md). Defect correction per
  ! column on A = H_kq - e_n + alpha_pv P_v, which is fixed for all SCF calls of one irrep:
  !   FP64 state  x = dpsi, b = dvpsi (dV psi + orthogonalize, FP64 as before), Ax (kept across calls)
  !   round:      r = Ax - b (FP64); converged if sqrt(<h_diag r, r>) < thresh (the cgsolve_all test,
  !               on the true FP64 residual); otherwise solve A e = -r by the cgsolve_all PCG in FP32
  !               (C2C cuFFT, CGEMM, FP32 W/vrs/D/h_diag; per-column dots accumulated in FP64) to
  !               max(sth_mp_safe*thresh, sth_mp_eta*|r|), then x += e and Ax = A x in FP64.
  ! A call therefore costs one FP64 operator application per round (usually one) plus the FP32 CG
  ! steps, against 1 + (CG steps) FP64 applications on the FP64 path. dvpsi, orthogonalize and
  ! incdrhoscf (drho) are unchanged and FP64.
  INTEGER, PARAMETER :: sp = SELECTED_REAL_KIND(6, 37)
  LOGICAL, PUBLIC, SAVE :: sth_mp = .FALSE.        ! set before sth_engine_init (per q)
  REAL(dp), PUBLIC, SAVE :: sth_mp_eta = 1.0e-4_dp, sth_mp_safe = 0.5_dp
  INTEGER, PUBLIC, SAVE :: sth_mp_maxround = 10
  INTEGER(8), PUBLIC, SAVE :: sth_n64 = 0, sth_n32 = 0   ! FP64 / FP32 column applications, last solve
  INTEGER, PUBLIC, SAVE :: sth_nround = 0
  REAL(dp), PUBLIC, SAVE :: sth_t_cg64 = 0                ! FP64 applications inside the MP CG (sth_timing)
  LOGICAL, SAVE :: mp_q = .FALSE.                  ! the per-q state has the FP32 copies
  LOGICAL, SAVE :: ax_valid = .FALSE.              ! d_g holds A d_dpsi (FP64) on every valid slot
  INTEGER, SAVE :: plans32(0:MAXPLAN-1) = 0, nplan32 = 0
  REAL(sp), DEVICE, ALLOCATABLE :: s_vrs(:), s_hd(:,:,:), s_g2(:,:)
  REAL(dp), DEVICE, ALLOCATABLE :: d_tol(:)
  INTEGER, DEVICE, ALLOCATABLE :: d_iconv(:), d_oact(:)
  COMPLEX(sp), DEVICE, ALLOCATABLE :: s_W(:,:,:), s_dmat(:,:), s_box(:,:)
  COMPLEX(sp), DEVICE, ALLOCATABLE :: s_e(:,:), s_g(:,:), s_h(:,:), s_t(:,:), s_cf(:,:), s_cf2(:,:)
  !
  ! rtsc LIFT-F preconditioner (RTSC_STH_LIFT = E_c' in Ry, 0 = off; groups/theory/returns/M3/1.md with the referee
  ! corrections of referees/M3/1-A.md and 1-B.md; SOLVE-LOG.md "LIFT-F canary"). FP64 CG path only (not under MP).
  ! Per k+q (fixed for the whole q): coarse set L = {G : |k+q+G|^2 < E_c'} (N' = |L|), H = the rest. All energies are
  ! referred to V0 = the G = 0 component of vrs (gauge-free: A' = A~ - V0, e~_n = e_n - V0), A~ = H_kq + alpha_pv P_v.
  !   Y_H = -T_H^-1 A~_HL ;  S0 = A'_LL - A~_LH T_H^-1 A~_HL + alpha Z Z^H, Z = Y_H^H Psi_v,H (the alpha P_v part of
  !   the dropped remainder, kept exactly) ;  B0 = I + Y_H^H Y_H ;  S0 C = B0 C Lambda, C^H B0 C = I ;  Q = Y C.
  ! M_n^-1 g = Q diag(1/max(Lambda - e~_n, delta)) Q^H g + D_n g, D_n = 1/max(T_G - e~_n, T_G/2) on H, 0 on L.
  ! The stopping test stays sqrt(<g|h_diag|g>) < thresh; rho = <g|M^-1 g>. Q is stored in FP32 by chunks of k+q
  ! (device-resident when they fit, else pinned host chunks streamed per application; RTSC_STH_LIFT_KCHUNK forces it).
  REAL(dp), PUBLIC, SAVE :: sth_lift = 0.0_dp, sth_lift_delta = 0.02_dp
  INTEGER, PUBLIC, SAVE :: sth_lift_nmax = 1500, sth_lift_kchunk = 0
  REAL(dp), PUBLIC, SAVE :: sth_t_lbuild = 0, sth_t_lapply = 0
  INTEGER, PUBLIC, SAVE :: sth_lift_ncol = 0, sth_lift_nclamp = 0, sth_lift_nout = 0
  ! LIFT-F v2 (RTSC_STH_LIFT_FB=1; SOLVE-LOG.md "LIFT-F v2" pre-registration): per-column fallback to TPA (h_diag) where the
  ! pencil's lowest value is unreliable: no computed empty band at k+q (a), g_n <= 0 (b), clamp (c), or
  ! (Lambda_1 - e~_n)/g_n outside [1/3, 3] (d). lok(n, k) = 1: LIFT-F; 0: TPA. Fixed per q.
  INTEGER, PUBLIC, SAVE :: sth_lift_fb = 0
  INTEGER, PUBLIC, SAVE :: sth_lift_nfb(4) = 0
  INTEGER, DEVICE, ALLOCATABLE :: d_lok(:,:)
  !
  ! rtsc CBSR (RTSC_STH_CBSR=1; SOLVE-LOG.md "CBSR canary" and session 4 "production"). Exact spectral solve of the engine's
  ! own system A~ = H_kq + alpha_pv P_v (nscf evq, energy zero V0): A~ - V0 = U Lambda U^H, and each SCF call replaces the
  ! CG by dpsi_n = U diag(1/(Lambda - e~_n)) U^H b_n on the engine's orthogonalised right-hand side (QE's system solved
  ! exactly; referee B's theta-weighted closed form with the nscf psi as the single psi source).
  ! Production path:
  ! - U is built on the GPU (dense assembly kernel + cuBLAS + cuSOLVER zheevd) once per RTSC_NSCF_SYMROT cache entry
  !   (one per full-group orbit of k-points) and kept on the device across q (cb_rep, in the entry's own G order).
  !   Every k+q of a q that SYMROT rotated from entry i0 uses U_i0 through the rotation's G map (imap, phase, time
  !   reversal; symrot_kmap): b is gathered into the entry's G order, solved, and scattered back. A k+q without a map
  !   (diagonalized itself, or q = Gamma, which reads the scf psi) gets a per-q basis (cb_tmp), built the same way.
  ! - The ZGEMMs run batched over every column that shares a basis (all k+q of the q in its star, all patterns and
  !   bands): columns are grouped per basis, padded to a multiple of 16, and bases of equal padded width go into one
  !   cublasZgemmBatched.
  LOGICAL, PUBLIC, SAVE :: sth_cbsr = .FALSE., sth_cbsr_skip = .FALSE.
  REAL(dp), PUBLIC, SAVE :: sth_t_cbuild = 0, sth_t_csolve = 0, sth_t_ceig = 0
  LOGICAL, SAVE :: cbsr_q = .FALSE.
  REAL(dp), SAVE :: cbsr_v0 = 0.0_dp
  INTEGER, PARAMETER :: CB_MAXREP = 8192
  TYPE cb_slot_t
    COMPLEX(dp), DEVICE, ALLOCATABLE :: u(:,:)
    COMPLEX(sp), DEVICE, ALLOCATABLE :: u32(:,:)   ! RTSC_STH_CBSR_RF: tier 1 (columns cb_m0+1..n0) in FP32; u holds tier 0
  END TYPE cb_slot_t
  TYPE(cb_slot_t), SAVE :: cb_rep(CB_MAXREP)
  TYPE(cb_slot_t), ALLOCATABLE, SAVE :: cb_tmp(:)
  REAL(dp), ALLOCATABLE, SAVE :: h_cblam(:,:), h_cbtlam(:,:)
  INTEGER, SAVE :: cb_n0 = 0, cb_nres = 0, cb_nbuilt = 0, cb_qcount = 0, cb_lastq(CB_MAXREP) = 0
  ! per-q map of every k+q onto its basis (sth_engine_cbsr_map): rep id (0: per-q basis), time reversal, G map, phase
  INTEGER, ALLOCATABLE, SAVE :: h_ckrep(:), h_cktr(:), h_ckmap(:,:), h_ckq(:), h_ckdev(:), h_ckop(:,:)
  COMPLEX(dp), ALLOCATABLE, SAVE :: h_ckph(:,:)
  INTEGER, DEVICE, ALLOCATABLE :: d_ckmap(:,:), d_cktr(:)
  COMPLEX(dp), DEVICE, ALLOCATABLE :: d_ckph(:,:)
  ! per-q bases of this q (nqs): rep id (> 0) or -tmp index; per-irrep plan of gathered columns and batched GEMMs
  INTEGER, SAVE :: cb_nqs = 0, cb_npl = -1, cb_ctot = 0, cb_ngrp = 0, cb_ntmp = 0, cb_pad = 0
  INTEGER, ALLOCATABLE, SAVE :: cb_qs(:), cb_gN(:), cb_gnb(:), cb_goff(:)
  INTEGER, DEVICE, ALLOCATABLE :: d_cgsc(:), d_cgl(:)
  REAL(dp), DEVICE, ALLOCATABLE :: d_cqlam(:,:)
  TYPE(c_devptr), DEVICE, ALLOCATABLE :: d_pA(:), d_pX(:), d_pY(:)
  COMPLEX(dp), DEVICE, ALLOCATABLE :: d_cx(:,:), d_cy(:,:)
  ! build workspace (GPU assembly + cuSOLVER)
  COMPLEX(dp), DEVICE, ALLOCATABLE :: d_ca(:,:), d_cvg(:), d_cwork(:), d_cx1(:,:)
  REAL(dp), DEVICE, ALLOCATABLE :: d_cw(:)
  INTEGER, DEVICE, ALLOCATABLE :: d_cinfo(:)
  TYPE(cusolverDnHandle), SAVE :: hsolv
  LOGICAL, SAVE :: have_solv = .FALSE.
  INTEGER, SAVE :: cb_lwork = 0
  !
  ! rtsc b2r: CBSR in refinement form (RTSC_STH_CBSR_RF=1 with RTSC_STH_CBSR=1; SOLVE-LOG.md session 5, theory M6-A with the
  ! referees' fixes). Each SYMROT entry's basis is stored compressed: its lowest cb_m0 eigenvectors (tier 0: the energy
  ! window around the occupied energies, both the alpha_pv-shifted occupied and the empty states) in FP64 (u), the rest
  ! in FP32 (u32). Both projectors stay those of the engine's own FP64 operator: A~ is assembled from the engine's evq and
  ! the residual is the engine's apply_op. R~ = U0 D0 U0^H + U1 D1 U1^H is a solver step under QE's own FP64 test:
  !   g = A dpsi - b (A dpsi kept from the previous step / call in d_h), dpsi -= R~ g, A dpsi by apply_op (one FP64
  !   application per step), sqrt(<g|h_diag|g>) < thresh on EVERY column (the per-column guard), failing columns take
  !   another step (at most cb_rfmax).
  ! Columns are processed one chunk of whole bases (stars) at a time, at most cb_cmax gathered columns per chunk.
  LOGICAL, SAVE :: cbsr_rf = .FALSE., cbsr_rf_read = .FALSE., cb_tf32 = .FALSE., cb_chk = .FALSE., cb_act = .TRUE.
  INTEGER, SAVE :: cb_m0 = 64, cb_cmax = 8192, cb_rfmax = 6, cb_nch = 0, cb_nbt = 0, cb_cw = 0
  REAL(dp), SAVE :: cb_d0win = 1.0_dp
  INTEGER, ALLOCATABLE, SAVE :: cb_chb0(:), cb_chb1(:), cb_chc0(:), cb_chnc(:), cb_bN(:), cb_bnb(:), cb_boff(:)
  TYPE(c_devptr), DEVICE, ALLOCATABLE :: d_pA32(:), d_pX32(:), d_pY32(:)
  COMPLEX(sp), DEVICE, ALLOCATABLE :: d_cx32(:,:), d_cy32(:,:)
  COMPLEX(dp), DEVICE, ALLOCATABLE :: d_ctmp(:,:), d_cchk(:,:)
  REAL(dp), SAVE :: sth_t_crf = 0, sth_t_crfg = 0
  INTEGER(8), SAVE :: rf_conv(0:8) = 0, rf_gcols = 0
  ! pre-check statistics (cumulative over the run): evq residuals, denominators, tier-1 window
  REAL(dp), SAVE :: pc_resmax_o = 0, pc_resmax_e = 0, pc_win = HUGE(1.0_dp)
  INTEGER(8), SAVE :: pc_res(0:7) = 0, pc_dun(0:6) = 0, pc_doc(0:6) = 0, pc_dla(0:6) = 0
  !
  ! rtsc KDEV (RTSC_STH_KDEV=1, with QCACHE; SOLVE-LOG.md session 4): the adapter's per-k loop on the GPU. evq / evc of every k
  ! whose k and k+q are RTSC_NSCF_SYMROT images are built here by the same coefficient rotation SYMROT applies on the host,
  ! from a device copy of the SYMROT cache (d_rwf, kept across q); vkb comes from vkb_batch (sth_engine_set_wdev_n).
  COMPLEX(dp), DEVICE, ALLOCATABLE :: d_rwf(:,:,:)
  INTEGER, SAVE :: rwf_n = 0, rwf_ld = 0, rwf_nb = 0
  INTEGER, PUBLIC, SAVE :: sth_kdev_nrot = 0
  INTEGER, ALLOCATABLE, SAVE :: h_kdrep(:,:), h_kdtr(:,:), h_kdft(:,:), h_kdmap(:,:,:)
  ! KDEV v2 (RTSC_STH_KDEV=2): maps built on the device from the cache's Miller indices and the operation (S, tr, G0)
  INTEGER, DEVICE, ALLOCATABLE :: d_rmill(:,:,:)
  INTEGER, ALLOCATABLE, SAVE :: h_kdop(:,:,:)       ! (12, 2, nksq): S (9), G0 (3)
  LOGICAL, SAVE :: kd_v2 = .FALSE.
  COMPLEX(dp), ALLOCATABLE, SAVE :: h_kdph(:,:,:)
  LOGICAL, SAVE :: lift_q = .FALSE., lift_stream = .FALSE.
  INTEGER, SAVE :: lift_nm = 0, lift_kc = 0
  REAL(dp), SAVE :: lift_v0 = 0.0_dp
  COMPLEX(sp), DEVICE, ALLOCATABLE :: d_lq(:,:,:), d_lg(:,:), d_lz(:,:), d_lc(:,:)
  COMPLEX(sp), PINNED, ALLOCATABLE, SAVE :: h_lq(:,:,:)
  REAL(dp), DEVICE, ALLOCATABLE :: d_llam(:,:), d_lt(:,:)
  !
CONTAINS
  !
  !=================================================================================================
  ! device kernels
  !=================================================================================================
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_kqg_scale(t, x, kqg, ipol, npwx, nslot, npert, nksq, ntot)
    ! t(:, s) = x(:, s) * kqg(:, ipol, k(s)) for every slot s
    INTEGER, VALUE :: ipol, npwx, nslot, npert, nksq
    INTEGER(8), VALUE :: ntot
    COMPLEX(dp) :: t(*), x(*)
    REAL(dp) :: kqg(npwx, 3, nksq)
    INTEGER(8) :: i
    INTEGER :: ig, s, k
    i = INT(blockIdx%x - 1, 8) * blockDim%x + threadIdx%x
    DO WHILE (i <= ntot)
      ig = INT(MOD(i - 1, INT(npwx, 8))) + 1
      s = INT((i - 1) / npwx) + 1
      k = (s - 1) / (nslot * npert) + 1
      t(i) = x(i) * kqg(ig, ipol, k)
      i = i + INT(blockDim%x, 8) * gridDim%x
    ENDDO
  END SUBROUTINE k_kqg_scale
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_zero(a, n)
    INTEGER(8), VALUE :: n
    COMPLEX(dp) :: a(n)
    INTEGER(8) :: i
    i = INT(blockIdx%x - 1, 8) * blockDim%x + threadIdx%x
    DO WHILE (i <= n)
      a(i) = (0.0_dp, 0.0_dp)
      i = i + INT(blockDim%x, 8) * gridDim%x
    ENDDO
  END SUBROUTINE k_zero
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_copy(a, b, n)
    ! a = b
    INTEGER(8), VALUE :: n
    COMPLEX(dp) :: a(n), b(n)
    INTEGER(8) :: i
    i = INT(blockIdx%x - 1, 8) * blockDim%x + threadIdx%x
    DO WHILE (i <= n)
      a(i) = b(i)
      i = i + INT(blockDim%x, 8) * gridDim%x
    ENDDO
  END SUBROUTINE k_copy
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_sub(a, b, n)
    ! a = a - b
    INTEGER(8), VALUE :: n
    COMPLEX(dp) :: a(n), b(n)
    INTEGER(8) :: i
    i = INT(blockIdx%x - 1, 8) * blockDim%x + threadIdx%x
    DO WHILE (i <= n)
      a(i) = a(i) - b(i)
      i = i + INT(blockDim%x, 8) * gridDim%x
    ENDDO
  END SUBROUTINE k_sub
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_axpyc(y, x, a, n)
    ! y = y + a x
    INTEGER(8), VALUE :: n
    COMPLEX(dp), VALUE :: a
    COMPLEX(dp) :: y(n), x(n)
    INTEGER(8) :: i
    i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    DO WHILE (i <= n)
      y(i) = y(i) + a * x(i)
      i = i + INT(blockDim%x, 8) * gridDim%x
    ENDDO
  END SUBROUTINE k_axpyc
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_izero(a, n)
    INTEGER(8), VALUE :: n
    INTEGER :: a(n)
    INTEGER(8) :: i
    i = (blockIdx%x - 1) * INT(blockDim%x, 8) + threadIdx%x
    DO WHILE (i <= n)
      a(i) = 0
      i = i + INT(gridDim%x, 8) * blockDim%x
    ENDDO
  END SUBROUTINE k_izero
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_inv(inv, idxkq, npwq, nnr, npwx)
    !! inv(idxkq(ig, k), k) = ig for ig <= npwq(k); one block row per k (blockIdx%y)
    INTEGER, VALUE :: nnr, npwx
    INTEGER :: inv(nnr, *), idxkq(npwx, *), npwq(*)
    INTEGER :: ig, ik
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    ik = blockIdx%y
    IF (ig <= npwq(ik)) inv(idxkq(ig, ik), ik) = ig
  END SUBROUTINE k_inv
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_icopy(a, b, n)
    INTEGER, VALUE :: n
    INTEGER :: a(n), b(n)
    INTEGER :: i
    i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (i <= n) a(i) = b(i)
  END SUBROUTINE k_icopy
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_scatter(box, x, list, s0, nnr, npwx, nslot, npert, nksq, npwq, idxq)
    ! box(:, j) (pre-zeroed) <- x(:, list(s0+j)) placed at the k+q FFT indices
    INTEGER, VALUE :: s0, nnr, npwx, nslot, npert, nksq
    COMPLEX(dp) :: box(nnr, *), x(npwx, *)
    INTEGER :: list(*), npwq(nksq), idxq(npwx, nksq)
    INTEGER :: j, s, k, ig
    j = blockIdx%y
    s = list(s0 + j)
    k = (s - 1) / (nslot * npert) + 1
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ig <= npwq(k)) box(idxq(ig, k), j) = x(ig, s)
  END SUBROUTINE k_scatter
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_scatter_full(box, x, list, s0, nnr, npwx, nslot, npert, nksq, inv)
    ! box(:, j) <- x(:, list(s0+j)) at the k+q FFT indices, zero elsewhere (no separate zeroing pass)
    INTEGER, VALUE :: s0, nnr, npwx, nslot, npert, nksq
    COMPLEX(dp) :: box(nnr, *), x(npwx, *)
    INTEGER :: list(*), inv(nnr, nksq)
    INTEGER :: j, s, k, ir, ig
    j = blockIdx%y
    s = list(s0 + j)
    k = (s - 1) / (nslot * npert) + 1
    ir = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ir <= nnr) THEN
      ig = inv(ir, k)
      IF (ig > 0) THEN
        box(ir, j) = x(ig, s)
      ELSE
        box(ir, j) = (0.0_dp, 0.0_dp)
      ENDIF
    ENDIF
  END SUBROUTINE k_scatter_full
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_vmul(box, v, nnr)
    INTEGER, VALUE :: nnr
    COMPLEX(dp) :: box(nnr, *)
    REAL(dp) :: v(nnr)
    INTEGER :: ir, j
    j = blockIdx%y
    ir = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ir <= nnr) box(ir, j) = box(ir, j) * v(ir)
  END SUBROUTINE k_vmul
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_gather_h(y, x, box, list, s0, nnr, npwx, nslot, npert, nksq, nbnd, &
                                           npwq, idxq, g2, etk, scale, opvar)
    ! y = Vloc x (from the forward-FFT box, scaled) + g2kin x - e_n x ; rows > npwq zeroed.
    ! The nonlocal and P_v terms are added afterwards by a ZGEMM.
    INTEGER, VALUE :: s0, nnr, npwx, nslot, npert, nksq, nbnd, opvar
    REAL(dp), VALUE :: scale
    COMPLEX(dp) :: y(npwx, *), x(npwx, *), box(nnr, *)
    INTEGER :: list(*), npwq(nksq), idxq(npwx, nksq)
    REAL(dp) :: g2(npwx, nksq), etk(nbnd, nksq)
    INTEGER :: j, s, k, n, ig
    COMPLEX(dp) :: xv
    j = blockIdx%y
    s = list(s0 + j)
    k = (s - 1) / (nslot * npert) + 1
    n = MOD(s - 1, nslot) + 1
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ig <= npwx) THEN
      IF (ig <= npwq(k)) THEN
        xv = x(ig, s)
        IF (opvar == 0) THEN
          y(ig, s) = g2(ig, k) * xv + box(idxq(ig, k), j) * scale - etk(n, k) * xv
        ELSE
          y(ig, s) = (g2(ig, k) - etk(n, k)) * xv + box(idxq(ig, k), j) * scale
        ENDIF
      ELSE
        y(ig, s) = (0.0_dp, 0.0_dp)
      ENDIF
    ENDIF
  END SUBROUTINE k_gather_h
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_coef_h(cf2, cf, dmat, nproj, nkb, nslot, npert, nksq, nockq, alpha)
    ! operator coefficients: cf2(1:nkb) = D cf(1:nkb) (add_vuspsi);
    ! cf2(nkb+m) = alpha_pv cf(nkb+m) for m <= nocc_kq, 0 otherwise (ch_psi_all_k)
    INTEGER, VALUE :: nproj, nkb, nslot, npert, nksq
    REAL(dp), VALUE :: alpha
    COMPLEX(dp) :: cf2(nproj, *), cf(nproj, *), dmat(nkb, nkb)
    INTEGER :: nockq(nksq)
    INTEGER :: i, s, k, j, m
    COMPLEX(dp) :: acc
    s = blockIdx%x
    i = threadIdx%x
    IF (i > nproj) RETURN
    k = (s - 1) / (nslot * npert) + 1
    IF (i <= nkb) THEN
      acc = (0.0_dp, 0.0_dp)
      DO j = 1, nkb
        acc = acc + dmat(i, j) * cf(j, s)
      ENDDO
      cf2(i, s) = acc
    ELSE
      m = i - nkb
      IF (m <= nockq(k)) THEN
        cf2(i, s) = alpha * cf(i, s)
      ELSE
        cf2(i, s) = (0.0_dp, 0.0_dp)
      ENDIF
    ENDIF
  END SUBROUTINE k_coef_h
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_coef_ortho(cf2, cf, wwg, nproj, nkb, nbnd, nslot, npert, nksq, nock)
    ! orthogonalize: ps(j,i) *= wwg(j,i) on the evq rows; vkb rows zeroed
    INTEGER, VALUE :: nproj, nkb, nbnd, nslot, npert, nksq
    COMPLEX(dp) :: cf2(nproj, *), cf(nproj, *)
    REAL(dp) :: wwg(nbnd, nslot, nksq)
    INTEGER :: nock(nksq)
    INTEGER :: i, s, k, n
    s = blockIdx%x
    i = threadIdx%x
    IF (i > nproj) RETURN
    k = (s - 1) / (nslot * npert) + 1
    n = MOD(s - 1, nslot) + 1
    IF (i <= nkb .OR. n > nock(k)) THEN
      cf2(i, s) = (0.0_dp, 0.0_dp)
    ELSE
      cf2(i, s) = wwg(i - nkb, n, k) * cf(i, s)
    ENDIF
  END SUBROUTINE k_coef_ortho
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_scale_wwg(ps, wwg, nbnd, nslot, npert, nksq, nock)
    ! orthogonalize: ps(jbnd, ibnd) = wwg(jbnd) * ps(jbnd, ibnd)
    INTEGER, VALUE :: nbnd, nslot, npert, nksq
    COMPLEX(dp) :: ps(nbnd, *)
    REAL(dp) :: wwg(nbnd, nslot, nksq)
    INTEGER :: nock(nksq)
    INTEGER :: j, s, k, n
    s = blockIdx%x
    j = threadIdx%x
    IF (j > nbnd) RETURN
    k = (s - 1) / (nslot * npert) + 1
    n = MOD(s - 1, nslot) + 1
    IF (n <= nock(k)) ps(j, s) = wwg(j, n, k) * ps(j, s)
  END SUBROUTINE k_scale_wwg
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_scale_wg1(d, wg1, npwx, nslot, npert, nksq, nock, ntot)
    ! orthogonalize: dvpsi(:, i) *= wg1(i); invalid slots zeroed
    INTEGER, VALUE :: npwx, nslot, npert, nksq
    INTEGER(8), VALUE :: ntot
    COMPLEX(dp) :: d(ntot)
    REAL(dp) :: wg1(nslot, nksq)
    INTEGER :: nock(nksq)
    INTEGER(8) :: i
    INTEGER :: s, k, n
    i = INT(blockIdx%x - 1, 8) * blockDim%x + threadIdx%x
    DO WHILE (i <= ntot)
      s = INT((i - 1) / npwx) + 1
      k = (s - 1) / (nslot * npert) + 1
      n = MOD(s - 1, nslot) + 1
      IF (n <= nock(k)) THEN
        d(i) = wg1(n, k) * d(i)
      ELSE
        d(i) = (0.0_dp, 0.0_dp)
      ENDIF
      i = i + INT(blockDim%x, 8) * gridDim%x
    ENDDO
  END SUBROUTINE k_scale_wg1
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_check(list, nslot, npert, npwx, nksq, npwq, hd, g, h, rho, conv, &
                                        cnt, ethr, first, eta, r0, nst)
    ! cgsolve_all, per active column: h = h_diag*g ; rho = <h|g> ; converged if sqrt(rho) < ethr;
    ! otherwise h = -h + (rho/rhoold) hold  (hold is the previous h, kept in place).
    ! cnt(group) counts the columns that were active at this test (kter_eff).
    INTEGER, VALUE :: nslot, npert, npwx, nksq, first
    REAL(dp), VALUE :: ethr, eta
    INTEGER :: list(*), npwq(nksq), conv(*), cnt(*), nst(*)
    REAL(dp) :: hd(npwx, nslot, nksq), rho(*), r0(*)
    COMPLEX(dp) :: g(npwx, *), h(npwx, *)
    REAL(dp), SHARED :: red(RB)
    REAL(dp), SHARED :: gam
    INTEGER, SHARED :: cvd
    INTEGER :: s, k, n, ig, nq, tid, st, istat
    REAL(dp) :: acc, rn
    COMPLEX(dp) :: gv
    tid = threadIdx%x
    s = list(blockIdx%x)
    k = (s - 1) / (nslot * npert) + 1
    n = MOD(s - 1, nslot) + 1
    nq = npwq(k)
    acc = 0.0_dp
    DO ig = tid, nq, blockDim%x
      gv = g(ig, s)
      acc = acc + hd(ig, n, k) * (REAL(gv, dp)**2 + AIMAG(gv)**2)
    ENDDO
    red(tid) = acc
    CALL syncthreads()
    st = blockDim%x / 2
    DO WHILE (st > 0)
      IF (tid <= st) red(tid) = red(tid) + red(tid + st)
      CALL syncthreads()
      st = st / 2
    ENDDO
    IF (tid == 1) THEN
      rn = red(1)
      istat = atomicadd(cnt((s - 1) / nslot + 1), 1)
      IF (first == 1) r0(s) = SQRT(rn)
      IF (first == 1) THEN
        nst(s) = 1
      ELSE
        nst(s) = nst(s) + 1
      ENDIF
      IF (SQRT(rn) < MAX(ethr, eta * r0(s))) THEN
        conv(s) = 1
        cvd = 1
      ELSE
        cvd = 0
        IF (first == 1) THEN
          gam = 0.0_dp
        ELSE
          gam = rn / rho(s)
        ENDIF
        rho(s) = rn
      ENDIF
    ENDIF
    CALL syncthreads()
    IF (cvd == 0) THEN
      IF (first == 1) THEN
        DO ig = tid, nq, blockDim%x
          h(ig, s) = -(g(ig, s) * hd(ig, n, k))
        ENDDO
      ELSE
        DO ig = tid, nq, blockDim%x
          h(ig, s) = -(g(ig, s) * hd(ig, n, k)) + gam * h(ig, s)
        ENDDO
      ENDIF
    ENDIF
  END SUBROUTINE k_check
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_check_defl(list, nslot, npert, npwx, nksq, npwq, hd, g, h, rho, conv, &
                                             cnt, ethr, first, eta, r0, nst, W, nproj, nkb, nbnd, nockq, &
                                             etk, etkq, dfloor, trn)
    ! k_check with the additive spectral preconditioner (RTSC_STH_DEFL): M^-1 = h_diag +
    ! sum_m |evq_m><evq_m| / (e_m(k+q) - e_n(k)) over the known empty bands m = nockq(k)+1 .. nbnd at
    ! k+q, which are exact eigenvectors of the column operator (eigenvalue e_m - e_n, the smallest
    ! ones). The stopping test is unchanged (sqrt(<g|h_diag|g>) < thresh); the CG recurrence uses
    ! rho = <g|M^-1|g> and h = -M^-1 g + gam h_old.
    INTEGER, VALUE :: nslot, npert, npwx, nksq, first, nproj, nkb, nbnd, trn
    ! trn = 1 (RTSC_STH_TRUERES, diagnostic): the stopping test uses the plain residual norm ||g||_2 instead of
    ! QE's sqrt(<g|h_diag|g>); the CG recurrence is unchanged
    REAL(dp), VALUE :: ethr, eta, dfloor
    INTEGER :: list(*), npwq(nksq), conv(*), cnt(*), nst(*), nockq(nksq)
    REAL(dp) :: hd(npwx, nslot, nksq), rho(*), r0(*), etk(nbnd, nksq), etkq(nbnd, nksq)
    COMPLEX(dp) :: g(npwx, *), h(npwx, *), W(npwx, nproj, nksq)
    REAL(dp), SHARED :: red(RB), red2(RB)
    REAL(dp), SHARED :: gam, rn, cre(MAXE), cim(MAXE), dm(MAXE)
    INTEGER, SHARED :: cvd
    INTEGER :: s, k, n, ig, nq, tid, st, istat, m, ne, m0
    REAL(dp) :: acc, acc2, rc, rt
    COMPLEX(dp) :: gv, wv, hv
    tid = threadIdx%x
    s = list(blockIdx%x)
    k = (s - 1) / (nslot * npert) + 1
    n = MOD(s - 1, nslot) + 1
    nq = npwq(k)
    m0 = nockq(k)
    ne = MIN(MAXE, nbnd - m0)
    acc = 0.0_dp
    acc2 = 0.0_dp
    DO ig = tid, nq, blockDim%x
      gv = g(ig, s)
      acc = acc + hd(ig, n, k) * (REAL(gv, dp)**2 + AIMAG(gv)**2)
      acc2 = acc2 + REAL(gv, dp)**2 + AIMAG(gv)**2
    ENDDO
    red(tid) = acc
    red2(tid) = acc2
    CALL syncthreads()
    st = blockDim%x / 2
    DO WHILE (st > 0)
      IF (tid <= st) THEN
        red(tid) = red(tid) + red(tid + st)
        red2(tid) = red2(tid) + red2(tid + st)
      ENDIF
      CALL syncthreads()
      st = st / 2
    ENDDO
    IF (tid == 1) THEN
      rn = red(1)
      rt = rn
      IF (trn == 1) rt = red2(1)
      istat = atomicadd(cnt((s - 1) / nslot + 1), 1)
      IF (first == 1) r0(s) = SQRT(rt)
      IF (first == 1) THEN
        nst(s) = 1
      ELSE
        nst(s) = nst(s) + 1
      ENDIF
      IF (SQRT(rt) < MAX(ethr, eta * r0(s))) THEN
        conv(s) = 1
        cvd = 1
      ELSE
        cvd = 0
      ENDIF
    ENDIF
    CALL syncthreads()
    IF (cvd == 1) RETURN
    ! c_m = <evq_m | g>
    DO m = 1, ne
      acc = 0.0_dp
      acc2 = 0.0_dp
      DO ig = tid, nq, blockDim%x
        wv = W(ig, nkb + m0 + m, k)
        gv = g(ig, s)
        acc = acc + REAL(wv, dp) * REAL(gv, dp) + AIMAG(wv) * AIMAG(gv)
        acc2 = acc2 + REAL(wv, dp) * AIMAG(gv) - AIMAG(wv) * REAL(gv, dp)
      ENDDO
      red(tid) = acc
      red2(tid) = acc2
      CALL syncthreads()
      st = blockDim%x / 2
      DO WHILE (st > 0)
        IF (tid <= st) THEN
          red(tid) = red(tid) + red(tid + st)
          red2(tid) = red2(tid) + red2(tid + st)
        ENDIF
        CALL syncthreads()
        st = st / 2
      ENDDO
      IF (tid == 1) THEN
        cre(m) = red(1)
        cim(m) = red2(1)
        dm(m) = MAX(etkq(m0 + m, k) - etk(n, k), dfloor)
      ENDIF
      CALL syncthreads()
    ENDDO
    IF (tid == 1) THEN
      rc = rn
      DO m = 1, ne
        rc = rc + (cre(m)**2 + cim(m)**2) / dm(m)
      ENDDO
      IF (first == 1) THEN
        gam = 0.0_dp
      ELSE
        gam = rc / rho(s)
      ENDIF
      rho(s) = rc
      DO m = 1, ne
        cre(m) = cre(m) / dm(m)
        cim(m) = cim(m) / dm(m)
      ENDDO
    ENDIF
    CALL syncthreads()
    DO ig = tid, nq, blockDim%x
      hv = g(ig, s) * hd(ig, n, k)
      DO m = 1, ne
        hv = hv + W(ig, nkb + m0 + m, k) * CMPLX(cre(m), cim(m), dp)
      ENDDO
      IF (first == 1) THEN
        h(ig, s) = -hv
      ELSE
        h(ig, s) = -hv + gam * h(ig, s)
      ENDIF
    ENDDO
  END SUBROUTINE k_check_defl
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_check_lift(list, nslot, npert, npwx, nksq, npwq, hd, g, h, rho, conv, &
                                             cnt, ethr, first, eta, r0, nst, z, lt, etk, nbnd, v0, lok)
    ! k_check with the LIFT-F preconditioner: z = Q W_n Q^H g (FP32, lift_apply), M^-1 g = z + D_n g.
    ! Stopping test unchanged (sqrt(<g|h_diag|g>) < thresh); rho = <g|M^-1 g>; h = -M^-1 g + gam h_old.
    INTEGER, VALUE :: nslot, npert, npwx, nksq, first, nbnd
    REAL(dp), VALUE :: ethr, eta, v0
    INTEGER :: list(*), npwq(nksq), conv(*), cnt(*), nst(*)
    REAL(dp) :: hd(npwx, nslot, nksq), rho(*), r0(*), lt(npwx, nksq), etk(nbnd, nksq)
    INTEGER :: lok(nslot, nksq)
    COMPLEX(dp) :: g(npwx, *), h(npwx, *)
    COMPLEX(sp) :: z(npwx, *)
    REAL(dp), SHARED :: red(RB), red2(RB)
    REAL(dp), SHARED :: gam
    INTEGER, SHARED :: cvd
    INTEGER :: s, k, n, ig, nq, tid, st, istat, ok
    REAL(dp) :: acc, acc2, rn, en, t, dn
    COMPLEX(dp) :: gv, mv
    tid = threadIdx%x
    s = list(blockIdx%x)
    k = (s - 1) / (nslot * npert) + 1
    n = MOD(s - 1, nslot) + 1
    nq = npwq(k)
    en = etk(n, k) - v0
    ok = lok(n, k)
    acc = 0.0_dp
    acc2 = 0.0_dp
    DO ig = tid, nq, blockDim%x
      gv = g(ig, s)
      acc = acc + hd(ig, n, k) * (REAL(gv, dp)**2 + AIMAG(gv)**2)
      IF (ok == 1) THEN
        t = lt(ig, k)
        dn = 0.0_dp
        IF (t > 0.0_dp) dn = 1.0_dp / MAX(t - en, 0.5_dp * t)
        mv = CMPLX(z(ig, s), KIND=dp) + dn * gv
      ELSE
        mv = gv * hd(ig, n, k)
      ENDIF
      acc2 = acc2 + REAL(gv, dp) * REAL(mv, dp) + AIMAG(gv) * AIMAG(mv)
    ENDDO
    red(tid) = acc
    red2(tid) = acc2
    CALL syncthreads()
    st = blockDim%x / 2
    DO WHILE (st > 0)
      IF (tid <= st) THEN
        red(tid) = red(tid) + red(tid + st)
        red2(tid) = red2(tid) + red2(tid + st)
      ENDIF
      CALL syncthreads()
      st = st / 2
    ENDDO
    IF (tid == 1) THEN
      rn = red(1)
      istat = atomicadd(cnt((s - 1) / nslot + 1), 1)
      IF (first == 1) r0(s) = SQRT(rn)
      IF (first == 1) THEN
        nst(s) = 1
      ELSE
        nst(s) = nst(s) + 1
      ENDIF
      IF (SQRT(rn) < MAX(ethr, eta * r0(s))) THEN
        conv(s) = 1
        cvd = 1
      ELSE
        cvd = 0
        IF (first == 1) THEN
          gam = 0.0_dp
        ELSE
          gam = red2(1) / rho(s)
        ENDIF
        rho(s) = red2(1)
      ENDIF
    ENDIF
    CALL syncthreads()
    IF (cvd == 1) RETURN
    DO ig = tid, nq, blockDim%x
      gv = g(ig, s)
      IF (ok == 1) THEN
        t = lt(ig, k)
        dn = 0.0_dp
        IF (t > 0.0_dp) dn = 1.0_dp / MAX(t - en, 0.5_dp * t)
        mv = CMPLX(z(ig, s), KIND=dp) + dn * gv
      ELSE
        mv = gv * hd(ig, n, k)
      ENDIF
      IF (first == 1) THEN
        h(ig, s) = -mv
      ELSE
        h(ig, s) = -mv + gam * h(ig, s)
      ENDIF
    ENDDO
  END SUBROUTINE k_check_lift
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_lift_scale(c, lam, etk, nm, nbnd, nslot, ncolk, k0, ntot, v0, delta)
    ! c(j, sc) /= max(Lambda_j(k) - (e_n(k) - V0), delta) on the chunk's columns sc = k0*ncolk + 1 .. (k0+nk)*ncolk
    INTEGER, VALUE :: nm, nbnd, nslot, ncolk, k0
    INTEGER(8), VALUE :: ntot
    REAL(dp), VALUE :: v0, delta
    COMPLEX(sp) :: c(nm, *)
    REAL(dp) :: lam(nm, *), etk(nbnd, *)
    INTEGER(8) :: i
    INTEGER :: j, sc, k, n
    i = INT(blockIdx%x - 1, 8) * blockDim%x + threadIdx%x
    DO WHILE (i <= ntot)
      j = INT(MOD(i - 1, INT(nm, 8))) + 1
      sc = INT((i - 1) / nm) + 1 + k0 * ncolk
      k = (sc - 1) / ncolk + 1
      n = MOD(sc - 1, nslot) + 1
      c(j, sc) = c(j, sc) * REAL(1.0_dp / MAX(lam(j, k) - (etk(n, k) - v0), delta), sp)
      i = i + INT(blockDim%x, 8) * gridDim%x
    ENDDO
  END SUBROUTINE k_lift_scale
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_kd_rot(w, evc, rwf, rep, tr, ft, map, ph, npwq, npw, nock, npwx, nproj, nkb, nbnd, &
                                       nslot, ldr, nbr, nksq)
    ! KDEV: SYMROT's rotation on the device. which = 1: evc of k (bands <= nock) into d_evc; which = 2: evq of k+q (all
    ! bands) into W. dst(map(ig)) = src(ig) [* ph(ig)], conjugated for time reversal: exactly rotate's arithmetic.
    INTEGER, VALUE :: npwx, nproj, nkb, nbnd, nslot, ldr, nbr, nksq
    COMPLEX(dp) :: w(npwx, nproj, *), evc(npwx, *), rwf(ldr, nbr, *), ph(npwx, 2, *)
    INTEGER :: rep(2, *), tr(2, *), ft(2, *), map(npwx, 2, *), npwq(*), npw(*), nock(*)
    INTEGER :: ig, b, k, wh, r
    COMPLEX(dp) :: v
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    b = blockIdx%y
    k = blockIdx%z
    DO WHILE (k <= nksq)
      DO wh = 1, 2
        r = rep(wh, k)
        IF (r < 1) CYCLE
        IF (wh == 1) THEN
          IF (b > nock(k) .OR. ig > npw(k)) CYCLE
        ELSE
          IF (ig > npwq(k)) CYCLE
        ENDIF
        v = rwf(ig, b, r)
        IF (ft(wh, k) == 1) v = v * ph(ig, wh, k)
        IF (tr(wh, k) == -1) v = CONJG(v)
        IF (wh == 1) THEN
          evc(map(ig, 1, k), nslot * (k - 1) + b) = v
        ELSE
          w(map(ig, 2, k), nkb + b, k) = v
        ENDIF
      ENDDO
      k = k + gridDim%z
    ENDDO
  END SUBROUTINE k_kd_rot
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_kd_inv(inv, idxk, idxkq, npw, npwq, wh, k0, nk, nnr, npwx)
    ! KDEV v2: inv(idx(ig, k0 + j), j) = ig on the target sphere (k for wh = 1, k+q for wh = 2); the table is zeroed by
    ! k_kd_rot2 after use (every entry it set), so it is all zeros before the first call
    INTEGER, VALUE :: wh, k0, nk, nnr, npwx
    INTEGER :: inv(nnr, *), idxk(npwx, *), idxkq(npwx, *), npw(*), npwq(*)
    INTEGER :: ig, j, k
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    j = blockIdx%y
    DO WHILE (j <= nk)
      k = k0 + j
      IF (wh == 1) THEN
        IF (ig <= npw(k)) inv(idxk(ig, k), j) = ig
      ELSE
        IF (ig <= npwq(k)) inv(idxkq(ig, k), j) = ig
      ENDIF
      j = j + gridDim%y
    ENDDO
  END SUBROUTINE k_kd_inv
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_kd_rot2(w, evc, rwf, rmill, rep, tr, op, inv, npwq, npw, nock, wh, k0, nk, npwx, nproj, &
                                        nkb, nbnd, nslot, ldr, nbr, nnr, nr1, nr2, nr3)
    ! KDEV v2: dst(inv(box(tr S m - G0))) = src (conjugated for tr = -1) for every G of the cache entry: rotate's copy, with
    ! the map from the device (no fractional translations here; those k take the host path)
    INTEGER, VALUE :: wh, k0, nk, npwx, nproj, nkb, nbnd, nslot, ldr, nbr, nnr, nr1, nr2, nr3
    COMPLEX(dp) :: w(npwx, nproj, *), evc(npwx, *), rwf(ldr, nbr, *)
    INTEGER :: rmill(3, ldr, *), rep(2, *), tr(2, *), op(12, 2, *), inv(nnr, *), npwq(*), npw(*), nock(*)
    INTEGER :: ig, b, j, k, r, t, m1, m2, m3, a1, a2, a3, p, n
    COMPLEX(dp) :: v
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    b = blockIdx%y
    j = blockIdx%z
    DO WHILE (j <= nk)
      k = k0 + j
      r = rep(wh, k)
      IF (r > 0) THEN
        IF (wh == 1) THEN
          n = npw(k)
        ELSE
          n = npwq(k)
        ENDIF
        IF (ig <= n .AND. (wh == 2 .OR. b <= nock(k))) THEN
          t = tr(wh, k)
          m1 = rmill(1, ig, r); m2 = rmill(2, ig, r); m3 = rmill(3, ig, r)
          a1 = t * (op(1, wh, k) * m1 + op(4, wh, k) * m2 + op(7, wh, k) * m3) - op(10, wh, k)
          a2 = t * (op(2, wh, k) * m1 + op(5, wh, k) * m2 + op(8, wh, k) * m3) - op(11, wh, k)
          a3 = t * (op(3, wh, k) * m1 + op(6, wh, k) * m2 + op(9, wh, k) * m3) - op(12, wh, k)
          p = inv(1 + MODULO(a1, nr1) + nr1 * (MODULO(a2, nr2) + nr2 * MODULO(a3, nr3)), j)
          IF (p < 1) p = npwx       ! cannot happen for a SYMROT map (checked on the host); keeps the write in bounds
          v = rwf(ig, b, r)
          IF (t == -1) v = CONJG(v)
          IF (wh == 1) THEN
            evc(p, nslot * (k - 1) + b) = v
          ELSE
            w(p, nkb + b, k) = v
          ENDIF
        ENDIF
      ENDIF
      j = j + gridDim%z
    ENDDO
  END SUBROUTINE k_kd_rot2
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_kd_clr(inv, idxk, idxkq, npw, npwq, wh, k0, nk, nnr, npwx)
    ! KDEV v2: zero the entries k_kd_inv set (the table is reused by the next chunk)
    INTEGER, VALUE :: wh, k0, nk, nnr, npwx
    INTEGER :: inv(nnr, *), idxk(npwx, *), idxkq(npwx, *), npw(*), npwq(*)
    INTEGER :: ig, j, k
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    j = blockIdx%y
    DO WHILE (j <= nk)
      k = k0 + j
      IF (wh == 1) THEN
        IF (ig <= npw(k)) inv(idxk(ig, k), j) = 0
      ELSE
        IF (ig <= npwq(k)) inv(idxkq(ig, k), j) = 0
      ENDIF
      j = j + gridDim%y
    ENDDO
  END SUBROUTINE k_kd_clr
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_cb_mapdev(map, rmill, rep, tr, op, dv, inv, npwq, k0, nk, npwx, ldr, nnr, nr1, nr2, nr3)
    ! CBSR: map(ig, k) = position in k+q's sphere of tr S m_ig - G0 (m: Miller indices of the cache entry), for dv(k) = 1
    INTEGER, VALUE :: k0, nk, npwx, ldr, nnr, nr1, nr2, nr3
    INTEGER :: map(npwx, *), rmill(3, ldr, *), rep(*), tr(*), op(12, *), dv(*), inv(nnr, *), npwq(*)
    INTEGER :: ig, j, k, r, t, m1, m2, m3, a1, a2, a3
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    j = blockIdx%y
    DO WHILE (j <= nk)
      k = k0 + j
      IF (dv(k) == 1 .AND. ig <= npwq(k)) THEN
        r = rep(k); t = tr(k)
        m1 = rmill(1, ig, r); m2 = rmill(2, ig, r); m3 = rmill(3, ig, r)
        a1 = t * (op(1, k) * m1 + op(4, k) * m2 + op(7, k) * m3) - op(10, k)
        a2 = t * (op(2, k) * m1 + op(5, k) * m2 + op(8, k) * m3) - op(11, k)
        a3 = t * (op(3, k) * m1 + op(6, k) * m2 + op(9, k) * m3) - op(12, k)
        map(ig, k) = MAX(1, inv(1 + MODULO(a1, nr1) + nr1 * (MODULO(a2, nr2) + nr2 * MODULO(a3, nr3)), j))
      ENDIF
      j = j + gridDim%y
    ENDDO
  END SUBROUTINE k_cb_mapdev
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_cb_scal(a, f, n)
    INTEGER(8), VALUE :: n
    REAL(dp), VALUE :: f
    COMPLEX(dp) :: a(*)
    INTEGER(8) :: i
    i = INT(blockIdx%x - 1, 8) * blockDim%x + threadIdx%x
    DO WHILE (i <= n)
      a(i) = a(i) * f
      i = i + INT(blockDim%x, 8) * gridDim%x
    ENDDO
  END SUBROUTINE k_cb_scal
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_cb_asm(a, vg, idx, g2, nq, n0, nr1, nr2, nr3, v0)
    ! CBSR assembly: a(ig, j) = V(G_ig - G_j) + delta_ij (|k+q+G_j|^2 - V0) on the nq x nq block, zero elsewhere (n0 x n0)
    INTEGER, VALUE :: nq, n0, nr1, nr2, nr3
    REAL(dp), VALUE :: v0
    COMPLEX(dp) :: a(n0, *), vg(*)
    INTEGER :: idx(*)
    REAL(dp) :: g2(*)
    INTEGER :: ig, j, i1, j1, a1, a2, a3
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    j = blockIdx%y
    DO WHILE (j <= n0)
      IF (ig <= n0) THEN
        IF (ig <= nq .AND. j <= nq) THEN
          i1 = idx(ig) - 1; j1 = idx(j) - 1
          a1 = MODULO(MOD(i1, nr1) - MOD(j1, nr1), nr1)
          a2 = MODULO(MOD(i1 / nr1, nr2) - MOD(j1 / nr1, nr2), nr2)
          a3 = MODULO(i1 / (nr1 * nr2) - j1 / (nr1 * nr2), nr3)
          a(ig, j) = vg(1 + a1 + nr1 * (a2 + nr2 * a3))
          IF (ig == j) a(ig, j) = a(ig, j) + (g2(j) - v0)
        ELSE
          a(ig, j) = (0.0_dp, 0.0_dp)
        ENDIF
      ENDIF
      j = j + gridDim%y
    ENDDO
  END SUBROUTINE k_cb_asm
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_cb_herm(a, nq, n0)
    ! upper triangle (read by zheevd 'U') = the Hermitian part: a(i, j) = (a(i, j) + conj(a(j, i))) / 2, i <= j
    INTEGER, VALUE :: nq, n0
    COMPLEX(dp) :: a(n0, *)
    INTEGER :: i, j
    i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    j = blockIdx%y
    DO WHILE (j <= nq)
      IF (i < j) THEN
        a(i, j) = 0.5_dp * (a(i, j) + CONJG(a(j, i)))
      ELSE IF (i == j) THEN
        a(i, i) = CMPLX(REAL(a(i, i)), 0.0_dp, KIND=dp)
      ENDIF
      j = j + gridDim%y
    ENDDO
  END SUBROUTINE k_cb_herm
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_cb_torep(u, a, map, ph, tr, nq, n0)
    ! eigenvectors of the member k+q (a) into the basis' own G order: u(ig, j) = conj(ph) a(map(ig), j) (tr = 1),
    ! conj(ph a(map(ig), j)) (tr = -1); zero outside the nq x nq block
    INTEGER, VALUE :: tr, nq, n0
    COMPLEX(dp) :: u(n0, *), a(n0, *), ph(*)
    INTEGER :: map(*)
    INTEGER :: ig, j
    COMPLEX(dp) :: v
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    j = blockIdx%y
    DO WHILE (j <= n0)
      IF (ig <= n0) THEN
        IF (ig <= nq .AND. j <= nq) THEN
          v = a(map(ig), j)
          IF (tr == 1) THEN
            u(ig, j) = CONJG(ph(ig)) * v
          ELSE
            u(ig, j) = CONJG(ph(ig) * v)
          ENDIF
        ELSE
          u(ig, j) = (0.0_dp, 0.0_dp)
        ENDIF
      ENDIF
      j = j + gridDim%y
    ENDDO
  END SUBROUTINE k_cb_torep
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_cb_gather(x, d0, gsc, map, ph, tr, npwq, n0, npwx, ncolk, ctot)
    ! x(:, c) = the right-hand side of engine slot gsc(c) in its basis' G order (0 for padding columns)
    INTEGER, VALUE :: n0, npwx, ncolk, ctot
    COMPLEX(dp) :: x(n0, *), d0(npwx, *), ph(n0, *)
    INTEGER :: gsc(*), map(n0, *), tr(*), npwq(*)
    INTEGER :: j, c, sc, k
    COMPLEX(dp) :: b
    j = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    c = blockIdx%y
    DO WHILE (c <= ctot)
      IF (j <= n0) THEN
        sc = gsc(c)
        IF (sc > 0) THEN
          k = (sc - 1) / ncolk + 1
          IF (j <= npwq(k)) THEN
            b = d0(map(j, k), sc)
            IF (tr(k) == 1) THEN
              x(j, c) = CONJG(ph(j, k)) * b
            ELSE
              x(j, c) = CONJG(ph(j, k) * b)
            ENDIF
          ELSE
            x(j, c) = (0.0_dp, 0.0_dp)
          ENDIF
        ELSE
          x(j, c) = (0.0_dp, 0.0_dp)
        ENDIF
      ENDIF
      c = c + gridDim%y
    ENDDO
  END SUBROUTINE k_cb_gather
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_cb_scale(y, lam, gl, gsc, etk, v0, n0, nbnd, nslot, ncolk, ctot)
    ! y(j, c) /= (Lambda_j - (e_n(k) - V0)), Lambda of the column's basis (padding rows: Lambda = 1e30 -> 0)
    INTEGER, VALUE :: n0, nbnd, nslot, ncolk, ctot
    REAL(dp), VALUE :: v0
    COMPLEX(dp) :: y(n0, *)
    REAL(dp) :: lam(n0, *), etk(nbnd, *)
    INTEGER :: gl(*), gsc(*)
    INTEGER :: j, c, sc, k, n
    REAL(dp) :: d
    j = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    c = blockIdx%y
    DO WHILE (c <= ctot)
      IF (j <= n0) THEN
        sc = gsc(c)
        IF (sc > 0) THEN
          k = (sc - 1) / ncolk + 1
          n = MOD(sc - 1, nslot) + 1
          d = lam(j, gl(c)) - (etk(n, k) - v0)
          IF (ABS(d) < 1.0e-12_dp) d = SIGN(1.0e-12_dp, d)
          y(j, c) = y(j, c) / d
        ENDIF
      ENDIF
      c = c + gridDim%y
    ENDDO
  END SUBROUTINE k_cb_scale
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_cb_scatter(dpsi, x, gsc, map, ph, tr, npwq, n0, npwx, ncolk, ctot)
    ! dpsi(map(j, k), sc) = ph x(j, c) (tr = 1), conj(ph x(j, c)) (tr = -1); rows > npwq(k) are left as they are (zero)
    INTEGER, VALUE :: n0, npwx, ncolk, ctot
    COMPLEX(dp) :: dpsi(npwx, *), x(n0, *), ph(n0, *)
    INTEGER :: gsc(*), map(n0, *), tr(*), npwq(*)
    INTEGER :: j, c, sc, k
    j = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    c = blockIdx%y
    DO WHILE (c <= ctot)
      IF (j <= n0) THEN
        sc = gsc(c)
        IF (sc > 0) THEN
          k = (sc - 1) / ncolk + 1
          IF (j <= npwq(k)) THEN
            IF (tr(k) == 1) THEN
              dpsi(map(j, k), sc) = ph(j, k) * x(j, c)
            ELSE
              dpsi(map(j, k), sc) = CONJG(ph(j, k) * x(j, c))
            ENDIF
          ENDIF
        ENDIF
      ENDIF
      c = c + gridDim%y
    ENDDO
  END SUBROUTINE k_cb_scatter
  !
  ! ---- rtsc b2r: CBSR refinement form (RTSC_STH_CBSR_RF) ----
  ATTRIBUTES(GLOBAL) SUBROUTINE k_cb_split(u0, u1, a, n0, m0)
    ! u0 = a(:, 1:m0) (FP64), u1 = a(:, m0+1:n0) rounded to FP32
    INTEGER, VALUE :: n0, m0
    COMPLEX(dp) :: u0(n0, *), a(n0, *)
    COMPLEX(sp) :: u1(n0, *)
    INTEGER :: ig, j
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    j = blockIdx%y
    DO WHILE (j <= n0)
      IF (ig <= n0) THEN
        IF (j <= m0) THEN
          u0(ig, j) = a(ig, j)
        ELSE
          u1(ig, j - m0) = CMPLX(REAL(a(ig, j), dp), AIMAG(a(ig, j)), KIND=sp)
        ENDIF
      ENDIF
      j = j + gridDim%y
    ENDDO
  END SUBROUTINE k_cb_split
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_cb_z2c(y, x, n)
    INTEGER(8), VALUE :: n
    COMPLEX(sp) :: y(*)
    COMPLEX(dp) :: x(*)
    INTEGER(8) :: i
    i = INT(blockIdx%x - 1, 8) * blockDim%x + threadIdx%x
    DO WHILE (i <= n)
      y(i) = CMPLX(REAL(x(i), dp), AIMAG(x(i)), KIND=sp)
      i = i + INT(blockDim%x, 8) * gridDim%x
    ENDDO
  END SUBROUTINE k_cb_z2c
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_cb_addc(x, y, n)
    ! x += y (FP32 into FP64)
    INTEGER(8), VALUE :: n
    COMPLEX(dp) :: x(*)
    COMPLEX(sp) :: y(*)
    INTEGER(8) :: i
    i = INT(blockIdx%x - 1, 8) * blockDim%x + threadIdx%x
    DO WHILE (i <= n)
      x(i) = x(i) + CMPLX(REAL(y(i), dp), REAL(AIMAG(y(i)), dp), KIND=dp)
      i = i + INT(blockDim%x, 8) * gridDim%x
    ENDDO
  END SUBROUTINE k_cb_addc
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_cb_scale_rf(y0, y1, lam, gl, gsc, etk, v0, n0, m0, nbnd, nslot, ncolk, nc)
    ! tier 0: y0(j, c) /= (Lambda_j - e~_n), j <= m0 (FP64); tier 1: y1(j - m0, c) /= (Lambda_j - e~_n) (FP32)
    INTEGER, VALUE :: n0, m0, nbnd, nslot, ncolk, nc
    REAL(dp), VALUE :: v0
    COMPLEX(dp) :: y0(m0, *)
    COMPLEX(sp) :: y1(n0 - m0, *)
    REAL(dp) :: lam(n0, *), etk(nbnd, *)
    INTEGER :: gl(*), gsc(*)
    INTEGER :: j, c, sc, k, n
    REAL(dp) :: d
    j = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    c = blockIdx%y
    DO WHILE (c <= nc)
      IF (j <= n0) THEN
        sc = gsc(c)
        IF (sc > 0) THEN
          k = (sc - 1) / ncolk + 1
          n = MOD(sc - 1, nslot) + 1
          d = lam(j, gl(c)) - (etk(n, k) - v0)
          IF (ABS(d) < 1.0e-12_dp) d = SIGN(1.0e-12_dp, d)
          IF (j <= m0) THEN
            y0(j, c) = y0(j, c) / d
          ELSE
            y1(j - m0, c) = y1(j - m0, c) * REAL(1.0_dp / d, sp)
          ENDIF
        ELSE
          IF (j <= m0) THEN
            y0(j, c) = (0.0_dp, 0.0_dp)
          ELSE
            y1(j - m0, c) = (0.0_sp, 0.0_sp)
          ENDIF
        ENDIF
      ENDIF
      c = c + gridDim%y
    ENDDO
  END SUBROUTINE k_cb_scale_rf
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_cb_scatter_sub(dpsi, x, gsc, map, ph, tr, npwq, n0, npwx, ncolk, ctot)
    ! dpsi(map(j, k), sc) -= ph x(j, c) (tr = 1), conj(ph x(j, c)) (tr = -1)
    INTEGER, VALUE :: n0, npwx, ncolk, ctot
    COMPLEX(dp) :: dpsi(npwx, *), x(n0, *), ph(n0, *)
    INTEGER :: gsc(*), map(n0, *), tr(*), npwq(*)
    INTEGER :: j, c, sc, k
    j = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    c = blockIdx%y
    DO WHILE (c <= ctot)
      IF (j <= n0) THEN
        sc = gsc(c)
        IF (sc > 0) THEN
          k = (sc - 1) / ncolk + 1
          IF (j <= npwq(k)) THEN
            IF (tr(k) == 1) THEN
              dpsi(map(j, k), sc) = dpsi(map(j, k), sc) - ph(j, k) * x(j, c)
            ELSE
              dpsi(map(j, k), sc) = dpsi(map(j, k), sc) - CONJG(ph(j, k) * x(j, c))
            ENDIF
          ENDIF
        ENDIF
      ENDIF
      c = c + gridDim%y
    ENDDO
  END SUBROUTINE k_cb_scatter_sub
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_zero_conv(g, list, conv, npwx, n)
    ! g(:, s) = 0 for the slots s = list(1:n) with conv(s) /= 0 (their correction is then zero)
    INTEGER, VALUE :: npwx, n
    INTEGER :: list(*), conv(*)
    COMPLEX(dp) :: g(npwx, *)
    INTEGER :: s, ig
    IF (blockIdx%x > n) RETURN
    s = list(blockIdx%x)
    IF (conv(s) == 0) RETURN
    DO ig = threadIdx%x, npwx, blockDim%x
      g(ig, s) = (0.0_dp, 0.0_dp)
    ENDDO
  END SUBROUTINE k_zero_conv
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_sub_list(g, h, d, list, npwx)
    ! g(:, s) = h(:, s) - d(:, s) for s in list
    INTEGER, VALUE :: npwx
    INTEGER :: list(*)
    COMPLEX(dp) :: g(npwx, *), h(npwx, *), d(npwx, *)
    INTEGER :: s, ig
    s = list(blockIdx%x)
    DO ig = threadIdx%x, npwx, blockDim%x
      g(ig, s) = h(ig, s) - d(ig, s)
    ENDDO
  END SUBROUTINE k_sub_list
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_check_rf(list, nslot, npert, npwx, nksq, npwq, hd, g, rho, conv, cnt, ethr, first, r0, nst)
    ! QE's test (cgsolve_all) on the true residual g = A dpsi - b: converged if sqrt(<h_diag g, g>) < ethr; rho = <h_diag g, g>
    INTEGER, VALUE :: nslot, npert, npwx, nksq, first
    REAL(dp), VALUE :: ethr
    INTEGER :: list(*), npwq(nksq), conv(*), cnt(*), nst(*)
    REAL(dp) :: hd(npwx, nslot, nksq), rho(*), r0(*)
    COMPLEX(dp) :: g(npwx, *)
    REAL(dp), SHARED :: red(RB)
    INTEGER :: s, k, n, ig, nq, tid, st, istat
    REAL(dp) :: acc, rn
    COMPLEX(dp) :: gv
    tid = threadIdx%x
    s = list(blockIdx%x)
    k = (s - 1) / (nslot * npert) + 1
    n = MOD(s - 1, nslot) + 1
    nq = npwq(k)
    acc = 0.0_dp
    DO ig = tid, nq, blockDim%x
      gv = g(ig, s)
      acc = acc + hd(ig, n, k) * (REAL(gv, dp)**2 + AIMAG(gv)**2)
    ENDDO
    red(tid) = acc
    CALL syncthreads()
    st = blockDim%x / 2
    DO WHILE (st > 0)
      IF (tid <= st) red(tid) = red(tid) + red(tid + st)
      CALL syncthreads()
      st = st / 2
    ENDDO
    IF (tid == 1) THEN
      rn = red(1)
      istat = atomicadd(cnt((s - 1) / nslot + 1), 1)
      IF (first == 1) THEN
        r0(s) = SQRT(rn)
        nst(s) = 1
      ELSE
        nst(s) = nst(s) + 1
      ENDIF
      rho(s) = rn
      IF (SQRT(rn) < ethr) conv(s) = 1
    ENDIF
  END SUBROUTINE k_check_rf
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_r2c(a, b, n)
    ! a (complex) = b (real)
    INTEGER(8), VALUE :: n
    COMPLEX(dp) :: a(n)
    REAL(dp) :: b(n)
    INTEGER(8) :: i
    i = INT(blockIdx%x - 1, 8) * blockDim%x + threadIdx%x
    DO WHILE (i <= n)
      a(i) = CMPLX(b(i), 0.0_dp, KIND=dp)
      i = i + INT(blockDim%x, 8) * gridDim%x
    ENDDO
  END SUBROUTINE k_r2c
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_kter(ngrp, npert, nksq, nock, cnt, kter)
    ! kter_eff = kter_eff + DBLE(lbnd) / DBLE(nbnd_occ(ikk)), once per CG step of each solve
    INTEGER, VALUE :: ngrp, npert, nksq
    INTEGER :: nock(nksq), cnt(ngrp)
    REAL(dp) :: kter(ngrp)
    INTEGER :: gi, k
    gi = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (gi > ngrp) RETURN
    IF (cnt(gi) > 0) THEN
      k = (gi - 1) / npert + 1
      kter(gi) = kter(gi) + DBLE(cnt(gi)) / DBLE(nock(k))
      cnt(gi) = 0
    ENDIF
  END SUBROUTINE k_kter
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_compact(act, act2, nact, conv, nout)
    ! single block: keep the active columns with conv == 0, order preserved
    INTEGER, VALUE :: nact
    INTEGER :: act(*), act2(*), conv(*), nout(1)
    INTEGER, SHARED :: sc(1024)
    INTEGER, SHARED :: base
    INTEGER :: tid, bs, c0, i, f, s, off, v
    tid = threadIdx%x
    bs = blockDim%x
    IF (tid == 1) base = 0
    CALL syncthreads()
    DO c0 = 0, nact - 1, bs
      i = c0 + tid
      f = 0
      s = 0
      IF (i <= nact) THEN
        s = act(i)
        IF (conv(s) == 0) f = 1
      ENDIF
      sc(tid) = f
      CALL syncthreads()
      off = 1
      DO WHILE (off < bs)
        v = 0
        IF (tid > off) v = sc(tid - off)
        CALL syncthreads()
        sc(tid) = sc(tid) + v
        CALL syncthreads()
        off = off * 2
      ENDDO
      IF (f == 1) act2(base + sc(tid)) = s
      CALL syncthreads()
      IF (tid == 1) base = base + sc(bs)
      CALL syncthreads()
    ENDDO
    DO i = tid, base, bs
      act(i) = act2(i)
    ENDDO
    IF (tid == 1) nout(1) = base
  END SUBROUTINE k_compact
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_update(list, nslot, npert, npwx, nksq, npwq, h, g, t, dpsi, av, cv)
    ! cgsolve_all: a = <h|g>, c = <h|t>, lambda = -a/c ; dpsi += lambda h ; g += lambda t
    INTEGER, VALUE :: nslot, npert, npwx, nksq
    INTEGER :: list(*), npwq(nksq)
    COMPLEX(dp) :: h(npwx, *), g(npwx, *), t(npwx, *), dpsi(npwx, *)
    REAL(dp) :: av(*), cv(*)
    REAL(dp), SHARED :: ra(RB), rc(RB)
    REAL(dp), SHARED :: lam
    INTEGER :: s, k, ig, nq, tid, st
    REAL(dp) :: a, c
    COMPLEX(dp) :: hv, gv, tv
    tid = threadIdx%x
    s = list(blockIdx%x)
    k = (s - 1) / (nslot * npert) + 1
    nq = npwq(k)
    a = 0.0_dp
    c = 0.0_dp
    DO ig = tid, nq, blockDim%x
      hv = h(ig, s)
      gv = g(ig, s)
      tv = t(ig, s)
      a = a + REAL(hv, dp) * REAL(gv, dp) + AIMAG(hv) * AIMAG(gv)
      c = c + REAL(hv, dp) * REAL(tv, dp) + AIMAG(hv) * AIMAG(tv)
    ENDDO
    ra(tid) = a
    rc(tid) = c
    CALL syncthreads()
    st = blockDim%x / 2
    DO WHILE (st > 0)
      IF (tid <= st) THEN
        ra(tid) = ra(tid) + ra(tid + st)
        rc(tid) = rc(tid) + rc(tid + st)
      ENDIF
      CALL syncthreads()
      st = st / 2
    ENDDO
    IF (tid == 1) THEN
      lam = -ra(1) / rc(1)
      av(s) = ra(1)
      cv(s) = rc(1)
    ENDIF
    CALL syncthreads()
    DO ig = tid, nq, blockDim%x
      dpsi(ig, s) = dpsi(ig, s) + lam * h(ig, s)
      g(ig, s) = g(ig, s) + lam * t(ig, s)
    ENDDO
  END SUBROUTINE k_update
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_hprec(hd, W, g2, npwq, nock, npwx, nproj, nkb, nslot, nksq)
    ! h_prec: eprec = 1.35 <evq_n|g2kin|evq_n> ; h_diag = 1/max(1, g2kin/eprec), rows <= npwq
    INTEGER, VALUE :: npwx, nproj, nkb, nslot, nksq
    REAL(dp) :: hd(npwx, nslot, nksq), g2(npwx, nksq)
    COMPLEX(dp) :: W(npwx, nproj, nksq)
    INTEGER :: npwq(nksq), nock(nksq)
    REAL(dp), SHARED :: red(RB)
    INTEGER :: b, k, n, ig, nq, tid, st
    REAL(dp) :: acc, eprec
    COMPLEX(dp) :: ev
    tid = threadIdx%x
    b = blockIdx%x
    k = (b - 1) / nslot + 1
    n = MOD(b - 1, nslot) + 1
    nq = npwq(k)
    IF (n > nock(k)) THEN
      DO ig = tid, npwx, blockDim%x
        hd(ig, n, k) = 0.0_dp
      ENDDO
      RETURN
    ENDIF
    acc = 0.0_dp
    DO ig = tid, nq, blockDim%x
      ev = W(ig, nkb + n, k)
      acc = acc + g2(ig, k) * (REAL(ev, dp)**2 + AIMAG(ev)**2)
    ENDDO
    red(tid) = acc
    CALL syncthreads()
    st = blockDim%x / 2
    DO WHILE (st > 0)
      IF (tid <= st) red(tid) = red(tid) + red(tid + st)
      CALL syncthreads()
      st = st / 2
    ENDDO
    eprec = 1.35_dp * red(1)
    DO ig = tid, npwx, blockDim%x
      IF (ig <= nq) THEN
        hd(ig, n, k) = 1.0_dp / MAX(1.0_dp, g2(ig, k) / eprec)
      ELSE
        hd(ig, n, k) = 0.0_dp
      ENDIF
    ENDDO
  END SUBROUTINE k_hprec
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_scatter_evc(psir, evc, nnr, npwx, nslot, nksq, npw, nock, idxk)
    ! psir(:, b) (pre-zeroed) <- evc(:, b) at the k FFT indices, b = n + nslot*(k-1)
    INTEGER, VALUE :: nnr, npwx, nslot, nksq
    COMPLEX(dp) :: psir(nnr, *), evc(npwx, *)
    INTEGER :: npw(nksq), nock(nksq), idxk(npwx, nksq)
    INTEGER :: b, k, n, ig
    b = blockIdx%y
    k = (b - 1) / nslot + 1
    n = MOD(b - 1, nslot) + 1
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (n <= nock(k) .AND. ig <= npw(k)) psir(idxk(ig, k), b) = evc(ig, b)
  END SUBROUTINE k_scatter_evc
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_psir_chunk(box, evc, list, s0, nnr, npwx, nslot, npert, nksq, npw, idxk)
    ! no-cache psi(r): box(:, j) (pre-zeroed) <- evc of column list(s0+j)'s band at the k FFT indices
    INTEGER, VALUE :: s0, nnr, npwx, nslot, npert, nksq
    COMPLEX(dp) :: box(nnr, *), evc(npwx, *)
    INTEGER :: list(*), npw(nksq), idxk(npwx, nksq)
    INTEGER :: j, s, k, n, ig
    j = blockIdx%y
    s = list(s0 + j)
    k = (s - 1) / (nslot * npert) + 1
    n = MOD(s - 1, nslot) + 1
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ig <= npw(k)) box(idxk(ig, k), j) = evc(ig, n + nslot * (k - 1))
  END SUBROUTINE k_psir_chunk
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_dvpsir(box, psir, dv, list, s0, nnr, nslot, npert, pj)
    ! apply_dpot: box(:, j) = dvscf(:, p) * psi_kn(r); pj = 1: psir is per chunk column (no cache)
    INTEGER, VALUE :: s0, nnr, nslot, npert, pj
    COMPLEX(dp) :: box(nnr, *), psir(nnr, *), dv(nnr, *)
    INTEGER :: list(*)
    INTEGER :: j, s, k, n, p, b, ir
    j = blockIdx%y
    s = list(s0 + j)
    k = (s - 1) / (nslot * npert) + 1
    n = MOD(s - 1, nslot) + 1
    p = MOD((s - 1) / nslot, npert) + 1
    b = n + nslot * (k - 1)
    IF (pj == 1) b = j
    ir = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ir <= nnr) box(ir, j) = psir(ir, b) * dv(ir, p)
  END SUBROUTINE k_dvpsir
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_gather_add(d, box, list, s0, nnr, npwx, nslot, npert, nksq, npwq, idxq, scale)
    ! apply_dpot: dvpsi(G) += FFT(box)(k+q+G) / N
    INTEGER, VALUE :: s0, nnr, npwx, nslot, npert, nksq
    REAL(dp), VALUE :: scale
    COMPLEX(dp) :: d(npwx, *), box(nnr, *)
    INTEGER :: list(*), npwq(nksq), idxq(npwx, nksq)
    INTEGER :: j, s, k, ig
    j = blockIdx%y
    s = list(s0 + j)
    k = (s - 1) / (nslot * npert) + 1
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ig <= npwq(k)) d(ig, s) = d(ig, s) + box(idxq(ig, k), j) * scale
  END SUBROUTINE k_gather_add
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_drho(drho, box, psir, list, s0, nb, nnr, nslot, npert, wgt, pj)
    ! incdrhoscf: drho(r, p) += sum_j wgt(k_j) conj(psi_{k_j n_j}(r)) dpsi_j(r); deterministic order;
    ! pj = 1: psir is per chunk column (no cache)
    INTEGER, VALUE :: s0, nb, nnr, nslot, npert, pj
    COMPLEX(dp) :: drho(nnr, *), box(nnr, *), psir(nnr, *)
    INTEGER :: list(*)
    REAL(dp) :: wgt(*)
    INTEGER :: ir, j, s, k, n, p, b
    COMPLEX(dp) :: acc(MAXPERT)
    ir = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ir > nnr) RETURN
    DO p = 1, npert
      acc(p) = (0.0_dp, 0.0_dp)
    ENDDO
    DO j = 1, nb
      s = list(s0 + j)
      k = (s - 1) / (nslot * npert) + 1
      n = MOD(s - 1, nslot) + 1
      p = MOD((s - 1) / nslot, npert) + 1
      b = n + nslot * (k - 1)
      IF (pj == 1) b = j
      acc(p) = acc(p) + wgt(k) * (CONJG(psir(ir, b)) * box(ir, j))
    ENDDO
    DO p = 1, npert
      drho(ir, p) = drho(ir, p) + acc(p)
    ENDDO
  END SUBROUTINE k_drho
  !
  !-------------------------------------------------------------------------------------------------
  ! mixed precision (sth_mp): FP32 operator, FP32 PCG with FP64 dots, FP64 round control
  !-------------------------------------------------------------------------------------------------
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_d2s_c(a, b, n)
    ! a (FP32) = b (FP64)
    INTEGER(8), VALUE :: n
    COMPLEX(sp) :: a(n)
    COMPLEX(dp) :: b(n)
    INTEGER(8) :: i
    i = INT(blockIdx%x - 1, 8) * blockDim%x + threadIdx%x
    DO WHILE (i <= n)
      a(i) = CMPLX(b(i), KIND=sp)
      i = i + INT(blockDim%x, 8) * gridDim%x
    ENDDO
  END SUBROUTINE k_d2s_c
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_d2s_r(a, b, n)
    INTEGER(8), VALUE :: n
    REAL(sp) :: a(n)
    REAL(dp) :: b(n)
    INTEGER(8) :: i
    i = INT(blockIdx%x - 1, 8) * blockDim%x + threadIdx%x
    DO WHILE (i <= n)
      a(i) = REAL(b(i), sp)
      i = i + INT(blockDim%x, 8) * gridDim%x
    ENDDO
  END SUBROUTINE k_d2s_r
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_zero_s(a, n)
    INTEGER(8), VALUE :: n
    COMPLEX(sp) :: a(n)
    INTEGER(8) :: i
    i = INT(blockIdx%x - 1, 8) * blockDim%x + threadIdx%x
    DO WHILE (i <= n)
      a(i) = (0.0_sp, 0.0_sp)
      i = i + INT(blockDim%x, 8) * gridDim%x
    ENDDO
  END SUBROUTINE k_zero_s
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_scatter_s(box, x, list, s0, nnr, npwx, nslot, npert, nksq, inv)
    ! FP32 k_scatter_full
    INTEGER, VALUE :: s0, nnr, npwx, nslot, npert, nksq
    COMPLEX(sp) :: box(nnr, *), x(npwx, *)
    INTEGER :: list(*), inv(nnr, nksq)
    INTEGER :: j, s, k, ir, ig
    j = blockIdx%y
    s = list(s0 + j)
    k = (s - 1) / (nslot * npert) + 1
    ir = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ir <= nnr) THEN
      ig = inv(ir, k)
      IF (ig > 0) THEN
        box(ir, j) = x(ig, s)
      ELSE
        box(ir, j) = (0.0_sp, 0.0_sp)
      ENDIF
    ENDIF
  END SUBROUTINE k_scatter_s
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_vmul_s(box, v, nnr)
    INTEGER, VALUE :: nnr
    COMPLEX(sp) :: box(nnr, *)
    REAL(sp) :: v(nnr)
    INTEGER :: ir, j
    j = blockIdx%y
    ir = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ir <= nnr) box(ir, j) = box(ir, j) * v(ir)
  END SUBROUTINE k_vmul_s
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_gather_h_s(y, x, box, list, s0, nnr, npwx, nslot, npert, nksq, nbnd, &
                                             npwq, idxq, g2, etk, scale)
    ! FP32 k_gather_h: y = g2kin x + Vloc x - e_n x ; rows > npwq zeroed
    INTEGER, VALUE :: s0, nnr, npwx, nslot, npert, nksq, nbnd
    REAL(sp), VALUE :: scale
    COMPLEX(sp) :: y(npwx, *), x(npwx, *), box(nnr, *)
    INTEGER :: list(*), npwq(nksq), idxq(npwx, nksq)
    REAL(sp) :: g2(npwx, nksq)
    REAL(dp) :: etk(nbnd, nksq)
    INTEGER :: j, s, k, n, ig
    COMPLEX(sp) :: xv
    j = blockIdx%y
    s = list(s0 + j)
    k = (s - 1) / (nslot * npert) + 1
    n = MOD(s - 1, nslot) + 1
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ig <= npwx) THEN
      IF (ig <= npwq(k)) THEN
        xv = x(ig, s)
        y(ig, s) = g2(ig, k) * xv + box(idxq(ig, k), j) * scale - REAL(etk(n, k), sp) * xv
      ELSE
        y(ig, s) = (0.0_sp, 0.0_sp)
      ENDIF
    ENDIF
  END SUBROUTINE k_gather_h_s
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_coef_h_s(cf2, cf, dmat, nproj, nkb, nslot, npert, nksq, nockq, alpha)
    ! FP32 k_coef_h
    INTEGER, VALUE :: nproj, nkb, nslot, npert, nksq
    REAL(sp), VALUE :: alpha
    COMPLEX(sp) :: cf2(nproj, *), cf(nproj, *), dmat(nkb, nkb)
    INTEGER :: nockq(nksq)
    INTEGER :: i, s, k, j, m
    COMPLEX(sp) :: acc
    s = blockIdx%x
    i = threadIdx%x
    IF (i > nproj) RETURN
    k = (s - 1) / (nslot * npert) + 1
    IF (i <= nkb) THEN
      acc = (0.0_sp, 0.0_sp)
      DO j = 1, nkb
        acc = acc + dmat(i, j) * cf(j, s)
      ENDDO
      cf2(i, s) = acc
    ELSE
      m = i - nkb
      IF (m <= nockq(k)) THEN
        cf2(i, s) = alpha * cf(i, s)
      ELSE
        cf2(i, s) = (0.0_sp, 0.0_sp)
      ENDIF
    ENDIF
  END SUBROUTINE k_coef_h_s
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_check_s(list, nslot, npert, npwx, nksq, npwq, hd, g, h, rho, conv, &
                                          cnt, tol, first)
    ! k_check on FP32 g, h with an FP64 dot and a per-column tolerance tol(s)
    INTEGER, VALUE :: nslot, npert, npwx, nksq, first
    INTEGER :: list(*), npwq(nksq), conv(*), cnt(*)
    REAL(sp) :: hd(npwx, nslot, nksq)
    REAL(dp) :: rho(*), tol(*)
    COMPLEX(sp) :: g(npwx, *), h(npwx, *)
    REAL(dp), SHARED :: red(RB)
    REAL(sp), SHARED :: gam
    INTEGER, SHARED :: cvd
    INTEGER :: s, k, n, ig, nq, tid, st, istat
    REAL(dp) :: acc, rn
    COMPLEX(sp) :: gv
    tid = threadIdx%x
    s = list(blockIdx%x)
    k = (s - 1) / (nslot * npert) + 1
    n = MOD(s - 1, nslot) + 1
    nq = npwq(k)
    acc = 0.0_dp
    DO ig = tid, nq, blockDim%x
      gv = g(ig, s)
      acc = acc + REAL(hd(ig, n, k) * (REAL(gv)**2 + AIMAG(gv)**2), dp)
    ENDDO
    red(tid) = acc
    CALL syncthreads()
    st = blockDim%x / 2
    DO WHILE (st > 0)
      IF (tid <= st) red(tid) = red(tid) + red(tid + st)
      CALL syncthreads()
      st = st / 2
    ENDDO
    IF (tid == 1) THEN
      rn = red(1)
      istat = atomicadd(cnt((s - 1) / nslot + 1), 1)
      IF (SQRT(rn) < tol(s)) THEN
        conv(s) = 1
        cvd = 1
      ELSE
        cvd = 0
        IF (first == 1) THEN
          gam = 0.0_sp
        ELSE
          gam = REAL(rn / rho(s), sp)
        ENDIF
        rho(s) = rn
      ENDIF
    ENDIF
    CALL syncthreads()
    IF (cvd == 0) THEN
      IF (first == 1) THEN
        DO ig = tid, nq, blockDim%x
          h(ig, s) = -(g(ig, s) * hd(ig, n, k))
        ENDDO
      ELSE
        DO ig = tid, nq, blockDim%x
          h(ig, s) = -(g(ig, s) * hd(ig, n, k)) + gam * h(ig, s)
        ENDDO
      ENDIF
    ENDIF
  END SUBROUTINE k_check_s
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_update_s(list, nslot, npert, npwx, nksq, npwq, h, g, t, e)
    ! k_update on FP32 vectors, FP64 dots: lambda = -<h|g>/<h|t> ; e += lambda h ; g += lambda t
    INTEGER, VALUE :: nslot, npert, npwx, nksq
    INTEGER :: list(*), npwq(nksq)
    COMPLEX(sp) :: h(npwx, *), g(npwx, *), t(npwx, *), e(npwx, *)
    REAL(dp), SHARED :: ra(RB), rc(RB)
    REAL(sp), SHARED :: lam
    INTEGER :: s, k, ig, nq, tid, st
    REAL(dp) :: a, c
    COMPLEX(sp) :: hv, gv, tv
    tid = threadIdx%x
    s = list(blockIdx%x)
    k = (s - 1) / (nslot * npert) + 1
    nq = npwq(k)
    a = 0.0_dp
    c = 0.0_dp
    DO ig = tid, nq, blockDim%x
      hv = h(ig, s)
      gv = g(ig, s)
      tv = t(ig, s)
      a = a + REAL(REAL(hv) * REAL(gv) + AIMAG(hv) * AIMAG(gv), dp)
      c = c + REAL(REAL(hv) * REAL(tv) + AIMAG(hv) * AIMAG(tv), dp)
    ENDDO
    ra(tid) = a
    rc(tid) = c
    CALL syncthreads()
    st = blockDim%x / 2
    DO WHILE (st > 0)
      IF (tid <= st) THEN
        ra(tid) = ra(tid) + ra(tid + st)
        rc(tid) = rc(tid) + rc(tid + st)
      ENDIF
      CALL syncthreads()
      st = st / 2
    ENDDO
    IF (tid == 1) lam = REAL(-ra(1) / rc(1), sp)
    CALL syncthreads()
    DO ig = tid, nq, blockDim%x
      e(ig, s) = e(ig, s) + lam * h(ig, s)
      g(ig, s) = g(ig, s) + lam * t(ig, s)
    ENDDO
  END SUBROUTINE k_update_s
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_mp_round(list, nslot, npert, npwx, nksq, npwq, hd, ax, b, sg, se, conv, &
                                           iconv, tol, thresh, eta, safe)
    ! start of a round, per column of list: r = Ax - b (FP64); converged (conv = 1) if
    ! sqrt(<h_diag r, r>) < thresh; otherwise g32 = r, e32 = 0, inner tolerance
    ! tol = max(safe*thresh, eta*|r|), iconv = 0
    INTEGER, VALUE :: nslot, npert, npwx, nksq
    REAL(dp), VALUE :: thresh, eta, safe
    INTEGER :: list(*), npwq(nksq), conv(*), iconv(*)
    REAL(dp) :: hd(npwx, nslot, nksq), tol(*)
    COMPLEX(dp) :: ax(npwx, *), b(npwx, *)
    COMPLEX(sp) :: sg(npwx, *), se(npwx, *)
    REAL(dp), SHARED :: red(RB)
    INTEGER :: s, k, n, ig, nq, tid, st
    REAL(dp) :: acc, rn
    COMPLEX(dp) :: r
    tid = threadIdx%x
    s = list(blockIdx%x)
    k = (s - 1) / (nslot * npert) + 1
    n = MOD(s - 1, nslot) + 1
    nq = npwq(k)
    acc = 0.0_dp
    DO ig = tid, npwx, blockDim%x
      IF (ig <= nq) THEN
        r = ax(ig, s) - b(ig, s)
        acc = acc + hd(ig, n, k) * (REAL(r, dp)**2 + AIMAG(r)**2)
        sg(ig, s) = CMPLX(r, KIND=sp)
      ELSE
        sg(ig, s) = (0.0_sp, 0.0_sp)
      ENDIF
      se(ig, s) = (0.0_sp, 0.0_sp)
    ENDDO
    red(tid) = acc
    CALL syncthreads()
    st = blockDim%x / 2
    DO WHILE (st > 0)
      IF (tid <= st) red(tid) = red(tid) + red(tid + st)
      CALL syncthreads()
      st = st / 2
    ENDDO
    IF (tid == 1) THEN
      rn = SQRT(red(1))
      IF (rn < thresh) THEN
        conv(s) = 1
      ELSE
        conv(s) = 0
        iconv(s) = 0
        tol(s) = MAX(safe * thresh, eta * rn)
      ENDIF
    ENDIF
  END SUBROUTINE k_mp_round
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_mp_addx(list, nslot, npert, npwx, nksq, npwq, x, e)
    ! x (FP64) += e (FP32), per column of list
    INTEGER, VALUE :: nslot, npert, npwx, nksq
    INTEGER :: list(*), npwq(nksq)
    COMPLEX(dp) :: x(npwx, *)
    COMPLEX(sp) :: e(npwx, *)
    INTEGER :: s, k, ig
    s = list(blockIdx%x)
    k = (s - 1) / (nslot * npert) + 1
    DO ig = threadIdx%x, npwq(k), blockDim%x
      x(ig, s) = x(ig, s) + CMPLX(e(ig, s), KIND=dp)
    ENDDO
  END SUBROUTINE k_mp_addx
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_copy_list(a, b, list, npwx)
    ! a(:, s) = b(:, s) for s in list
    INTEGER, VALUE :: npwx
    INTEGER :: list(*)
    COMPLEX(dp) :: a(npwx, *), b(npwx, *)
    INTEGER :: s, ig
    s = list(blockIdx%x)
    DO ig = threadIdx%x, npwx, blockDim%x
      a(ig, s) = b(ig, s)
    ENDDO
  END SUBROUTINE k_copy_list
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_ps(ps, da, db, u, at, ia, nkb, nproj, nslot, npert, nksq, nat3, ctp)
    ! nonlocal coefficients of dV_bare psi (dvqpsi_us_only, NC) for every slot s = (n, p, k):
    !   ia = 0: ps(kb, s) = sum_a u(3(at(kb)-1)+a, p) da(kb, n, a, k)
    !   ia > 0: ps(kb, s) = ctp u(3(at(kb)-1)+ia, p) db(kb, n, k)          (ctp = -i tpiba)
    INTEGER, VALUE :: ia, nkb, nproj, nslot, npert, nksq, nat3
    COMPLEX(dp), VALUE :: ctp
    COMPLEX(dp) :: ps(nproj, *), da(nkb, nslot, 3, nksq), db(nkb, nslot, nksq), u(nat3, npert)
    INTEGER :: at(nkb)
    INTEGER :: s, kb, n, p, k, mu, a
    COMPLEX(dp) :: acc
    s = blockIdx%x
    kb = threadIdx%x
    IF (kb > nkb) RETURN
    n = MOD(s - 1, nslot) + 1
    p = MOD((s - 1) / nslot, npert) + 1
    k = (s - 1) / (nslot * npert) + 1
    mu = 3 * (at(kb) - 1)
    IF (ia == 0) THEN
      acc = (0.0_dp, 0.0_dp)
      DO a = 1, 3
        acc = acc + da(kb, n, a, k) * u(mu + a, p)
      ENDDO
      ps(kb, s) = acc
    ELSE
      ps(kb, s) = db(kb, n, k) * (0.0_dp, -1.0_dp) * u(mu + ia, p) * REAL(ctp, dp)
    ENDIF
  END SUBROUTINE k_ps
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_kqg_axpy(y, t, kqg, ipol, npwx, nslot, npert, nksq, ntot)
    ! y(:, s) += t(:, s) * kqg(:, ipol, k(s))
    INTEGER, VALUE :: ipol, npwx, nslot, npert, nksq
    INTEGER(8), VALUE :: ntot
    COMPLEX(dp) :: y(*), t(*)
    REAL(dp) :: kqg(npwx, 3, nksq)
    INTEGER(8) :: i
    INTEGER :: ig, s, k
    i = INT(blockIdx%x - 1, 8) * blockDim%x + threadIdx%x
    DO WHILE (i <= ntot)
      ig = INT(MOD(i - 1, INT(npwx, 8))) + 1
      s = INT((i - 1) / npwx) + 1
      k = (s - 1) / (nslot * npert) + 1
      y(i) = y(i) + t(i) * kqg(ig, ipol, k)
      i = i + INT(blockDim%x, 8) * gridDim%x
    ENDDO
  END SUBROUTINE k_kqg_axpy
  !
  !=================================================================================================
  ! host helpers
  !=================================================================================================
  !
  SUBROUTINE chk(istat, what)
    INTEGER, INTENT(IN) :: istat
    CHARACTER(LEN=*), INTENT(IN) :: what
    IF (istat /= 0) THEN
      WRITE(*, '(A,A,A,I0)') 'sth_engine: ', what, ' failed, status ', istat
      STOP 1
    ENDIF
  END SUBROUTINE chk
  !
  SUBROUTINE kerr(what)
    CHARACTER(LEN=*), INTENT(IN) :: what
    INTEGER :: istat
    istat = cudaGetLastError()
    IF (istat /= cudaSuccess) THEN
      WRITE(*, '(A,A,A,A)') 'sth_engine: kernel ', what, ': ', TRIM(cudaGetErrorString(istat))
      STOP 1
    ENDIF
  END SUBROUTINE kerr
  !
  SUBROUTINE errore_stop(what, ier)
    CHARACTER(LEN=*), INTENT(IN) :: what
    INTEGER, INTENT(IN) :: ier
    WRITE(*, '(A,A,A,I4)') 'sth_engine: ', what, ' failed, code', ier
    STOP 1
  END SUBROUTINE errore_stop
  !
  SUBROUTINE sth_mem_trim()
    !! return the device blocks freed so far (kept by the NVHPC runtime) to the driver
    INTEGER :: istat
    istat = cudaDeviceSynchronize()
    CALL acc_clear_freelists()
  END SUBROUTINE sth_mem_trim
  !
  SUBROUTINE sth_mem_sample(used)
    !! update the high-water mark of device memory in use; optionally return the current use (bytes)
    INTEGER(8), INTENT(OUT), OPTIONAL :: used
    INTEGER(8) :: freeb, totb
    INTEGER :: istat
    istat = cudaMemGetInfo(freeb, totb)
    mem_hw = MAX(mem_hw, totb - freeb)
    IF (PRESENT(used)) used = totb - freeb
  END SUBROUTINE sth_mem_sample
  !
  SUBROUTINE sth_mem_take_hw(hw, total)
    !! high-water mark since the last call (bytes), then reset it; total device memory (bytes)
    INTEGER(8), INTENT(OUT) :: hw, total
    INTEGER(8) :: freeb
    INTEGER :: istat
    istat = cudaMemGetInfo(freeb, total)
    hw = MAX(mem_hw, total - freeb)
    mem_hw = 0
  END SUBROUTINE sth_mem_take_hw
  !
  SUBROUTINE sth_mem_log(tag, need)
    !! rtsc b2 (RTSC_STH_MEMLOG=1; read-only): the engine's device arrays by component (GiB, from their
    !! allocated sizes), the driver's in-use total and the high-water so far. Used for the CBSR-B2 memory desk.
    !! need (optional): the setup's own estimate (bytes), printed with psir_cached
    CHARACTER(LEN=*), INTENT(IN) :: tag
    INTEGER(8), INTENT(IN), OPTIONAL :: need
    INTEGER(8) :: freeb, totb, bw, bev, bps, bbx, bhd, bix, binv, bkq, bg2, bnl, bcol, bcf, bpsx, bdv, bxp, bsp, bcb, bsum
    INTEGER :: istat, ios
    CHARACTER(LEN=8) :: sv
    LOGICAL, SAVE :: first = .TRUE., on = .FALSE.
    REAL(dp), PARAMETER :: g = 1024.0_dp**3
    IF (first) THEN
      CALL get_environment_variable('RTSC_STH_MEMLOG', sv, STATUS=ios)
      on = ios == 0 .AND. TRIM(sv) == '1'
      first = .FALSE.
    ENDIF
    IF (.NOT. on) RETURN
    istat = cudaMemGetInfo(freeb, totb)
    bw = 0; bev = 0; bps = 0; bbx = 0; bhd = 0; bix = 0; binv = 0; bkq = 0; bg2 = 0; bnl = 0
    bcol = 0; bcf = 0; bpsx = 0; bdv = 0; bxp = 0; bsp = 0; bcb = 0
    IF (ALLOCATED(d_W)) bw = 16_8 * SIZE(d_W, KIND=8)
    IF (ALLOCATED(d_evc)) bev = 16_8 * SIZE(d_evc, KIND=8)
    IF (ALLOCATED(d_psir)) bps = 16_8 * SIZE(d_psir, KIND=8)
    IF (ALLOCATED(d_box)) bbx = bbx + 16_8 * SIZE(d_box, KIND=8)
    IF (ALLOCATED(d_box2)) bbx = bbx + 16_8 * SIZE(d_box2, KIND=8)
    IF (ALLOCATED(d_hd)) bhd = 8_8 * SIZE(d_hd, KIND=8)
    IF (ALLOCATED(d_wwg)) bhd = bhd + 8_8 * SIZE(d_wwg, KIND=8)
    IF (ALLOCATED(d_idxk)) bix = 4_8 * (SIZE(d_idxk, KIND=8) + SIZE(d_idxkq, KIND=8))
    IF (ALLOCATED(d_inv)) binv = 4_8 * SIZE(d_inv, KIND=8)
    IF (ALLOCATED(d_kqgq)) bkq = 8_8 * SIZE(d_kqgq, KIND=8)
    IF (ALLOCATED(d_g2)) bg2 = 8_8 * SIZE(d_g2, KIND=8)
    IF (ALLOCATED(d_db)) bnl = 16_8 * (SIZE(d_db, KIND=8) + SIZE(d_da, KIND=8))
    IF (ALLOCATED(d_dvb)) bcol = 16_8 * (SIZE(d_dvb, KIND=8) + SIZE(d_d0, KIND=8) + SIZE(d_dpsi, KIND=8) &
                                         + SIZE(d_g, KIND=8) + SIZE(d_h, KIND=8) + SIZE(d_t, KIND=8))
    IF (ALLOCATED(d_cf)) bcf = 16_8 * (SIZE(d_cf, KIND=8) + SIZE(d_cf2, KIND=8))
    IF (ALLOCATED(d_ps)) bpsx = 16_8 * SIZE(d_ps, KIND=8)
    IF (ALLOCATED(d_dv)) bdv = 16_8 * (SIZE(d_dv, KIND=8) + SIZE(d_drho, KIND=8))
    IF (ALLOCATED(d_xp)) bxp = 16_8 * SIZE(d_xp, KIND=8)
    IF (ALLOCATED(s_W)) bsp = 8_8 * (SIZE(s_W, KIND=8) + SIZE(s_box, KIND=8)) + 4_8 * (SIZE(s_hd, KIND=8) + SIZE(s_g2, KIND=8))
    IF (ALLOCATED(s_e)) bsp = bsp + 8_8 * (SIZE(s_e, KIND=8) + SIZE(s_g, KIND=8) + SIZE(s_h, KIND=8) + SIZE(s_t, KIND=8) &
                                           + SIZE(s_cf, KIND=8) + SIZE(s_cf2, KIND=8))
    DO istat = 1, CB_MAXREP
      IF (ALLOCATED(cb_rep(istat)%u)) bcb = bcb + 16_8 * SIZE(cb_rep(istat)%u, KIND=8)
      IF (ALLOCATED(cb_rep(istat)%u32)) bcb = bcb + 8_8 * SIZE(cb_rep(istat)%u32, KIND=8)
    ENDDO
    IF (ALLOCATED(cb_tmp)) THEN
      DO istat = 1, SIZE(cb_tmp)
        IF (ALLOCATED(cb_tmp(istat)%u)) bcb = bcb + 16_8 * SIZE(cb_tmp(istat)%u, KIND=8)
        IF (ALLOCATED(cb_tmp(istat)%u32)) bcb = bcb + 8_8 * SIZE(cb_tmp(istat)%u32, KIND=8)
      ENDDO
    ENDIF
    bsum = bw + bev + bps + bbx + bhd + bix + binv + bkq + bg2 + bnl + bcol + bcf + bpsx + bdv + bxp + bsp + bcb
    WRITE(*, '(A,A,A,8I8,I10)') '     rtsc memlog ', tag, ' dims npwx nslot npert nksq nnr nproj nbnd bmax nbox:', &
         npwx, nslot, npert, nksq, nnr, nproj, nbnd, sth_bmax, nbox
    IF (PRESENT(need)) WRITE(*, '(A,A,A,F8.3,A,L2,A,F8.3)') '     rtsc memlog ', tag, ' need GiB', REAL(need, dp) / g, &
         ' psir_cached', psir_cached, ' free GiB', REAL(freeb, dp) / g
    WRITE(*, '(A,A,A,3F8.3)') '     rtsc memlog ', tag, ' GiB used hw engine_sum:', &
         REAL(totb - freeb, dp) / g, REAL(MAX(mem_hw, totb - freeb), dp) / g, REAL(bsum, dp) / g
    WRITE(*, '(A,A,A,17F8.3)') '     rtsc memlog ', tag, &
         ' GiB W evc psir box hd+wwg idx inv kqg g2 db+da cols6 cf+cf2 ps dv+drho xp fp32 cbsr:', &
         REAL(bw, dp) / g, REAL(bev, dp) / g, REAL(bps, dp) / g, REAL(bbx, dp) / g, REAL(bhd, dp) / g, REAL(bix, dp) / g, &
         REAL(binv, dp) / g, REAL(bkq, dp) / g, REAL(bg2, dp) / g, REAL(bnl, dp) / g, REAL(bcol, dp) / g, REAL(bcf, dp) / g, &
         REAL(bpsx, dp) / g, REAL(bdv, dp) / g, REAL(bxp, dp) / g, REAL(bsp, dp) / g, REAL(bcb, dp) / g
  END SUBROUTINE sth_mem_log
  !
  REAL(dp) FUNCTION wtime()
    INTEGER(8) :: c, r
    CALL SYSTEM_CLOCK(c, r)
    wtime = REAL(c, dp) / REAL(r, dp)
  END FUNCTION wtime
  !
  SUBROUTINE tsync()
    INTEGER :: istat
    istat = cudaStreamSynchronize(strm)
    CALL chk(istat, 'cudaStreamSynchronize')
  END SUBROUTINE tsync
  !
  INTEGER FUNCTION nblk(n, b)
    INTEGER, INTENT(IN) :: n, b
    nblk = MAX(1, (n + b - 1) / b)
  END FUNCTION nblk
  !
  SUBROUTINE zero_c(a, n)
    COMPLEX(dp), DEVICE :: a(*)
    INTEGER(8), INTENT(IN) :: n
    IF (n <= 0) RETURN
    CALL k_zero<<<MIN(INT((n + TPB - 1) / TPB), 4096), TPB, 0, strm>>>(a, n)
    CALL kerr('zero')
  END SUBROUTINE zero_c
  !
  ! largest plan index whose batch (2**ip) does not exceed nb columns: the column chunks are exact
  ! powers of two, so no FFT runs on padding boxes (cuFFT gives the same bits for any batch size)
  INTEGER FUNCTION plan_for(nb)
    INTEGER, INTENT(IN) :: nb
    plan_for = nplan - 1
    DO WHILE (plan_for > 0 .AND. 2**plan_for > nb)
      plan_for = plan_for - 1
    ENDDO
  END FUNCTION plan_for
  !
  SUBROUTINE make_plans()
    INTEGER :: ip, istat, n(3), emb(3)
    IF (nplan > 0 .AND. plan_nnr == nnr .AND. ALL(plan_nr == nr) .AND. 2**(nplan - 1) == sth_bmax) RETURN
    CALL destroy_plans()
    n = (/ nr(3), nr(2), nr(1) /)
    emb = (/ nrx(3), nrx(2), nrx(1) /)
    nplan = 0
    DO ip = 0, MAXPLAN - 1
      IF (2**ip > sth_bmax) EXIT
      istat = cufftPlanMany(plans(ip), 3, n, emb, 1, nnr, emb, 1, nnr, CUFFT_Z2Z, 2**ip)
      CALL chk(istat, 'cufftPlanMany')
      istat = cufftSetStream(plans(ip), strm)
      CALL chk(istat, 'cufftSetStream')
      nplan = ip + 1
    ENDDO
    plan_nnr = nnr
    plan_nr = nr
  END SUBROUTINE make_plans
  !
  SUBROUTINE psir_part_build()
    !! RTSC_STH_PSIR_PART: cache psi_kn(r) for the first psir_kc k-points, as many as fit after a reserve for the irrep
    !! arrays (8 column arrays at sth_npmax), the CBSR chunk buffers and RTSC_STH_PSIR_RES GiB (default 1.5)
    INTEGER(8) :: freeb, totb, res, per
    INTEGER :: istat, b0, nb, ip, nbx
    REAL(dp) :: rg
    CHARACTER(LEN=16) :: sv
    rg = 1.5_dp
    CALL get_environment_variable('RTSC_STH_PSIR_RES', sv, STATUS=istat)
    IF (istat == 0 .AND. LEN_TRIM(sv) > 0) READ(sv, *, IOSTAT=istat) rg
    res = INT(rg * 1024.0_dp**3, 8) + 8_8 * 16_8 * npwx * nslot * sth_npmax * nksq
    IF (cbsr_q .AND. cbsr_rf) res = res + 48_8 * cb_n0 * cb_cmax
    IF (cbsr_q .AND. .NOT. cbsr_rf) res = res + 2_8 * 16_8 * cb_n0 * nslot * sth_npmax * nksq
    CALL sth_mem_trim()
    istat = cudaMemGetInfo(freeb, totb)
    per = 16_8 * nnr * nslot
    psir_kc = INT(MAX(0_8, MIN(INT(nksq, 8), (freeb - res) / per)))
    ! RTSC_STH_PSIR_KC: a cap on the cached k-points (tests the cached / recomputed split where everything would fit)
    CALL get_environment_variable('RTSC_STH_PSIR_KC', sv, STATUS=istat)
    IF (istat == 0 .AND. LEN_TRIM(sv) > 0) THEN
      READ(sv, *, IOSTAT=istat) b0
      IF (istat == 0) psir_kc = MAX(0, MIN(psir_kc, b0))
    ENDIF
    IF (psir_kc > 0) THEN
      nbx = nslot * psir_kc
      ALLOCATE(d_psir(nnr, nbx))
      CALL zero_c(d_psir, INT(nnr, 8) * nbx)
      CALL k_scatter_evc<<<dim3(nblk(npwx, TPB), nbx, 1), TPB, 0, strm>>>(d_psir, d_evc, nnr, npwx, &
           nslot, psir_kc, d_npw, d_nock, d_idxk)
      CALL kerr('scatter_evc (part)')
      b0 = 0
      DO WHILE (b0 < nbx)
        nb = MIN(sth_bmax, nbx - b0)
        ip = nplan - 1
        DO WHILE (2**ip > nb)
          ip = ip - 1
        ENDDO
        CALL fft_exec(ip, d_psir(:, b0 + 1:b0 + 2**ip), CUFFT_INVERSE)
        b0 = b0 + 2**ip
      ENDDO
      CALL tsync()
    ENDIF
    CALL sth_mem_sample()
    WRITE(*, '(A,I6,A,I6,A,F8.2,A,F8.2,A,I2,A)') '     rtsc psir part: psi(r) cached for ', psir_kc, ' of ', nksq, &
         ' k (', REAL(per * psir_kc, dp) / 1024.0_dp**3, ' GiB; free before ', REAL(freeb, dp) / 1024.0_dp**3, &
         ' GiB, reserve at npert ', sth_npmax, ')'
  END SUBROUTINE psir_part_build
  !
  SUBROUTINE psir_chunk(s0, nb, ip)
    !! no-cache mode: psi_kn(r) of the nb = 2**ip columns list(s0+1 .. s0+nb) into d_box2
    INTEGER, INTENT(IN) :: s0, nb, ip
    CALL zero_c(d_box2, INT(nnr, 8) * nb)
    CALL k_psir_chunk<<<dim3(nblk(npwx, TPB), nb, 1), TPB, 0, strm>>>(d_box2, d_evc, d_valid, s0, nnr, npwx, &
         nslot, npert, nksq, d_npw, d_idxk)
    CALL kerr('psir_chunk')
    CALL fft_exec(ip, d_box2, CUFFT_INVERSE)
  END SUBROUTINE psir_chunk
  !
  SUBROUTINE destroy_plans()
    INTEGER :: ip, istat
    DO ip = 0, nplan - 1
      istat = cufftDestroy(plans(ip))
      plans(ip) = 0
    ENDDO
    nplan = 0
    plan_nnr = 0
    CALL destroy_plans32()
  END SUBROUTINE destroy_plans
  !
  SUBROUTINE make_plans32()
    ! C2C plans with the same batch sizes as the Z2Z plans (sth_mp)
    INTEGER :: ip, istat, n(3), emb(3)
    IF (nplan32 == nplan) RETURN
    CALL destroy_plans32()
    n = (/ nr(3), nr(2), nr(1) /)
    emb = (/ nrx(3), nrx(2), nrx(1) /)
    DO ip = 0, nplan - 1
      istat = cufftPlanMany(plans32(ip), 3, n, emb, 1, nnr, emb, 1, nnr, CUFFT_C2C, 2**ip)
      CALL chk(istat, 'cufftPlanMany (C2C)')
      istat = cufftSetStream(plans32(ip), strm)
      CALL chk(istat, 'cufftSetStream (C2C)')
      nplan32 = ip + 1
    ENDDO
  END SUBROUTINE make_plans32
  !
  SUBROUTINE destroy_plans32()
    INTEGER :: ip, istat
    DO ip = 0, nplan32 - 1
      istat = cufftDestroy(plans32(ip))
      plans32(ip) = 0
    ENDDO
    nplan32 = 0
  END SUBROUTINE destroy_plans32
  !
  SUBROUTINE fft_exec_s(ip, a, dir)
    INTEGER, INTENT(IN) :: ip, dir
    COMPLEX(sp), DEVICE :: a(*)
    INTEGER :: istat
    istat = cufftExecC2C(plans32(ip), a, a, dir)
    CALL chk(istat, 'cufftExecC2C')
  END SUBROUTINE fft_exec_s
  !
  SUBROUTINE gemm_proj_s(x, cf)
    COMPLEX(sp), DEVICE :: x(npwx, *), cf(nproj, *)
    INTEGER :: istat
    istat = cublasCgemmStridedBatched(hblas, CUBLAS_OP_C, CUBLAS_OP_N, nproj, nslot * npert, npwx, &
            (1.0_sp, 0.0_sp), s_W, npwx, INT(npwx, 8) * nproj, x, npwx, INT(npwx, 8) * nslot * npert, &
            (0.0_sp, 0.0_sp), cf, nproj, INT(nproj, 8) * nslot * npert, nksq)
    CALL chk(istat, 'cublasCgemmStridedBatched (proj)')
  END SUBROUTINE gemm_proj_s
  !
  SUBROUTINE gemm_back_s(cf, y)
    COMPLEX(sp), DEVICE :: cf(nproj, *), y(npwx, *)
    INTEGER :: istat
    istat = cublasCgemmStridedBatched(hblas, CUBLAS_OP_N, CUBLAS_OP_N, npwx, nslot * npert, nproj, &
            (1.0_sp, 0.0_sp), s_W, npwx, INT(npwx, 8) * nproj, cf, nproj, INT(nproj, 8) * nslot * npert, &
            (1.0_sp, 0.0_sp), y, npwx, INT(npwx, 8) * nslot * npert, nksq)
    CALL chk(istat, 'cublasCgemmStridedBatched (back)')
  END SUBROUTINE gemm_back_s
  !
  SUBROUTINE apply_op_s(x, y, list, nl)
    ! FP32 apply_op: y = (H_kq - e_n + alpha_pv P_v) x on the nl columns list(1:nl)
    COMPLEX(sp), DEVICE :: x(npwx, *), y(npwx, *)
    INTEGER, DEVICE :: list(*)
    INTEGER, INTENT(IN) :: nl
    INTEGER :: s0, nb, ip
    REAL(sp) :: scale
    scale = 1.0_sp / REAL(nr(1) * nr(2) * nr(3), sp)
    CALL gemm_proj_s(x, s_cf)
    CALL k_coef_h_s<<<nsl, 32 * ((nproj + 31) / 32), 0, strm>>>(s_cf2, s_cf, s_dmat, nproj, nkb, nslot, &
         npert, nksq, d_nockq, REAL(alpha_pv, sp))
    CALL kerr('coef_h_s')
    IF (fdx_on) THEN
      ip = fdx_local_s(x, y, list, nl, nslot, npert, npwx, nksq, nbnd, s_vrs, d_npwq, s_g2, d_etk, scale, strm)
      IF (ip /= 0) CALL errore_stop('fdx_local_s', ip)
      CALL gemm_back_s(s_cf2, y)
      RETURN
    ENDIF
    IF (pfs_ok()) THEN
      CALL pfs_local(x, y, list, nl, nslot, npert, npwx, nksq, nbnd, s_vrs, d_npwq, s_g2, d_etk, scale, strm)
      CALL gemm_back_s(s_cf2, y)
      RETURN
    ENDIF
    s0 = 0
    DO WHILE (s0 < nl)
      nb = MIN(sth_bmax, nl - s0)
      ip = plan_for(nb)
      nb = 2**ip
      CALL k_scatter_s<<<dim3(nblk(nnr, TPB), nb, 1), TPB, 0, strm>>>(s_box, x, list, s0, nnr, npwx, &
           nslot, npert, nksq, d_inv)
      CALL kerr('scatter_s')
      CALL fft_exec_s(ip, s_box, CUFFT_INVERSE)
      CALL k_vmul_s<<<dim3(nblk(nnr, TPB), nb, 1), TPB, 0, strm>>>(s_box, s_vrs, nnr)
      CALL kerr('vmul_s')
      CALL fft_exec_s(ip, s_box, CUFFT_FORWARD)
      CALL k_gather_h_s<<<dim3(nblk(npwx, TPB), nb, 1), TPB, 0, strm>>>(y, x, s_box, list, s0, nnr, npwx, &
           nslot, npert, nksq, nbnd, d_npwq, d_idxkq, s_g2, d_etk, scale)
      CALL kerr('gather_h_s')
      s0 = s0 + nb
    ENDDO
    CALL gemm_back_s(s_cf2, y)
  END SUBROUTINE apply_op_s
  !
  SUBROUTINE d2s_c(a, b, n)
    COMPLEX(sp), DEVICE :: a(*)
    COMPLEX(dp), DEVICE :: b(*)
    INTEGER(8), INTENT(IN) :: n
    CALL k_d2s_c<<<MIN(INT((n + TPB - 1) / TPB), 4096), TPB, 0, strm>>>(a, b, n)
    CALL kerr('d2s_c')
  END SUBROUTINE d2s_c
  !
  SUBROUTINE d2s_r(a, b, n)
    REAL(sp), DEVICE :: a(*)
    REAL(dp), DEVICE :: b(*)
    INTEGER(8), INTENT(IN) :: n
    CALL k_d2s_r<<<MIN(INT((n + TPB - 1) / TPB), 4096), TPB, 0, strm>>>(a, b, n)
    CALL kerr('d2s_r')
  END SUBROUTINE d2s_r
  !
  SUBROUTINE zero_s(a, n)
    COMPLEX(sp), DEVICE :: a(*)
    INTEGER(8), INTENT(IN) :: n
    IF (n <= 0) RETURN
    CALL k_zero_s<<<MIN(INT((n + TPB - 1) / TPB), 4096), TPB, 0, strm>>>(a, n)
    CALL kerr('zero_s')
  END SUBROUTINE zero_s
  !
  SUBROUTINE mp_cg(thresh, nout)
    ! the FP32 PCG of one round on the columns d_oact(1:nout): solve A e = -r from e = 0 (r = g32 on
    ! entry) to the per-column tolerance d_tol; the iteration counts go to d_cnt / d_kter as in the
    ! FP64 path
    REAL(dp), INTENT(IN) :: thresh
    INTEGER, INTENT(IN) :: nout
    INTEGER :: istat, iter, nact, first
    CALL k_icopy<<<nblk(nout, TPB), TPB, 0, strm>>>(d_act, d_oact, nout)
    CALL kerr('icopy (mp)')
    nact = nout
    DO iter = 1, maxter
      first = 0
      IF (iter == 1) first = 1
      CALL k_check_s<<<nact, sth_rb, 0, strm>>>(d_act, nslot, npert, npwx, nksq, d_npwq, s_hd, s_g, s_h, &
           d_rho, d_iconv, d_cnt, d_tol, first)
      CALL kerr('check_s')
      CALL k_kter<<<nblk(ngrp, TPB), TPB, 0, strm>>>(ngrp, npert, nksq, d_nock, d_cnt, d_kter)
      CALL kerr('kter')
      CALL k_compact<<<1, 1024, 0, strm>>>(d_act, d_act2, nact, d_iconv, d_nact)
      CALL kerr('compact')
      istat = cudaMemcpyAsync(h_nact, d_nact, 1, strm)
      CALL tsync()
      nact = h_nact(1)
      sth_nsteps = sth_nsteps + 1
      IF (nact == 0) EXIT
      CALL apply_op_s(s_h, s_t, d_act, nact)
      sth_n32 = sth_n32 + nact
      CALL k_update_s<<<nact, sth_rb, 0, strm>>>(d_act, nslot, npert, npwx, nksq, d_npwq, s_h, s_g, s_t, s_e)
      CALL kerr('update_s')
    ENDDO
  END SUBROUTINE mp_cg
  !
  SUBROUTINE mp_solve(first_iter, thresh)
    ! the CG part of sth_engine_solve in mixed precision (see the module header); sets d_conv
    INTEGER :: istat, nout, ir
    LOGICAL, INTENT(IN) :: first_iter
    REAL(dp), INTENT(IN) :: thresh
    REAL(dp) :: ta
    sth_n64 = 0; sth_n32 = 0; sth_nround = 0
    sth_t_cg64 = 0
    sth_nsteps = 0
    d_conv = 0
    d_cnt = 0
    d_kter = 0.0_dp
    d_rho = 0.0_dp
    istat = cudaDeviceSynchronize()
    IF (first_iter) THEN
      CALL zero_c(d_dpsi, INT(npwx, 8) * nsl)
      CALL zero_c(d_g, INT(npwx, 8) * nsl)          ! A 0 = 0
      ax_valid = .TRUE.
    ENDIF
    IF (.NOT. ax_valid) THEN
      ! warm start from outside (restart, or a previous FP64-path solve): Ax in FP64 once
      CALL apply_op(d_dpsi, d_t, d_valid, nvalid)
      CALL k_copy_list<<<nvalid, TPB, 0, strm>>>(d_g, d_t, d_valid, npwx)
      CALL kerr('copy_list')
      sth_n64 = sth_n64 + nvalid
      ax_valid = .TRUE.
    ENDIF
    CALL k_icopy<<<nblk(nvalid, TPB), TPB, 0, strm>>>(d_oact, d_valid, nvalid)
    nout = nvalid
    DO ir = 1, sth_mp_maxround + 1
      ! r = Ax - b; test; FP32 start vectors for the columns still open
      CALL k_mp_round<<<nout, sth_rb, 0, strm>>>(d_oact, nslot, npert, npwx, nksq, d_npwq, d_hd, d_g, d_d0, &
           s_g, s_e, d_conv, d_iconv, d_tol, thresh, sth_mp_eta, sth_mp_safe)
      CALL kerr('mp_round')
      CALL k_compact<<<1, 1024, 0, strm>>>(d_oact, d_act2, nout, d_conv, d_nact)
      CALL kerr('compact (mp)')
      istat = cudaMemcpyAsync(h_nact, d_nact, 1, strm)
      CALL tsync()
      nout = h_nact(1)
      IF (nout == 0 .OR. ir > sth_mp_maxround) EXIT
      sth_nround = ir
      CALL mp_cg(thresh, nout)
      ! x += e ; Ax = A x (FP64, into d_t, then copied to d_g on these columns only: apply_op leaves
      ! the other slots of its output undefined)
      CALL k_mp_addx<<<nout, TPB, 0, strm>>>(d_oact, nslot, npert, npwx, nksq, d_npwq, d_dpsi, s_e)
      CALL kerr('mp_addx')
      IF (sth_timing) THEN
        CALL tsync(); ta = wtime()
      ENDIF
      CALL apply_op(d_dpsi, d_t, d_oact, nout)
      CALL k_copy_list<<<nout, TPB, 0, strm>>>(d_g, d_t, d_oact, npwx)
      CALL kerr('copy_list')
      IF (sth_timing) THEN
        CALL tsync(); sth_t_cg64 = sth_t_cg64 + wtime() - ta
      ENDIF
      sth_n64 = sth_n64 + nout
    ENDDO
    sth_ncolapp = sth_n32 + sth_n64
  END SUBROUTINE mp_solve
  !
  SUBROUTINE fft_exec(ip, a, dir)
    INTEGER, INTENT(IN) :: ip, dir
    COMPLEX(dp), DEVICE :: a(*)
    INTEGER :: istat
    istat = cufftExecZ2Z(plans(ip), a, a, dir)
    CALL chk(istat, 'cufftExecZ2Z')
  END SUBROUTINE fft_exec
  !
  SUBROUTINE gemm_proj(x, cf)
    ! cf(:, slots of k) = W_k^H x(:, slots of k), batched over k
    COMPLEX(dp), DEVICE :: x(npwx, *), cf(nproj, *)
    INTEGER :: istat
    istat = cublasZgemmStridedBatched(hblas, CUBLAS_OP_C, CUBLAS_OP_N, nproj, nslot * npert, npwx, &
            (1.0_dp, 0.0_dp), d_W, npwx, INT(npwx, 8) * nproj, x, npwx, INT(npwx, 8) * nslot * npert, &
            (0.0_dp, 0.0_dp), cf, nproj, INT(nproj, 8) * nslot * npert, nksq)
    CALL chk(istat, 'cublasZgemmStridedBatched (proj)')
  END SUBROUTINE gemm_proj
  !
  SUBROUTINE gemm_back(cf, y, beta)
    ! y(:, slots of k) = W_k cf + beta y, batched over k
    COMPLEX(dp), DEVICE :: cf(nproj, *), y(npwx, *)
    COMPLEX(dp), INTENT(IN) :: beta
    INTEGER :: istat
    istat = cublasZgemmStridedBatched(hblas, CUBLAS_OP_N, CUBLAS_OP_N, npwx, nslot * npert, nproj, &
            (1.0_dp, 0.0_dp), d_W, npwx, INT(npwx, 8) * nproj, cf, nproj, INT(nproj, 8) * nslot * npert, &
            beta, y, npwx, INT(npwx, 8) * nslot * npert, nksq)
    CALL chk(istat, 'cublasZgemmStridedBatched (back)')
  END SUBROUTINE gemm_back
  !
  SUBROUTINE debug_step(iter)
    INTEGER, INTENT(IN) :: iter
    REAL(dp) :: r(1), a(1), c(1)
    INTEGER :: cv(1), s
    s = sth_debug_slot
    CALL tsync()
    r = d_rho(s:s); a = d_av(s:s); c = d_cv(s:s); cv = d_conv(s:s)
    IF (cv(1) == 0) WRITE(*, '(A,I4,A,ES22.14,A,ES22.14,A,ES22.14,A,ES22.14)') '   dbg iter', iter, ' rho', r(1), &
         ' a', a(1), ' c', c(1), ' lam', -a(1) / c(1)
  END SUBROUTINE debug_step
  !
  SUBROUTINE ortho_exact()
    INTEGER :: ik, ip, s1, istat
    DO ik = 1, nksq
      DO ip = 1, npert
        s1 = slot(1, ip, ik)
        istat = cublasZgemm(hblas, CUBLAS_OP_C, CUBLAS_OP_N, nbnd, h_nock(ik), h_npwq(ik), (1.0_dp, 0.0_dp), &
                d_W(1, nkb + 1, ik), npwx, d_d0(1, s1), npwx, (0.0_dp, 0.0_dp), d_ps(1, s1), nbnd)
        CALL chk(istat, 'cublasZgemm (ortho ps)')
      ENDDO
    ENDDO
    CALL k_scale_wwg<<<nsl, 32 * ((nbnd + 31) / 32), 0, strm>>>(d_ps, d_wwg, nbnd, nslot, npert, nksq, d_nock)
    CALL kerr('scale_wwg')
    CALL k_scale_wg1<<<4096, TPB, 0, strm>>>(d_d0, d_wg1, npwx, nslot, npert, nksq, d_nock, &
         INT(npwx, 8) * nsl)
    CALL kerr('scale_wg1')
    DO ik = 1, nksq
      DO ip = 1, npert
        s1 = slot(1, ip, ik)
        istat = cublasZgemm(hblas, CUBLAS_OP_N, CUBLAS_OP_N, h_npwq(ik), h_nock(ik), nbnd, (1.0_dp, 0.0_dp), &
                d_W(1, nkb + 1, ik), npwx, d_ps(1, s1), nbnd, (-1.0_dp, 0.0_dp), d_d0(1, s1), npwx)
        CALL chk(istat, 'cublasZgemm (ortho back)')
      ENDDO
    ENDDO
  END SUBROUTINE ortho_exact
  !
  SUBROUTINE ortho_batched()
    ! orthogonalize as ortho_exact, but with two strided-batched ZGEMMs over k on all slots (FP64).
    ! Used with sth_mp, where the exact vanilla GEMM shapes no longer matter: 2 launches instead of
    ! 2 nksq npert (latency-bound on consumer GPUs). Invalid slots and padding rows are zero on input
    ! and stay zero.
    INTEGER :: istat
    istat = cublasZgemmStridedBatched(hblas, CUBLAS_OP_C, CUBLAS_OP_N, nbnd, nslot * npert, npwx, &
            (1.0_dp, 0.0_dp), d_W(1, nkb + 1, 1), npwx, INT(npwx, 8) * nproj, d_d0, npwx, INT(npwx, 8) * nslot * npert, &
            (0.0_dp, 0.0_dp), d_ps, nbnd, INT(nbnd, 8) * nslot * npert, nksq)
    CALL chk(istat, 'cublasZgemmStridedBatched (ortho ps)')
    CALL k_scale_wwg<<<nsl, 32 * ((nbnd + 31) / 32), 0, strm>>>(d_ps, d_wwg, nbnd, nslot, npert, nksq, d_nock)
    CALL kerr('scale_wwg')
    CALL k_scale_wg1<<<4096, TPB, 0, strm>>>(d_d0, d_wg1, npwx, nslot, npert, nksq, d_nock, &
         INT(npwx, 8) * nsl)
    CALL kerr('scale_wg1')
    istat = cublasZgemmStridedBatched(hblas, CUBLAS_OP_N, CUBLAS_OP_N, npwx, nslot * npert, nbnd, &
            (1.0_dp, 0.0_dp), d_W(1, nkb + 1, 1), npwx, INT(npwx, 8) * nproj, d_ps, nbnd, INT(nbnd, 8) * nslot * npert, &
            (-1.0_dp, 0.0_dp), d_d0, npwx, INT(npwx, 8) * nslot * npert, nksq)
    CALL chk(istat, 'cublasZgemmStridedBatched (ortho back)')
  END SUBROUTINE ortho_batched
  !
  SUBROUTINE r0_profile(thr)
    ! histogram of sqrt(rho)/thresh over the columns still active after the first test (rho of
    ! those columns was just written by k_check); converged columns are not counted
    REAL(dp), INTENT(IN) :: thr
    REAL(dp), ALLOCATABLE :: rh(:), v(:)
    INTEGER, ALLOCATABLE :: cv(:), hv(:)
    INTEGER :: i, b, m
    ALLOCATE(rh(nsl), cv(nsl), hv(nvalid))
    rh = d_rho
    cv = d_conv
    hv = d_valid
    sth_r0hist = 0
    ALLOCATE(v(nvalid))
    m = 0
    DO i = 1, nvalid
      IF (cv(hv(i)) /= 0) CYCLE
      m = m + 1
      v(m) = SQRT(MAX(rh(hv(i)), 0.0_dp)) / thr
      b = MIN(11, MAX(0, INT(2.0_dp * LOG10(MAX(v(m), 1.0_dp)))))
      sth_r0hist(b) = sth_r0hist(b) + 1
    ENDDO
    sth_r0med = 0
    IF (m > 0) THEN
      CALL sort_r(v, m)
      sth_r0med = v((m + 1) / 2)
    ENDIF
    DEALLOCATE(rh, cv, hv, v)
  END SUBROUTINE r0_profile
  !
  SUBROUTINE band_profile(thr)
    ! mean CG tests per column by distance of band n from the top occupied band nock(k), the mean
    ! log10(r0/thresh) per bin, and tests and decades by the smallest known gap to the empty bands
    REAL(dp), INTENT(IN) :: thr
    INTEGER, ALLOCATABLE :: ns(:), hv(:)
    REAL(dp), ALLOCATABLE :: r0h(:)
    REAL(dp) :: cnt(0:6), dec, gap
    INTEGER :: i, s, k, n, b, m, g
    ALLOCATE(ns(nsl), hv(nvalid), r0h(nsl))
    ns = d_nst
    hv = d_valid
    r0h = d_r0
    sth_stband = 0; sth_r0band = 0; sth_gapst = 0; sth_gapdec = 0; sth_gapn = 0
    cnt = 0
    DO i = 1, nvalid
      s = hv(i)
      k = (s - 1) / (nslot * npert) + 1
      n = MOD(s - 1, nslot) + 1
      b = MIN(6, h_nock(k) - n)
      dec = LOG10(MAX(r0h(s) / thr, 1.0_dp))
      sth_stband(b) = sth_stband(b) + ns(s)
      sth_r0band(b) = sth_r0band(b) + dec
      cnt(b) = cnt(b) + 1
      gap = 1.0e10_dp
      DO m = h_nockq(k) + 1, nbnd
        gap = MIN(gap, h_etkq(m, k) - h_etk(n, k))
      ENDDO
      IF (gap > 1.0e9_dp) THEN
        g = 5
      ELSEIF (gap < 0.02_dp) THEN
        g = 1
      ELSEIF (gap < 0.1_dp) THEN
        g = 2
      ELSEIF (gap < 0.5_dp) THEN
        g = 3
      ELSE
        g = 4
      ENDIF
      sth_gapst(g) = sth_gapst(g) + ns(s)
      sth_gapdec(g) = sth_gapdec(g) + dec
      sth_gapn(g) = sth_gapn(g) + 1
    ENDDO
    DO b = 0, 6
      IF (cnt(b) > 0) sth_stband(b) = sth_stband(b) / cnt(b)
      IF (cnt(b) > 0) sth_r0band(b) = sth_r0band(b) / cnt(b)
    ENDDO
    DO g = 1, 5
      IF (sth_gapn(g) > 0) sth_gapst(g) = sth_gapst(g) / sth_gapn(g)
      IF (sth_gapn(g) > 0) sth_gapdec(g) = sth_gapdec(g) / sth_gapn(g)
    ENDDO
    DEALLOCATE(ns, hv, r0h)
  END SUBROUTINE band_profile
  !
  SUBROUTINE sort_r(a, n)
    ! in-place heapsort (ascending)
    INTEGER, INTENT(IN) :: n
    REAL(dp), INTENT(INOUT) :: a(n)
    INTEGER :: i, j, l, ir
    REAL(dp) :: t
    IF (n < 2) RETURN
    l = n / 2 + 1
    ir = n
    DO
      IF (l > 1) THEN
        l = l - 1
        t = a(l)
      ELSE
        t = a(ir)
        a(ir) = a(1)
        ir = ir - 1
        IF (ir == 1) THEN
          a(1) = t
          RETURN
        ENDIF
      ENDIF
      i = l
      j = l + l
      DO WHILE (j <= ir)
        IF (j < ir) THEN
          IF (a(j) < a(j + 1)) j = j + 1
        ENDIF
        IF (t < a(j)) THEN
          a(i) = a(j)
          i = j
          j = j + j
        ELSE
          j = ir + 1
        ENDIF
      ENDDO
      a(i) = t
    ENDDO
  END SUBROUTINE sort_r
  !
  SUBROUTINE dpot_fdx(d, s0, nb, ip, scale)
    ! RTSC_STH_FDXDP: d(:, list(s0+1 .. s0+nb)) += scale gather_kq(FFT(psi_kn(r) d_dv(r, p))) on the pruned path
    COMPLEX(dp), DEVICE :: d(npwx, *)
    INTEGER, INTENT(IN) :: s0, nb, ip
    REAL(dp), INTENT(IN) :: scale
    INTEGER :: istat
    IF (s0 < psir_nv) THEN
      istat = fdx_dpot_d(d, d_psir, d_dv, d_valid, s0, nb, nslot, npert, npwx, nnr, 0, scale, strm)
    ELSE
      CALL psir_chunk(s0, nb, ip)
      istat = fdx_dpot_d(d, d_box2, d_dv, d_valid, s0, nb, nslot, npert, npwx, nnr, 1, scale, strm)
    ENDIF
    IF (istat /= 0) CALL errore_stop('fdx_dpot_d', istat)
  END SUBROUTINE dpot_fdx
  !
  SUBROUTINE dp_compare(tgt, base, zbase, scale, what)
    ! diagnostic (RTSC_STH_FDXDP_CHECK=1): tgt = base + (pruned dpot of d_dv); rebuild base + (box dpot) in d_t and
    ! print max|diff| / max|box dpot| and the relative L2 over all slots. zbase: base is zero.
    COMPLEX(dp), DEVICE :: tgt(npwx, *), base(npwx, *)
    LOGICAL, INTENT(IN) :: zbase
    REAL(dp), INTENT(IN) :: scale
    CHARACTER(LEN=*), INTENT(IN) :: what
    COMPLEX(dp), ALLOCATABLE :: a(:,:), b(:,:), c(:,:)
    INTEGER :: s0, nb, ip
    REAL(dp) :: dmax, ymax, d2, y2
    IF (zbase) THEN
      CALL zero_c(d_t, INT(npwx, 8) * nsl)
    ELSE
      CALL k_copy<<<4096, TPB, 0, strm>>>(d_t, base, INT(npwx, 8) * nsl)
    ENDIF
    s0 = 0
    DO WHILE (s0 < nvalid)
      nb = MIN(sth_bmax, nvalid - s0)
      ip = plan_for(nb)
      nb = 2**ip
      IF (psir_cached) THEN
        CALL k_dvpsir<<<dim3(nblk(nnr, TPB), nb, 1), TPB, 0, strm>>>(d_box, d_psir, d_dv, d_valid, s0, &
             nnr, nslot, npert, 0)
      ELSE
        CALL psir_chunk(s0, nb, ip)
        CALL k_dvpsir<<<dim3(nblk(nnr, TPB), nb, 1), TPB, 0, strm>>>(d_box, d_box2, d_dv, d_valid, s0, &
             nnr, nslot, npert, 1)
      ENDIF
      CALL fft_exec(ip, d_box, CUFFT_FORWARD)
      CALL k_gather_add<<<dim3(nblk(npwx, TPB), nb, 1), TPB, 0, strm>>>(d_t, d_box, d_valid, s0, nnr, &
           npwx, nslot, npert, nksq, d_npwq, d_idxkq, scale)
      s0 = s0 + nb
    ENDDO
    CALL kerr('dp_compare')
    CALL tsync()
    ALLOCATE(a(npwx, nsl), b(npwx, nsl), c(npwx, nsl))
    a = tgt(:, 1:nsl)
    b = d_t(:, 1:nsl)
    IF (zbase) THEN
      c = (0.0_dp, 0.0_dp)
    ELSE
      c = base(:, 1:nsl)
    ENDIF
    dmax = MAXVAL(ABS(a - b)); ymax = MAXVAL(ABS(b - c))
    d2 = SUM(ABS(a - b)**2); y2 = SUM(ABS(b - c)**2)
    IF (ymax > 0.0_dp) WRITE(*, '(A,A,A,I8,A,ES10.3,A,ES10.3)') '     rtsc fdxdp check (', what, '): columns', nvalid, &
         ', max|d|/max ', dmax / ymax, ', rel L2 ', SQRT(d2 / y2)
    DEALLOCATE(a, b, c)
    CALL zero_c(d_t, INT(npwx, 8) * nsl)
  END SUBROUTINE dp_compare
  !
  SUBROUTINE dr_compare(drho0)
    ! diagnostic (RTSC_STH_FDXDP_CHECK=1): d_drho = drho0 + (pruned drho); rebuild drho0 + (box drho) and compare
    COMPLEX(dp), INTENT(IN) :: drho0(nnr, npert)
    COMPLEX(dp), DEVICE, ALLOCATABLE :: d_r2(:,:)
    COMPLEX(dp), ALLOCATABLE :: a(:,:), b(:,:)
    INTEGER :: s0, nb, ip
    REAL(dp) :: dmax, ymax, d2, y2
    ALLOCATE(d_r2(nnr, npert), a(nnr, npert), b(nnr, npert))
    d_r2 = drho0
    s0 = 0
    DO WHILE (s0 < nvalid)
      nb = MIN(sth_bmax, nvalid - s0)
      ip = plan_for(nb)
      nb = 2**ip
      CALL k_scatter_full<<<dim3(nblk(nnr, TPB), nb, 1), TPB, 0, strm>>>(d_box, d_dpsi, d_valid, s0, nnr, &
           npwx, nslot, npert, nksq, d_inv)
      CALL fft_exec(ip, d_box, CUFFT_INVERSE)
      IF (psir_cached) THEN
        CALL k_drho<<<nblk(nnr, TPB), TPB, 0, strm>>>(d_r2, d_box, d_psir, d_valid, s0, nb, nnr, nslot, &
             npert, d_wgt, 0)
      ELSE
        CALL psir_chunk(s0, nb, ip)
        CALL k_drho<<<nblk(nnr, TPB), TPB, 0, strm>>>(d_r2, d_box, d_box2, d_valid, s0, nb, nnr, nslot, &
             npert, d_wgt, 1)
      ENDIF
      s0 = s0 + nb
    ENDDO
    CALL kerr('dr_compare')
    CALL tsync()
    a = d_drho
    b = d_r2
    dmax = MAXVAL(ABS(a - b)); ymax = MAXVAL(ABS(b - drho0))
    d2 = SUM(ABS(a - b)**2); y2 = SUM(ABS(b - drho0)**2)
    IF (ymax > 0.0_dp) WRITE(*, '(A,I8,A,ES10.3,A,ES10.3)') '     rtsc fdxdp check (drho): columns', nvalid, &
         ', max|d|/max ', dmax / ymax, ', rel L2 ', SQRT(d2 / y2)
    DEALLOCATE(d_r2, a, b)
  END SUBROUTINE dr_compare
  !
  SUBROUTINE lift_free()
    IF (ALLOCATED(d_lq)) DEALLOCATE(d_lq)
    IF (ALLOCATED(h_lq)) DEALLOCATE(h_lq)
    IF (ALLOCATED(d_llam)) DEALLOCATE(d_llam, d_lt)
    IF (ALLOCATED(d_lg)) DEALLOCATE(d_lg, d_lz, d_lc)
    IF (ALLOCATED(d_lok)) DEALLOCATE(d_lok)
    lift_q = .FALSE.
    lift_stream = .FALSE.
  END SUBROUTINE lift_free
  !
  SUBROUTINE lift_build()
    !! LIFT-F factor for every k+q of this q (host build from the staged h_W, h_g2, h_idxkq; see the header).
    !! Runs inside sth_engine_setup, after make_plans and before h_W is released. On any problem it prints why
    !! and leaves lift_q = .FALSE. (the TPA path is used).
    INTEGER :: ik, nq, nl, nh, i, j, ig, jg, nocc, info, lwork, istat, a1, a2, a3, b1, b2, b3, kc
    INTEGER :: nlmin, nlmax, ncap
    INTEGER(8) :: freeb, totb, qbytes, work, avail
    REAL(dp) :: t0, en, gn, r, lmin, rmin, rmax
    REAL(dp), ALLOCATABLE :: lam(:), rwork(:), rr(:)
    COMPLEX(dp), ALLOCATABLE :: vg(:), acol(:,:), x(:,:), yh(:,:), s0(:,:), b0(:,:), z(:,:), cw(:), pl(:,:), &
                                psil(:,:), psih(:,:), qk(:,:)
    INTEGER, ALLOCATABLE :: lidx(:), hidx(:), m3(:,:), nlk(:)
    REAL(dp), ALLOCATABLE :: hlam(:,:), hlt(:,:)
    INTEGER, ALLOCATABLE :: hlok(:,:)
    INTEGER :: nr_rr, nlok
    LOGICAL :: w_staged
    EXTERNAL :: zgemm, zhegv
    t0 = wtime()
    CALL lift_free()
    sth_lift_ncol = 0; sth_lift_nclamp = 0; sth_lift_nout = 0; sth_lift_nfb = 0
    ! QCACHE stages W straight into d_W (no host h_W): copy it back to the host for the build
    w_staged = .FALSE.
    IF (.NOT. ALLOCATED(h_W)) THEN
      ALLOCATE(h_W(npwx, nproj, nksq))
      CALL tsync()
      h_W = d_W
      w_staged = .TRUE.
    ENDIF
    IF (ANY(nrx /= nr)) THEN
      WRITE(*, '(A)') '     rtsc lift: padded FFT box (nrx /= nr) not supported; TPA kept'
      IF (w_staged) DEALLOCATE(h_W)
      RETURN
    ENDIF
    ! V(G) = (1/N) sum_r vrs(r) e^{-iGr} on the box (CUFFT_FORWARD), exactly the matrix elements of apply_op
    CALL k_r2c<<<4096, TPB, 0, strm>>>(d_box, d_vrs, INT(nnr, 8))
    CALL kerr('r2c (lift)')
    CALL fft_exec(0, d_box, CUFFT_FORWARD)
    ALLOCATE(vg(nnr))
    CALL tsync()
    vg = d_box(:, 1)
    vg = vg / REAL(nr(1) * nr(2) * nr(3), dp)
    CALL zero_c(d_box, INT(nnr, 8))
    lift_v0 = REAL(vg(1), dp)
    ! coarse sets
    ALLOCATE(nlk(nksq))
    ncap = 0
    DO ik = 1, nksq
      nlk(ik) = COUNT(h_g2(1:h_npwq(ik), ik) < sth_lift)
      IF (nlk(ik) > sth_lift_nmax) THEN
        nlk(ik) = sth_lift_nmax
        ncap = ncap + 1
      ENDIF
    ENDDO
    lift_nm = MAXVAL(nlk)
    nlmin = MINVAL(nlk); nlmax = lift_nm
    IF (nlmin < 1) THEN
      WRITE(*, '(A)') '     rtsc lift: empty coarse set at some k+q; TPA kept'
      DEALLOCATE(vg, nlk)
      IF (w_staged) DEALLOCATE(h_W)
      RETURN
    ENDIF
    ! storage by k chunks
    istat = cudaMemGetInfo(freeb, totb)
    qbytes = 8_8 * npwx * lift_nm * nksq
    work = (16_8 * npwx + 8_8 * lift_nm) * nslot * MAXPERT * nksq + 16_8 * npwx * nslot * MAXPERT * nksq * 8
    avail = freeb - 3_8 * 1024_8**3 / 2 - work
    kc = sth_lift_kchunk
    IF (kc <= 0) THEN
      IF (qbytes <= avail) THEN
        kc = nksq
      ELSE
        kc = INT(MAX(1_8, avail / (8_8 * npwx * lift_nm)))
      ENDIF
    ENDIF
    lift_kc = MIN(kc, nksq)
    lift_stream = lift_kc < nksq
    IF (lift_stream) THEN
      ALLOCATE(h_lq(npwx, lift_nm, nksq), d_lq(npwx, lift_nm, lift_kc))
    ELSE
      ALLOCATE(h_lq(npwx, lift_nm, nksq), d_lq(npwx, lift_nm, nksq))
    ENDIF
    h_lq = (0.0_sp, 0.0_sp)
    ALLOCATE(hlam(lift_nm, nksq), hlt(npwx, nksq), rr(nslot * nksq), hlok(nslot, nksq))
    hlok = 1
    hlam = 1.0e30_dp
    hlt = 0.0_dp
    rr = 0.0_dp
    nr_rr = 0
    lmin = HUGE(1.0_dp)
    ! Miller triplets of the box points
    ALLOCATE(m3(3, nnr))
    DO i = 1, nnr
      m3(1, i) = MOD(i - 1, nr(1))
      m3(2, i) = MOD((i - 1) / nr(1), nr(2))
      m3(3, i) = (i - 1) / (nr(1) * nr(2))
    ENDDO
    DO ik = 1, nksq
      nq = h_npwq(ik)
      nl = nlk(ik)
      nh = nq - nl
      nocc = h_nockq(ik)
      ALLOCATE(lidx(nl), hidx(MAX(nh, 1)))
      j = 0; i = 0
      DO ig = 1, nq
        IF (h_g2(ig, ik) < sth_lift .AND. j < nl) THEN
          j = j + 1
          lidx(j) = ig
        ELSE
          i = i + 1
          hidx(i) = ig
          hlt(ig, ik) = h_g2(ig, ik)
        ENDIF
      ENDDO
      ! A~(:, L): local V(G - G') + kinetic (- V0 on the diagonal) + nonlocal + alpha P_v
      ALLOCATE(acol(nq, nl))
      DO j = 1, nl
        jg = lidx(j)
        b1 = m3(1, h_idxkq(jg, ik)); b2 = m3(2, h_idxkq(jg, ik)); b3 = m3(3, h_idxkq(jg, ik))
        DO ig = 1, nq
          a1 = MODULO(m3(1, h_idxkq(ig, ik)) - b1, nr(1))
          a2 = MODULO(m3(2, h_idxkq(ig, ik)) - b2, nr(2))
          a3 = MODULO(m3(3, h_idxkq(ig, ik)) - b3, nr(3))
          acol(ig, j) = vg(1 + a1 + nr(1) * (a2 + nr(2) * a3))
        ENDDO
        acol(jg, j) = acol(jg, j) + h_g2(jg, ik) - lift_v0
      ENDDO
      IF (nkb > 0) THEN
        ALLOCATE(pl(nl, nkb), x(nkb, nl))
        DO j = 1, nl
          pl(j, :) = h_W(lidx(j), 1:nkb, ik)
        ENDDO
        BLOCK
          COMPLEX(dp), ALLOCATABLE :: dm(:,:)
          ALLOCATE(dm(nkb, nkb))
          dm = d_dmat
          CALL zgemm('N', 'C', nkb, nl, nkb, (1.0_dp, 0.0_dp), dm, nkb, pl, nl, (0.0_dp, 0.0_dp), x, nkb)
          DEALLOCATE(dm)
        END BLOCK
        CALL zgemm('N', 'N', nq, nl, nkb, (1.0_dp, 0.0_dp), h_W(1, 1, ik), npwx, x, nkb, (1.0_dp, 0.0_dp), acol, nq)
        DEALLOCATE(pl, x)
      ENDIF
      ALLOCATE(psil(nl, MAX(nocc, 1)), psih(MAX(nh, 1), MAX(nocc, 1)))
      DO j = 1, nl
        psil(j, 1:nocc) = h_W(lidx(j), nkb + 1:nkb + nocc, ik)
      ENDDO
      DO i = 1, nh
        psih(i, 1:nocc) = h_W(hidx(i), nkb + 1:nkb + nocc, ik)
      ENDDO
      IF (nocc > 0) CALL zgemm('N', 'C', nq, nl, nocc, CMPLX(alpha_pv, 0.0_dp, KIND=dp), h_W(1, nkb + 1, ik), npwx, &
                               psil, nl, (1.0_dp, 0.0_dp), acol, nq)
      ! S0 = A'_LL + A~_HL^H Y_H + alpha Z Z^H ; B0 = I + Y_H^H Y_H
      ALLOCATE(s0(nl, nl), b0(nl, nl), yh(MAX(nh, 1), nl), z(nl, MAX(nocc, 1)), lam(nl))
      DO j = 1, nl
        DO i = 1, nl
          s0(i, j) = acol(lidx(i), j)
        ENDDO
        DO i = 1, nh
          yh(i, j) = -acol(hidx(i), j) / h_g2(hidx(i), ik)
        ENDDO
      ENDDO
      b0 = (0.0_dp, 0.0_dp)
      IF (nh > 0) THEN
        ! acol(H rows) is no longer needed apart from this product: reuse it compactly
        BLOCK
          COMPLEX(dp), ALLOCATABLE :: ahl(:,:)
          ALLOCATE(ahl(nh, nl))
          DO j = 1, nl
            DO i = 1, nh
              ahl(i, j) = acol(hidx(i), j)
            ENDDO
          ENDDO
          CALL zgemm('C', 'N', nl, nl, nh, (1.0_dp, 0.0_dp), ahl, nh, yh, nh, (1.0_dp, 0.0_dp), s0, nl)
          DEALLOCATE(ahl)
        END BLOCK
        IF (nocc > 0) THEN
          CALL zgemm('C', 'N', nl, nocc, nh, (1.0_dp, 0.0_dp), yh, nh, psih, nh, (0.0_dp, 0.0_dp), z, nl)
          CALL zgemm('N', 'C', nl, nl, nocc, CMPLX(alpha_pv, 0.0_dp, KIND=dp), z, nl, z, nl, (1.0_dp, 0.0_dp), s0, nl)
        ENDIF
        CALL zgemm('C', 'N', nl, nl, nh, (1.0_dp, 0.0_dp), yh, nh, yh, nh, (0.0_dp, 0.0_dp), b0, nl)
      ENDIF
      DO i = 1, nl
        b0(i, i) = b0(i, i) + 1.0_dp
      ENDDO
      s0 = 0.5_dp * (s0 + TRANSPOSE(CONJG(s0)))
      b0 = 0.5_dp * (b0 + TRANSPOSE(CONJG(b0)))
      lwork = MAX(1, 4 * nl)
      ALLOCATE(cw(lwork), rwork(MAX(1, 3 * nl - 2)))
      CALL zhegv(1, 'V', 'U', nl, s0, nl, b0, nl, lam, cw, lwork, rwork, info)
      DEALLOCATE(cw, rwork)
      IF (info /= 0) THEN
        WRITE(*, '(A,I6,A,I6,A)') '     rtsc lift: zhegv info ', info, ' at k+q ', ik, '; TPA kept'
        DEALLOCATE(lidx, hidx, acol, psil, psih, s0, b0, yh, z, lam, vg, nlk, m3, hlam, hlt, rr, hlok)
        IF (w_staged) DEALLOCATE(h_W)
        CALL lift_free()
        RETURN
      ENDIF
      ! Q = Y C: rows L = C, rows H = Y_H C
      ALLOCATE(qk(MAX(nh, 1), nl))
      IF (nh > 0) CALL zgemm('N', 'N', nh, nl, nl, (1.0_dp, 0.0_dp), yh, nh, s0, nl, (0.0_dp, 0.0_dp), qk, nh)
      DO j = 1, nl
        DO i = 1, nl
          h_lq(lidx(i), j, ik) = CMPLX(s0(i, j), KIND=sp)
        ENDDO
        DO i = 1, nh
          h_lq(hidx(i), j, ik) = CMPLX(qk(i, j), KIND=sp)
        ENDDO
      ENDDO
      hlam(1:nl, ik) = lam(1:nl)
      lmin = MIN(lmin, lam(1))
      ! diagnostics per valid column (k, n <= nocc_k): clamp; lowest pencil gap against the true lowest gap
      DO i = 1, h_nock(ik)
        en = h_etk(i, ik) - lift_v0
        sth_lift_ncol = sth_lift_ncol + 1
        ! v2 criterion, in the pre-registered order: (a) no empty band, (b) g_n <= 0, (c) clamp, (d) ratio
        IF (nocc >= nbnd) THEN
          hlok(i, ik) = 0
          sth_lift_nfb(1) = sth_lift_nfb(1) + 1
        ELSE
          gn = MIN(h_etkq(nocc + 1, ik), h_etkq(1, ik) + alpha_pv) - h_etk(i, ik)
          IF (gn <= 0.0_dp) THEN
            hlok(i, ik) = 0
            sth_lift_nfb(2) = sth_lift_nfb(2) + 1
          ELSEIF (lam(1) - en < sth_lift_delta) THEN
            hlok(i, ik) = 0
            sth_lift_nfb(3) = sth_lift_nfb(3) + 1
          ELSE
            r = (lam(1) - en) / gn
            IF (ABS(LOG(r)) > LOG(3.0_dp)) THEN
              hlok(i, ik) = 0
              sth_lift_nfb(4) = sth_lift_nfb(4) + 1
            ENDIF
          ENDIF
        ENDIF
        IF (lam(1) - en < sth_lift_delta) THEN
          sth_lift_nclamp = sth_lift_nclamp + 1
          CYCLE
        ENDIF
        IF (nocc < nbnd) THEN
          gn = MIN(h_etkq(nocc + 1, ik), h_etkq(1, ik) + alpha_pv) - h_etk(i, ik)
          IF (gn > 0.0_dp) THEN
            r = (lam(1) - en) / gn
            nr_rr = nr_rr + 1
            rr(nr_rr) = r
            IF (ABS(LOG(r)) > LOG(3.0_dp)) sth_lift_nout = sth_lift_nout + 1
          ENDIF
        ENDIF
      ENDDO
      DEALLOCATE(lidx, hidx, acol, psil, psih, s0, b0, yh, z, lam, qk)
    ENDDO
    ALLOCATE(d_llam(lift_nm, nksq), d_lt(npwx, nksq), d_lok(nslot, nksq))
    d_llam = hlam
    d_lt = hlt
    IF (sth_lift_fb == 0) hlok = 1
    d_lok = hlok
    nlok = SUM(sth_lift_nfb)
    IF (w_staged) DEALLOCATE(h_W)
    IF (.NOT. lift_stream) d_lq = h_lq
    IF (.NOT. lift_stream) DEALLOCATE(h_lq)
    rmin = 0; rmax = 0
    IF (nr_rr > 0) THEN
      CALL sort_r(rr, nr_rr)
      rmin = rr(1); rmax = rr(nr_rr)
    ENDIF
    CALL tsync()
    sth_t_lbuild = wtime() - t0
    lift_q = .TRUE.
    WRITE(*, '(A,F6.2,A,I5,A,I5,A,I3,A,F9.4,A,F9.4,A,F7.2,A)') '     rtsc lift: Ec'' ', sth_lift, ' Ry, N'' ', nlmin, &
         ' ..', nlmax, ' (capped at', ncap, ' k+q), V0 ', lift_v0, ' Ry, min Lambda ', lmin, ' Ry, build ', &
         sth_t_lbuild, ' s'
    WRITE(*, '(A,I8,A,I6,A,I6,A,I6,A,3F8.3,A,I5,A,L1)') '     rtsc lift: columns ', sth_lift_ncol, ', clamped ', &
         sth_lift_nclamp, ', outliers ', sth_lift_nout, ' (of ', nr_rr, ' with a known gap); (Lambda1-e)/gap min/med/max', &
         rmin, rr(MAX(1, (nr_rr + 1) / 2)), rmax, '; k chunk ', lift_kc, ', streamed ', lift_stream
    WRITE(*, '(A,I2,A,I8,A,I8,A,F7.3,A,4I7,A)') '     rtsc lift v2: fallback switch ', sth_lift_fb, ', columns ', &
         sth_lift_ncol, ', unreliable ', nlok, ' (', 100.0 * REAL(nlok) / REAL(MAX(1, sth_lift_ncol)), &
         ' %; a/b/c/d =', sth_lift_nfb, '), TPA on them only if the switch is 1'
    DEALLOCATE(vg, nlk, m3, hlam, hlt, rr, hlok)
  END SUBROUTINE lift_build
  !
  SUBROUTINE lift_apply()
    !! d_lz(:, s) = Q_k diag(1/max(Lambda - e~_n, delta)) Q_k^H d_g(:, s) for every slot (FP32 batched GEMMs per k chunk)
    INTEGER :: c0, nk, istat, ncolk
    INTEGER(8) :: off
    REAL(dp) :: ta
    IF (sth_timing) THEN
      CALL tsync(); ta = wtime()
    ENDIF
    ncolk = nslot * npert
    IF (ALLOCATED(d_lg)) THEN
      IF (SIZE(d_lg, 2) /= nsl) DEALLOCATE(d_lg, d_lz, d_lc)
    ENDIF
    IF (.NOT. ALLOCATED(d_lg)) ALLOCATE(d_lg(npwx, nsl), d_lz(npwx, nsl), d_lc(lift_nm, nsl))
    CALL k_d2s_c<<<4096, TPB, 0, strm>>>(d_lg, d_g, INT(npwx, 8) * nsl)
    CALL kerr('d2s (lift)')
    DO c0 = 0, nksq - 1, lift_kc
      nk = MIN(lift_kc, nksq - c0)
      off = INT(c0, 8) * ncolk + 1
      IF (lift_stream) THEN
        istat = cudaMemcpyAsync(d_lq, h_lq(:, :, c0 + 1:c0 + nk), INT(npwx, 8) * lift_nm * nk, strm)
        CALL chk(istat, 'cudaMemcpyAsync (lift chunk)')
        istat = cublasCgemmStridedBatched(hblas, CUBLAS_OP_C, CUBLAS_OP_N, lift_nm, ncolk, npwx, (1.0_sp, 0.0_sp), &
             d_lq, npwx, INT(npwx, 8) * lift_nm, d_lg(1, off), npwx, INT(npwx, 8) * ncolk, (0.0_sp, 0.0_sp), &
             d_lc(1, off), lift_nm, INT(lift_nm, 8) * ncolk, nk)
      ELSE
        istat = cublasCgemmStridedBatched(hblas, CUBLAS_OP_C, CUBLAS_OP_N, lift_nm, ncolk, npwx, (1.0_sp, 0.0_sp), &
             d_lq(1, 1, c0 + 1), npwx, INT(npwx, 8) * lift_nm, d_lg(1, off), npwx, INT(npwx, 8) * ncolk, &
             (0.0_sp, 0.0_sp), d_lc(1, off), lift_nm, INT(lift_nm, 8) * ncolk, nk)
      ENDIF
      CALL chk(istat, 'cublasCgemmStridedBatched (lift QHg)')
      CALL k_lift_scale<<<4096, TPB, 0, strm>>>(d_lc, d_llam, d_etk, lift_nm, nbnd, nslot, ncolk, c0, &
           INT(lift_nm, 8) * ncolk * nk, lift_v0, sth_lift_delta)
      CALL kerr('lift_scale')
      IF (lift_stream) THEN
        istat = cublasCgemmStridedBatched(hblas, CUBLAS_OP_N, CUBLAS_OP_N, npwx, ncolk, lift_nm, (1.0_sp, 0.0_sp), &
             d_lq, npwx, INT(npwx, 8) * lift_nm, d_lc(1, off), lift_nm, INT(lift_nm, 8) * ncolk, (0.0_sp, 0.0_sp), &
             d_lz(1, off), npwx, INT(npwx, 8) * ncolk, nk)
      ELSE
        istat = cublasCgemmStridedBatched(hblas, CUBLAS_OP_N, CUBLAS_OP_N, npwx, ncolk, lift_nm, (1.0_sp, 0.0_sp), &
             d_lq(1, 1, c0 + 1), npwx, INT(npwx, 8) * lift_nm, d_lc(1, off), lift_nm, INT(lift_nm, 8) * ncolk, &
             (0.0_sp, 0.0_sp), d_lz(1, off), npwx, INT(npwx, 8) * ncolk, nk)
      ENDIF
      CALL chk(istat, 'cublasCgemmStridedBatched (lift Qc)')
    ENDDO
    IF (sth_timing) THEN
      CALL tsync(); sth_t_lapply = sth_t_lapply + wtime() - ta
    ENDIF
  END SUBROUTINE lift_apply
  !
  INTEGER FUNCTION sth_engine_rwf_count()
    !! KDEV: SYMROT cache entries already on the device
    sth_engine_rwf_count = rwf_n
  END FUNCTION sth_engine_rwf_count
  !
  SUBROUTINE sth_engine_rwf_put(i, ngs, nb, wfc, mill)
    !! KDEV: SYMROT cache entry i (ngs coefficients x nb bands, cache G order) onto the device (entries arrive in order);
    !! v2 also keeps its Miller indices
    INTEGER, INTENT(IN) :: i, ngs, nb
    COMPLEX(dp), INTENT(IN) :: wfc(:,:)
    INTEGER, INTENT(IN), OPTIONAL :: mill(:,:)
    COMPLEX(dp), DEVICE, ALLOCATABLE :: tmp(:,:,:)
    INTEGER, DEVICE, ALLOCATABLE :: itmp(:,:,:)
    INTEGER :: cap
    IF (rwf_ld == 0) THEN
      rwf_ld = SIZE(wfc, 1); rwf_nb = nb
    ENDIF
    IF (ngs > rwf_ld .OR. nb /= rwf_nb .OR. i /= rwf_n + 1) CALL errore_stop('sth_engine_rwf_put (shape or order)', i)
    cap = 0
    IF (ALLOCATED(d_rwf)) cap = SIZE(d_rwf, 3)
    IF (i > cap) THEN
      ALLOCATE(tmp(rwf_ld, rwf_nb, MAX(64, 2 * cap)))
      IF (cap > 0) THEN
        tmp(:, :, 1:cap) = d_rwf(:, :, 1:cap)
        DEALLOCATE(d_rwf)
      ENDIF
      CALL MOVE_ALLOC(tmp, d_rwf)
      ALLOCATE(itmp(3, rwf_ld, SIZE(d_rwf, 3)))
      itmp = 0
      IF (cap > 0) THEN
        itmp(:, :, 1:cap) = d_rmill(:, :, 1:cap)
        DEALLOCATE(d_rmill)
      ENDIF
      CALL MOVE_ALLOC(itmp, d_rmill)
    ENDIF
    d_rwf(1:ngs, 1:nb, i) = wfc(1:ngs, 1:nb)
    IF (PRESENT(mill)) d_rmill(1:3, 1:ngs, i) = mill(1:3, 1:ngs)
    rwf_n = i
  END SUBROUTINE sth_engine_rwf_put
  !
  SUBROUTINE sth_engine_kdev_map(ik, wh, i0, tr, ft, imap, ph)
    !! KDEV, per q: k (wh = 1) or k+q (wh = 2) of column block ik is SYMROT cache entry i0 rotated by (imap, ph, tr);
    !! ft = 1 if the rotation carries a phase. i0 = 0: host path (the caller uploaded evc / evq through set_k).
    INTEGER, INTENT(IN) :: ik, wh, i0, tr, ft, imap(*)
    COMPLEX(dp), INTENT(IN) :: ph(*)
    INTEGER :: n
    IF (ALLOCATED(h_kdrep)) THEN
      IF (SIZE(h_kdrep, 2) /= nksq .OR. SIZE(h_kdmap, 1) /= npwx) DEALLOCATE(h_kdrep, h_kdtr, h_kdft, h_kdmap, h_kdph)
    ENDIF
    IF (.NOT. ALLOCATED(h_kdrep)) THEN
      ALLOCATE(h_kdrep(2, nksq), h_kdtr(2, nksq), h_kdft(2, nksq), h_kdmap(npwx, 2, nksq), h_kdph(npwx, 2, nksq))
      h_kdrep = 0
    ENDIF
    h_kdrep(wh, ik) = i0
    IF (i0 < 1) RETURN
    h_kdtr(wh, ik) = tr; h_kdft(wh, ik) = ft
    IF (wh == 1) THEN
      n = h_npw(ik)
    ELSE
      n = h_npwq(ik)
    ENDIF
    h_kdmap(1:n, wh, ik) = imap(1:n)
    IF (ft == 1) h_kdph(1:n, wh, ik) = ph(1:n)
  END SUBROUTINE sth_engine_kdev_map
  !
  SUBROUTINE sth_engine_kdev_op(ik, wh, i0, tr, sop, g0)
    !! KDEV v2, per q: k (wh = 1) or k+q (wh = 2) of block ik is cache entry i0 under (sop, tr, g0); i0 = 0: host path
    INTEGER, INTENT(IN) :: ik, wh, i0, tr, sop(3, 3), g0(3)
    IF (ALLOCATED(h_kdrep)) THEN
      IF (SIZE(h_kdrep, 2) /= nksq) DEALLOCATE(h_kdrep, h_kdtr, h_kdft, h_kdmap, h_kdph)
    ENDIF
    IF (ALLOCATED(h_kdop)) THEN
      IF (SIZE(h_kdop, 3) /= nksq) DEALLOCATE(h_kdop)
    ENDIF
    IF (.NOT. ALLOCATED(h_kdrep)) THEN
      ALLOCATE(h_kdrep(2, nksq), h_kdtr(2, nksq), h_kdft(2, nksq), h_kdmap(1, 2, 1), h_kdph(1, 2, 1))
      h_kdrep = 0
    ENDIF
    IF (.NOT. ALLOCATED(h_kdop)) ALLOCATE(h_kdop(12, 2, nksq))
    kd_v2 = .TRUE.
    h_kdrep(wh, ik) = i0
    IF (i0 < 1) RETURN
    h_kdtr(wh, ik) = tr; h_kdft(wh, ik) = 0
    h_kdop(1:9, wh, ik) = RESHAPE(sop, (/ 9 /))
    h_kdop(10:12, wh, ik) = g0
  END SUBROUTINE sth_engine_kdev_op
  !
  SUBROUTINE sth_engine_kdev_build(nrot)
    !! KDEV, per q (after the per-k loop, before setup): evc / evq of every mapped k into d_evc / W, one kernel
    INTEGER, INTENT(OUT) :: nrot
    INTEGER, DEVICE, ALLOCATABLE :: d_rep(:,:), d_tr(:,:), d_ft(:,:), d_map(:,:,:)
    COMPLEX(dp), DEVICE, ALLOCATABLE :: d_ph(:,:,:)
    nrot = 0
    IF (.NOT. ALLOCATED(h_kdrep) .OR. .NOT. wdev_q) RETURN
    nrot = COUNT(h_kdrep > 0)
    IF (nrot == 0) RETURN
    IF (kd_v2) THEN
      CALL kdev_build_v2()
      kd_v2 = .FALSE.
      h_kdrep = 0
      RETURN
    ENDIF
    ALLOCATE(d_rep(2, nksq), d_tr(2, nksq), d_ft(2, nksq), d_map(npwx, 2, nksq))
    d_rep = h_kdrep; d_tr = h_kdtr; d_ft = h_kdft; d_map = h_kdmap
    IF (ANY(h_kdft == 1 .AND. h_kdrep > 0)) THEN
      ALLOCATE(d_ph(npwx, 2, nksq))
      d_ph = h_kdph
    ELSE
      ALLOCATE(d_ph(1, 2, 1))
    ENDIF
    CALL k_kd_rot<<<dim3(nblk(npwx, TPB), nbnd, MIN(nksq, 65535)), TPB, 0, strm>>>(d_W, d_evc, d_rwf, d_rep, d_tr, d_ft, &
         d_map, d_ph, d_npwq, d_npw, d_nock, npwx, nproj, nkb, nbnd, nslot, rwf_ld, rwf_nb, nksq)
    CALL kerr('kd_rot')
    CALL tsync()
    DEALLOCATE(d_rep, d_tr, d_ft, d_map, d_ph)
    h_kdrep = 0
  END SUBROUTINE sth_engine_kdev_build
  !
  SUBROUTINE kdev_build_v2()
    !! KDEV v2: per chunk of k, the inverse of each target sphere's FFT index list (box table), then the rotation through
    !! (S, tr, G0) of the cache's Miller indices; k and k+q in two passes
    INTEGER, DEVICE, ALLOCATABLE :: d_rep(:,:), d_tr(:,:), d_op(:,:,:), d_inv2(:,:)
    INTEGER :: kc, k0, nk, wh
    IF (ANY(nrx /= nr)) CALL errore_stop('kdev v2: padded FFT box', 1)
    kc = MAX(1, MIN(nksq, INT(2_8 * 1024_8**3 / (4_8 * nnr))))
    kc = MIN(kc, 512)
    ALLOCATE(d_rep(2, nksq), d_tr(2, nksq), d_op(12, 2, nksq), d_inv2(nnr, kc))
    d_rep = h_kdrep; d_tr = h_kdtr; d_op = h_kdop
    d_inv2 = 0
    DO wh = 1, 2
      DO k0 = 0, nksq - 1, kc
        nk = MIN(kc, nksq - k0)
        CALL k_kd_inv<<<dim3(nblk(npwx, TPB), MIN(nk, 65535), 1), TPB, 0, strm>>>(d_inv2, d_idxk, d_idxkq, d_npw, d_npwq, &
             wh, k0, nk, nnr, npwx)
        CALL kerr('kd_inv')
        CALL k_kd_rot2<<<dim3(nblk(npwx, TPB), nbnd, MIN(nk, 65535)), TPB, 0, strm>>>(d_W, d_evc, d_rwf, d_rmill, d_rep, &
             d_tr, d_op, d_inv2, d_npwq, d_npw, d_nock, wh, k0, nk, npwx, nproj, nkb, nbnd, nslot, rwf_ld, rwf_nb, nnr, &
             nr(1), nr(2), nr(3))
        CALL kerr('kd_rot2')
        CALL k_kd_clr<<<dim3(nblk(npwx, TPB), MIN(nk, 65535), 1), TPB, 0, strm>>>(d_inv2, d_idxk, d_idxkq, d_npw, d_npwq, &
             wh, k0, nk, nnr, npwx)
        CALL kerr('kd_clr')
      ENDDO
    ENDDO
    CALL tsync()
    DEALLOCATE(d_rep, d_tr, d_op, d_inv2)
  END SUBROUTINE kdev_build_v2
  !
  SUBROUTINE sth_engine_set_wdev_n(ik0, n, npwq, vkbb)
    !! KDEV: vkb of k+q blocks ik0 .. ik0+n-1 (vkb_batch's chunk, on the device; npwq(j) rows each) into W, one sync
    INTEGER, INTENT(IN) :: ik0, n, npwq(n)
    COMPLEX(dp), DEVICE, INTENT(IN) :: vkbb(npwx, nkb, *)
    INTEGER :: j, istat
    IF (nkb < 1) RETURN
    istat = cudaDeviceSynchronize()
    DO j = 1, n
      istat = cudaMemcpy2DAsync(d_W(1, 1, ik0 + j - 1), npwx, vkbb(1, 1, j), npwx, npwq(j), nkb, &
                                cudaMemcpyDeviceToDevice, strm)
      CALL chk(istat, 'cudaMemcpy2DAsync (vkb chunk -> W)')
    ENDDO
    istat = cudaDeviceSynchronize()
  END SUBROUTINE sth_engine_set_wdev_n
  !
  SUBROUTINE cbsr_free()
    !! per-q CBSR state (the bases of cb_rep stay resident across q)
    INTEGER :: i
    IF (ALLOCATED(cb_tmp)) THEN
      DO i = 1, SIZE(cb_tmp)
        IF (ALLOCATED(cb_tmp(i)%u)) DEALLOCATE(cb_tmp(i)%u)
        IF (ALLOCATED(cb_tmp(i)%u32)) DEALLOCATE(cb_tmp(i)%u32)
      ENDDO
      DEALLOCATE(cb_tmp)
    ENDIF
    IF (ALLOCATED(h_cbtlam)) DEALLOCATE(h_cbtlam)
    IF (ALLOCATED(d_ckmap)) DEALLOCATE(d_ckmap, d_cktr, d_ckph)
    IF (ALLOCATED(cb_qs)) DEALLOCATE(cb_qs)
    IF (ALLOCATED(cb_gN)) DEALLOCATE(cb_gN, cb_gnb, cb_goff)
    IF (ALLOCATED(d_cgsc)) DEALLOCATE(d_cgsc, d_cgl)
    IF (ALLOCATED(d_cqlam)) DEALLOCATE(d_cqlam)
    IF (ALLOCATED(d_pA)) DEALLOCATE(d_pA, d_pX, d_pY)
    IF (ALLOCATED(d_cx)) DEALLOCATE(d_cx)
    IF (ALLOCATED(d_cy)) DEALLOCATE(d_cy)
    IF (ALLOCATED(d_ca)) DEALLOCATE(d_ca, d_cvg, d_cw, d_cinfo, d_cx1)
    IF (ALLOCATED(d_cwork)) DEALLOCATE(d_cwork)
    IF (ALLOCATED(cb_chb0)) DEALLOCATE(cb_chb0, cb_chb1, cb_chc0, cb_chnc, cb_bN, cb_bnb, cb_boff)
    IF (ALLOCATED(d_pA32)) DEALLOCATE(d_pA32, d_pX32, d_pY32)
    IF (ALLOCATED(d_cx32)) DEALLOCATE(d_cx32, d_cy32)
    IF (ALLOCATED(d_ctmp)) DEALLOCATE(d_ctmp)
    IF (ALLOCATED(d_cchk)) DEALLOCATE(d_cchk)
    cb_lwork = 0
    cb_nqs = 0; cb_npl = -1; cb_ctot = 0; cb_ngrp = 0; cb_ntmp = 0; cb_nch = 0; cb_nbt = 0; cb_cw = 0
    cbsr_q = .FALSE.
  END SUBROUTINE cbsr_free
  !
  SUBROUTINE cbsr_release()
    !! every CBSR basis, including the resident ones (shutdown)
    INTEGER :: i, istat
    CALL cbsr_free()
    DO i = 1, CB_MAXREP
      IF (ALLOCATED(cb_rep(i)%u)) DEALLOCATE(cb_rep(i)%u)
      IF (ALLOCATED(cb_rep(i)%u32)) DEALLOCATE(cb_rep(i)%u32)
    ENDDO
    cb_nres = 0
    IF (ALLOCATED(h_ckrep)) DEALLOCATE(h_ckrep, h_cktr, h_ckmap, h_ckph, h_ckq)
    IF (have_solv) THEN
      istat = cusolverDnDestroy(hsolv)
      have_solv = .FALSE.
    ENDIF
  END SUBROUTINE cbsr_release
  !
  SUBROUTINE cb_map_alloc(ik)
    INTEGER, INTENT(IN) :: ik
    IF (ALLOCATED(h_ckrep)) THEN
      IF (SIZE(h_ckrep) /= nksq .OR. SIZE(h_ckmap, 1) /= npwx) DEALLOCATE(h_ckrep, h_cktr, h_ckmap, h_ckph, h_ckq, h_ckdev, h_ckop)
    ENDIF
    IF (.NOT. ALLOCATED(h_ckrep)) ALLOCATE(h_ckrep(nksq), h_cktr(nksq), h_ckmap(npwx, nksq), h_ckph(npwx, nksq), &
                                           h_ckq(nksq), h_ckdev(nksq), h_ckop(12, nksq))
    IF (ik == 1) THEN
      h_ckrep = -1; h_ckdev = 0
    ENDIF
  END SUBROUTINE cb_map_alloc
  !
  SUBROUTINE sth_engine_cbsr_op(ik, irep, tr, sop, g0)
    !! as sth_engine_cbsr_map, for a rotation without a fractional translation: only the operation (S, tr, G0) is passed;
    !! the G map is built on the device from the SYMROT cache's Miller indices (d_rmill; sth_engine_rwf_put)
    INTEGER, INTENT(IN) :: ik, irep, tr, sop(3, 3), g0(3)
    INTEGER :: n
    CALL cb_map_alloc(ik)
    n = h_npwq(ik)
    IF (irep < 1 .OR. irep > MIN(CB_MAXREP, rwf_n)) THEN
      h_ckrep(ik) = 0; h_cktr(ik) = 1; h_ckdev(ik) = 0
      DO n = 1, h_npwq(ik)
        h_ckmap(n, ik) = n
      ENDDO
      h_ckph(:, ik) = (0.0_dp, 0.0_dp)
      h_ckph(1:h_npwq(ik), ik) = (1.0_dp, 0.0_dp)
      h_ckmap(h_npwq(ik) + 1:npwx, ik) = 1
      RETURN
    ENDIF
    h_ckrep(ik) = irep; h_cktr(ik) = tr; h_ckdev(ik) = 1
    h_ckop(1:9, ik) = RESHAPE(sop, (/ 9 /)); h_ckop(10:12, ik) = g0
    h_ckmap(:, ik) = 1
    h_ckph(:, ik) = (0.0_dp, 0.0_dp)
    h_ckph(1:n, ik) = (1.0_dp, 0.0_dp)
  END SUBROUTINE sth_engine_cbsr_op
  !
  SUBROUTINE sth_engine_cbsr_map(ik, irep, ngs, tr, imap, ph)
    !! (after sth_engine_init, before sth_engine_setup) k+q of column block ik is the image of RTSC_NSCF_SYMROT cache
    !! entry irep: evq_ik(imap(ig)) = ph(ig) psi_irep(ig) (tr = 1) or conj(ph(ig) psi_irep(ig)) (tr = -1), ig = 1..ngs.
    !! irep = 0: no map; k+q gets its own per-q basis.
    INTEGER, INTENT(IN) :: ik, irep, ngs, tr, imap(*)
    COMPLEX(dp), INTENT(IN) :: ph(*)
    INTEGER :: ig
    CALL cb_map_alloc(ik)
    h_ckdev(ik) = 0
    IF (irep > 0 .AND. irep <= CB_MAXREP .AND. ngs == h_npwq(ik)) THEN
      h_ckrep(ik) = irep
      h_cktr(ik) = tr
      h_ckmap(1:ngs, ik) = imap(1:ngs)
      h_ckph(1:ngs, ik) = ph(1:ngs)
    ELSE
      h_ckrep(ik) = 0
      h_cktr(ik) = 1
      DO ig = 1, h_npwq(ik)
        h_ckmap(ig, ik) = ig
        h_ckph(ig, ik) = (1.0_dp, 0.0_dp)
      ENDDO
    ENDIF
    IF (h_npwq(ik) < npwx) THEN
      h_ckmap(h_npwq(ik) + 1:npwx, ik) = 1
      h_ckph(h_npwq(ik) + 1:npwx, ik) = (0.0_dp, 0.0_dp)
    ENDIF
  END SUBROUTINE sth_engine_cbsr_map
  !
  INTEGER FUNCTION cb_bisect(lam, n, x)
    !! index of the element of the ascending lam(1:n) nearest to x
    INTEGER, INTENT(IN) :: n
    REAL(dp), INTENT(IN) :: lam(n), x
    INTEGER :: lo, hi, mid
    lo = 1; hi = n
    DO WHILE (hi - lo > 1)
      mid = (lo + hi) / 2
      IF (lam(mid) <= x) THEN
        lo = mid
      ELSE
        hi = mid
      ENDIF
    ENDDO
    cb_bisect = lo
    IF (ABS(lam(hi) - x) < ABS(lam(lo) - x)) cb_bisect = hi
  END FUNCTION cb_bisect
  !
  SUBROUTINE cb_build_one(ik, sl, hlam, map, ph, tr)
    !! the basis of member k+q ik: dense A~ - V0 on the device, zheevd (cuSOLVER), eigenvectors into the basis' G order
    !! (RTSC_STH_CBSR_RF: tier 0 = the lowest cb_m0 in FP64, the rest in FP32)
    INTEGER, INTENT(IN) :: ik, tr
    TYPE(cb_slot_t), INTENT(INOUT) :: sl
    COMPLEX(dp), DEVICE :: ph(*)
    INTEGER, DEVICE :: map(*)
    REAL(dp), INTENT(OUT) :: hlam(cb_n0)
    INTEGER :: nq, nocc, istat, info(1), lw
    REAL(dp) :: te
    nq = h_npwq(ik)
    nocc = h_nockq(ik)
    CALL k_cb_asm<<<dim3(nblk(cb_n0, TPB), MIN(cb_n0, 65535), 1), TPB, 0, strm>>>(d_ca, d_cvg, d_idxkq(1, ik), &
         d_g2(1, ik), nq, cb_n0, nr(1), nr(2), nr(3), cbsr_v0)
    CALL kerr('cb_asm')
    IF (nkb > 0) THEN
      ! A += vkb D vkb^H
      istat = cublasZgemm(hblas, CUBLAS_OP_N, CUBLAS_OP_C, nkb, nq, nkb, (1.0_dp, 0.0_dp), d_dmat, nkb, &
                          d_W(1, 1, ik), npwx, (0.0_dp, 0.0_dp), d_cx1, nkb)
      CALL chk(istat, 'cublasZgemm (cbsr D W^H)')
      istat = cublasZgemm(hblas, CUBLAS_OP_N, CUBLAS_OP_N, nq, nq, nkb, (1.0_dp, 0.0_dp), d_W(1, 1, ik), npwx, &
                          d_cx1, nkb, (1.0_dp, 0.0_dp), d_ca, cb_n0)
      CALL chk(istat, 'cublasZgemm (cbsr W D W^H)')
    ENDIF
    IF (nocc > 0) THEN
      ! A += alpha_pv evq evq^H (occupied bands of k+q, the engine's own evq)
      istat = cublasZgemm(hblas, CUBLAS_OP_N, CUBLAS_OP_C, nq, nq, nocc, CMPLX(alpha_pv, 0.0_dp, KIND=dp), &
                          d_W(1, nkb + 1, ik), npwx, d_W(1, nkb + 1, ik), npwx, (1.0_dp, 0.0_dp), d_ca, cb_n0)
      CALL chk(istat, 'cublasZgemm (cbsr alpha P_v)')
    ENDIF
    CALL k_cb_herm<<<dim3(nblk(nq, TPB), MIN(nq, 65535), 1), TPB, 0, strm>>>(d_ca, nq, cb_n0)
    CALL kerr('cb_herm')
    IF (cb_lwork == 0) THEN
      istat = cusolverDnZheevd_bufferSize(hsolv, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, cb_n0, d_ca, cb_n0, &
                                          d_cw, lw)
      CALL chk(istat, 'cusolverDnZheevd_bufferSize')
      cb_lwork = lw
      ALLOCATE(d_cwork(cb_lwork))
    ENDIF
    CALL tsync(); te = wtime()
    istat = cusolverDnZheevd(hsolv, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, nq, d_ca, cb_n0, d_cw, &
                             d_cwork, cb_lwork, d_cinfo(1))
    CALL chk(istat, 'cusolverDnZheevd')
    info = d_cinfo
    IF (info(1) /= 0) CALL errore_stop('cusolverDnZheevd (cbsr)', info(1))
    CALL tsync(); sth_t_ceig = sth_t_ceig + wtime() - te
    hlam = 1.0e30_dp
    hlam(1:nq) = d_cw(1:nq)
    IF (cb_chk) CALL cb_check_evq(ik, nq, nocc, hlam)
    IF (cbsr_rf) THEN
      ALLOCATE(sl%u(cb_n0, cb_m0), sl%u32(cb_n0, cb_n0 - cb_m0))
      CALL k_cb_torep<<<dim3(nblk(cb_n0, TPB), MIN(cb_n0, 65535), 1), TPB, 0, strm>>>(d_ctmp, d_ca, map, ph, tr, nq, cb_n0)
      CALL kerr('cb_torep (rf)')
      CALL k_cb_split<<<dim3(nblk(cb_n0, TPB), MIN(cb_n0, 65535), 1), TPB, 0, strm>>>(sl%u, sl%u32, d_ctmp, cb_n0, cb_m0)
      CALL kerr('cb_split')
      ! tier-1 window: Lambda_{m0+1} against the highest occupied e~_n of the q
      pc_win = MIN(pc_win, hlam(cb_m0 + 1) - (MAXVAL(h_etk(1:nslot, 1:nksq)) - cbsr_v0))
    ELSE
      ALLOCATE(sl%u(cb_n0, cb_n0))
      CALL k_cb_torep<<<dim3(nblk(cb_n0, TPB), MIN(cb_n0, 65535), 1), TPB, 0, strm>>>(sl%u, d_ca, map, ph, tr, nq, cb_n0)
      CALL kerr('cb_torep')
    ENDIF
  END SUBROUTINE cb_build_one
  !
  SUBROUTINE cb_maps_dev()
    !! CBSR: the G maps of the k+q passed as operations (sth_engine_cbsr_op), built on the device through the inverse of
    !! each k+q sphere's FFT index list (the same integers symrot_kmap computes on the host)
    INTEGER, DEVICE, ALLOCATABLE :: d_inv2(:,:), d_dv(:), d_op(:,:), d_rp(:)
    INTEGER :: kc, k0, nk
    kc = MAX(1, MIN(nksq, INT(2_8 * 1024_8**3 / (4_8 * nnr))))
    kc = MIN(kc, 512)
    ALLOCATE(d_inv2(nnr, kc), d_dv(nksq), d_op(12, nksq), d_rp(nksq))
    d_inv2 = 0
    d_dv = h_ckdev; d_op = h_ckop; d_rp = h_ckrep
    DO k0 = 0, nksq - 1, kc
      nk = MIN(kc, nksq - k0)
      CALL k_kd_inv<<<dim3(nblk(npwx, TPB), MIN(nk, 65535), 1), TPB, 0, strm>>>(d_inv2, d_idxk, d_idxkq, d_npw, d_npwq, &
           2, k0, nk, nnr, npwx)
      CALL kerr('kd_inv (cbsr)')
      CALL k_cb_mapdev<<<dim3(nblk(npwx, TPB), MIN(nk, 65535), 1), TPB, 0, strm>>>(d_ckmap, d_rmill, d_rp, d_cktr, d_op, &
           d_dv, d_inv2, d_npwq, k0, nk, npwx, rwf_ld, nnr, nr(1), nr(2), nr(3))
      CALL kerr('cb_mapdev')
      CALL k_kd_clr<<<dim3(nblk(npwx, TPB), MIN(nk, 65535), 1), TPB, 0, strm>>>(d_inv2, d_idxk, d_idxkq, d_npw, d_npwq, &
           2, k0, nk, nnr, npwx)
      CALL kerr('kd_clr (cbsr)')
    ENDDO
    CALL tsync()
    DEALLOCATE(d_inv2, d_dv, d_op, d_rp)
  END SUBROUTINE cb_maps_dev
  !
  SUBROUTINE cbsr_build()
    !! per q: the basis of every k+q (resident rep bases reused; missing ones built), the assembly check, the device maps
    INTEGER :: ik, i, r, s, m, nq, nnew, ntmp, istat, ndone, jk
    INTEGER(8) :: freeb, totb, sbytes, reserve
    REAL(dp) :: t0, de, demo, deme, gmin, eexp, rgib
    INTEGER, ALLOCATABLE :: first(:), tmpk(:)
    LOGICAL, ALLOCATABLE :: need(:)
    REAL(dp), ALLOCATABLE :: lam1(:)
    CHARACTER(LEN=32) :: sv
    t0 = wtime()
    sth_t_ceig = 0.0_dp
    CALL cbsr_free()
    IF (sth_cbsr_skip) THEN
      WRITE(*, '(A)') '     rtsc cbsr: off at q = Gamma (RTSC_STH_CBSR_GAMMA unset); CG kept'
      RETURN
    ENDIF
    IF (ANY(nrx /= nr)) THEN
      WRITE(*, '(A)') '     rtsc cbsr: padded FFT box (nrx /= nr) not supported; CG kept'
      RETURN
    ENDIF
    IF (.NOT. ALLOCATED(h_ckrep)) THEN
      WRITE(*, '(A)') '     rtsc cbsr: no k+q map (sth_engine_cbsr_map not called); CG kept'
      RETURN
    ENDIF
    IF (SIZE(h_ckrep) /= nksq .OR. ANY(h_ckrep < 0)) THEN
      WRITE(*, '(A)') '     rtsc cbsr: incomplete k+q map; CG kept'
      RETURN
    ENDIF
    IF (cb_n0 == 0) cb_n0 = npwx
    IF (npwx /= cb_n0) THEN
      WRITE(*, '(A,I6,A,I6,A)') '     rtsc cbsr: npwx ', npwx, ' /= the resident bases'' ', cb_n0, '; CG kept at this q'
      RETURN
    ENDIF
    IF (.NOT. have_solv) THEN
      istat = cusolverDnCreate(hsolv)
      CALL chk(istat, 'cusolverDnCreate')
      istat = cusolverDnSetStream(hsolv, strm)
      CALL chk(istat, 'cusolverDnSetStream')
      have_solv = .TRUE.
    ENDIF
    IF (.NOT. ALLOCATED(h_cblam)) THEN
      ALLOCATE(h_cblam(cb_n0, CB_MAXREP))
      h_cblam = 1.0e30_dp
    ENDIF
    IF (.NOT. cbsr_rf_read) THEN
      ! rtsc b2r: refinement form and the pre-check prints (off by default)
      cbsr_rf_read = .TRUE.
      CALL get_environment_variable('RTSC_STH_CBSR_RF', sv, STATUS=istat)
      cbsr_rf = istat == 0 .AND. TRIM(sv) == '1'
      CALL get_environment_variable('RTSC_STH_CBSR_CHK', sv, STATUS=istat)
      cb_chk = istat == 0 .AND. TRIM(sv) == '1'
      CALL get_environment_variable('RTSC_STH_CBSR_TF32', sv, STATUS=istat)
      cb_tf32 = istat == 0 .AND. TRIM(sv) == '1'
      CALL get_environment_variable('RTSC_STH_CBSR_ACT', sv, STATUS=istat)
      cb_act = .NOT. (istat == 0 .AND. TRIM(sv) == '0')
      CALL get_environment_variable('RTSC_STH_CBSR_M0', sv, STATUS=istat)
      IF (istat == 0 .AND. LEN_TRIM(sv) > 0) READ(sv, *, IOSTAT=istat) cb_m0
      CALL get_environment_variable('RTSC_STH_CBSR_CMAX', sv, STATUS=istat)
      IF (istat == 0 .AND. LEN_TRIM(sv) > 0) READ(sv, *, IOSTAT=istat) cb_cmax
      CALL get_environment_variable('RTSC_STH_CBSR_RFMAX', sv, STATUS=istat)
      IF (istat == 0 .AND. LEN_TRIM(sv) > 0) READ(sv, *, IOSTAT=istat) cb_rfmax
      cb_m0 = MAX(8, MIN(cb_m0, cb_n0 - 8))
      cb_cmax = MAX(64, cb_cmax)
      cb_rfmax = MAX(1, MIN(cb_rfmax, 8))
      IF (cbsr_rf) WRITE(*, '(A,I5,A,I7,A,I2,A,L2)') '     rtsc cbsr rf: refinement form, tier 0 (FP64) m0 ', cb_m0, &
           ', chunk <= ', cb_cmax, ' columns, <= ', cb_rfmax, ' steps per call, TF32 tensor cores', cb_tf32
    ENDIF
    cb_qcount = cb_qcount + 1
    !
    ! which bases this q needs: resident reps, reps to build (first member), per-q bases
    ALLOCATE(first(CB_MAXREP), need(CB_MAXREP), tmpk(nksq))
    first = 0; need = .FALSE.
    ntmp = 0
    DO ik = 1, nksq
      r = h_ckrep(ik)
      IF (r > 0) THEN
        need(r) = .TRUE.
        IF (first(r) == 0) first(r) = ik
      ELSE
        ntmp = ntmp + 1
        tmpk(ntmp) = ik
      ENDIF
    ENDDO
    nnew = 0
    DO r = 1, CB_MAXREP
      IF (need(r) .AND. .NOT. ALLOCATED(cb_rep(r)%u)) nnew = nnew + 1
    ENDDO
    ! memory: the new bases plus a reserve for the irrep arrays (8 column arrays) and RTSC_STH_CBSR_RESERVE GiB (default 2)
    sbytes = 16_8 * cb_n0 * cb_n0
    IF (cbsr_rf) sbytes = 16_8 * cb_n0 * cb_m0 + 8_8 * cb_n0 * (cb_n0 - cb_m0)
    rgib = 2.0_dp
    CALL get_environment_variable('RTSC_STH_CBSR_RESERVE', sv, STATUS=istat)
    IF (istat == 0 .AND. LEN_TRIM(sv) > 0) READ(sv, *, IOSTAT=istat) rgib
    reserve = INT(rgib * 1024.0_dp**3, 8) + 8_8 * 16_8 * npwx * nslot * npert * nksq
    ! rf: the chunk buffers (2 FP64 + 2 FP32 arrays of cb_cmax columns) and the build's FP64 staging copy
    IF (cbsr_rf) reserve = reserve + 48_8 * cb_n0 * cb_cmax + 16_8 * cb_n0 * cb_n0
    CALL sth_mem_trim()
    istat = cudaMemGetInfo(freeb, totb)
    IF (sbytes * (nnew + ntmp) + reserve > freeb) THEN
      ! evict resident bases this q does not use, least recently used first
      DO WHILE (sbytes * (nnew + ntmp) + reserve > freeb)
        m = 0
        DO r = 1, CB_MAXREP
          IF (ALLOCATED(cb_rep(r)%u) .AND. .NOT. need(r)) THEN
            IF (m == 0) THEN
              m = r
            ELSE IF (cb_lastq(r) < cb_lastq(m)) THEN
              m = r
            ENDIF
          ENDIF
        ENDDO
        IF (m == 0) EXIT
        DEALLOCATE(cb_rep(m)%u)
        IF (ALLOCATED(cb_rep(m)%u32)) DEALLOCATE(cb_rep(m)%u32)
        cb_nres = cb_nres - 1
        CALL sth_mem_trim()
        istat = cudaMemGetInfo(freeb, totb)
      ENDDO
      IF (sbytes * (nnew + ntmp) + reserve > freeb) THEN
        WRITE(*, '(A,I6,A,F8.2,A,F8.2,A)') '     rtsc cbsr: ', nnew + ntmp, ' new bases need ', &
             REAL(sbytes * (nnew + ntmp), dp) / 1024.0_dp**3, ' GiB, free after the reserve ', &
             REAL(freeb - reserve, dp) / 1024.0_dp**3, ' GiB: CG kept at this q'
        DEALLOCATE(first, need, tmpk)
        RETURN
      ENDIF
    ENDIF
    !
    ! build workspace, V(G) (exactly as the canary: the matrix elements of apply_op's local part), maps on the device
    ALLOCATE(d_ca(cb_n0, cb_n0), d_cvg(nnr), d_cw(cb_n0), d_cinfo(1), d_cx1(MAX(nkb, 1), cb_n0))
    IF (cbsr_rf .AND. nnew + ntmp > 0) ALLOCATE(d_ctmp(cb_n0, cb_n0))
    IF (cb_chk) ALLOCATE(d_cchk(cb_n0, nbnd))
    CALL k_r2c<<<4096, TPB, 0, strm>>>(d_box, d_vrs, INT(nnr, 8))
    CALL kerr('r2c (cbsr)')
    CALL fft_exec(0, d_box, CUFFT_FORWARD)
    istat = cudaMemcpyAsync(d_cvg, d_box(:, 1), nnr, strm)
    CALL chk(istat, 'cudaMemcpyAsync (cbsr vg)')
    CALL k_cb_scal<<<4096, TPB, 0, strm>>>(d_cvg, 1.0_dp / REAL(nr(1) * nr(2) * nr(3), dp), INT(nnr, 8))
    CALL kerr('scale (cbsr vg)')
    CALL zero_c(d_box, INT(nnr, 8))
    BLOCK
      COMPLEX(dp) :: v1(1)
      CALL tsync()
      v1 = d_cvg(1:1)
      cbsr_v0 = REAL(v1(1), dp)
    END BLOCK
    ALLOCATE(d_ckmap(npwx, nksq), d_cktr(nksq), d_ckph(npwx, nksq))
    d_ckmap = h_ckmap; d_cktr = h_cktr; d_ckph = h_ckph
    IF (ANY(h_ckdev == 1)) CALL cb_maps_dev()
    !
    ! per-q basis list: the reps this q uses, then the per-q bases
    cb_nqs = COUNT(need) + ntmp
    ALLOCATE(cb_qs(cb_nqs))
    s = 0
    DO r = 1, CB_MAXREP
      IF (need(r)) THEN
        s = s + 1
        cb_qs(s) = r
        cb_lastq(r) = cb_qcount
        DO ik = 1, nksq
          IF (h_ckrep(ik) == r) h_ckq(ik) = s
        ENDDO
      ENDIF
    ENDDO
    cb_ntmp = ntmp
    IF (ntmp > 0) THEN
      ALLOCATE(cb_tmp(ntmp), h_cbtlam(cb_n0, ntmp))
      DO i = 1, ntmp
        s = s + 1
        cb_qs(s) = -i
        h_ckq(tmpk(i)) = s
      ENDDO
    ENDIF
    ! build the missing bases
    ndone = 0
    DO s = 1, cb_nqs
      r = cb_qs(s)
      IF (r > 0) THEN
        IF (ALLOCATED(cb_rep(r)%u)) CYCLE
        jk = first(r)
        CALL cb_build_one(jk, cb_rep(r), h_cblam(:, r), d_ckmap(1, jk), d_ckph(1, jk), h_cktr(jk))
        cb_nres = cb_nres + 1
        cb_nbuilt = cb_nbuilt + 1
      ELSE
        i = -r
        jk = tmpk(i)
        CALL cb_build_one(jk, cb_tmp(i), h_cbtlam(:, i), d_ckmap(1, jk), d_ckph(1, jk), h_cktr(jk))
      ENDIF
      ndone = ndone + 1
    ENDDO
    ! Lambda of every basis of this q on the device; assembly check for EVERY k+q (the eigenvalues of its basis against
    ! its own nscf eigenvalues e_m + alpha [m <= nocc_kq], m = 1..nbnd: occupied and empty bands reported apart)
    ALLOCATE(d_cqlam(cb_n0, cb_nqs), lam1(cb_nqs))
    BLOCK
      REAL(dp), ALLOCATABLE :: hl(:,:)
      ALLOCATE(hl(cb_n0, cb_nqs))
      DO s = 1, cb_nqs
        IF (cb_qs(s) > 0) THEN
          hl(:, s) = h_cblam(:, cb_qs(s))
        ELSE
          hl(:, s) = h_cbtlam(:, -cb_qs(s))
        ENDIF
        lam1(s) = hl(1, s)
      ENDDO
      d_cqlam = hl
      demo = 0.0_dp; deme = 0.0_dp; gmin = HUGE(1.0_dp)
      DO ik = 1, nksq
        s = h_ckq(ik)
        nq = h_npwq(ik)
        DO m = 1, nbnd
          eexp = h_etkq(m, ik) - cbsr_v0
          IF (m <= h_nockq(ik)) eexp = eexp + alpha_pv
          de = ABS(hl(cb_bisect(hl(1:nq, s), nq, eexp), s) - eexp)
          IF (m <= h_nockq(ik)) THEN
            demo = MAX(demo, de)
          ELSE
            deme = MAX(deme, de)
          ENDIF
        ENDDO
        DO i = 1, h_nock(ik)
          gmin = MIN(gmin, lam1(s) - (h_etk(i, ik) - cbsr_v0))
        ENDDO
      ENDDO
      IF (cb_chk) CALL cb_denoms(hl)
      DEALLOCATE(hl)
    END BLOCK
    DEALLOCATE(d_ca, d_cvg, d_cw, d_cinfo, d_cx1)
    IF (ALLOCATED(d_cwork)) DEALLOCATE(d_cwork)
    IF (ALLOCATED(d_ctmp)) DEALLOCATE(d_ctmp)
    IF (ALLOCATED(d_cchk)) DEALLOCATE(d_cchk)
    cb_lwork = 0
    DEALLOCATE(first, need, tmpk, lam1)
    CALL tsync()
    CALL sth_mem_sample()
    sth_t_cbuild = wtime() - t0
    cbsr_q = .TRUE.
    cb_npl = -1
    WRITE(*, '(A,I6,A,I6,A,I5,A,I5,A,I5,A,I5,A,ES10.3,A,ES10.3,A,F9.4,A,F8.2,A,F8.2,A,F7.2,A)') &
         '     rtsc cbsr: k+q ', nksq, ', npwx ', cb_n0, ', bases ', cb_nqs, ' (reps ', cb_nqs - ntmp, ', built ', ndone - ntmp, &
         ', per-q ', ntmp, '), assembly check max|dE| occ ', demo, ' empty ', deme, ' Ry, min(Lambda1 - e~_n) ', gmin, &
         ' Ry, build ', sth_t_cbuild, ' s (zheevd ', sth_t_ceig, ' s), resident ', REAL(sbytes * cb_nres, dp) / 1024.0_dp**3, ' GiB'
    IF (cbsr_rf) WRITE(*, '(A,I5,A,F9.4,A)') '     rtsc cbsr rf: tier 0 = lowest ', cb_m0, &
         ' states in FP64; min over the bases so far of Lambda(m0+1) - max e~_n = ', pc_win, ' Ry (tier-1 window)'
    IF (cb_chk) THEN
      WRITE(*, '(A,ES10.3,A,ES10.3,A,8I8)') '     rtsc precheck evq residual |(A~ - t_m) evq_m| (cumulative): max occ ', &
           pc_resmax_o, ' empty ', pc_resmax_e, ' Ry; decades <1e-11,1e-11..,..,1e-6..,>=1e-5:', pc_res
      WRITE(*, '(A,7I9)') '     rtsc precheck denominators (columns k,n<=nock; cumulative), decades <1e-5,1e-5..,..,>=1 Ry: unocc', pc_dun
      WRITE(*, '(A,7I9)') '     rtsc precheck denominators occ (e_m(k+q) + alpha_pv - e_n(k), m <= nock(k+q))          :', pc_doc
      WRITE(*, '(A,7I9)') '     rtsc precheck denominators min_j |Lambda_j - e~_n| over the full basis spectrum        :', pc_dla
    ENDIF
  END SUBROUTINE cbsr_build
  !
  SUBROUTINE cb_check_evq(ik, nq, nocc, hlam)
    !! pre-check (M6-A referee E2): residual of the engine's evq in the exact eigenbasis of A~ (d_ca, member G order):
    !! c = U^H evq_m, |(A~ - V0 - t_m) evq_m|^2 = sum_j |c_jm|^2 (Lambda_j - t_m)^2, t_m = e_m - V0 + alpha_pv [m <= nocc]
    INTEGER, INTENT(IN) :: ik, nq, nocc
    REAL(dp), INTENT(IN) :: hlam(cb_n0)
    COMPLEX(dp), ALLOCATABLE :: c(:,:)
    INTEGER :: istat, m, j, b
    REAL(dp) :: t, r
    istat = cublasZgemm(hblas, CUBLAS_OP_C, CUBLAS_OP_N, nq, nbnd, nq, (1.0_dp, 0.0_dp), d_ca, cb_n0, &
                        d_W(1, nkb + 1, ik), npwx, (0.0_dp, 0.0_dp), d_cchk, cb_n0)
    CALL chk(istat, 'cublasZgemm (cbsr evq check)')
    ALLOCATE(c(cb_n0, nbnd))
    c = d_cchk
    DO m = 1, nbnd
      t = h_etkq(m, ik) - cbsr_v0
      IF (m <= nocc) t = t + alpha_pv
      r = 0.0_dp
      DO j = 1, nq
        r = r + (REAL(c(j, m), dp)**2 + AIMAG(c(j, m))**2) * (hlam(j) - t)**2
      ENDDO
      r = SQRT(r)
      IF (m <= nocc) THEN
        pc_resmax_o = MAX(pc_resmax_o, r)
      ELSE
        pc_resmax_e = MAX(pc_resmax_e, r)
      ENDIF
      b = MIN(7, MAX(0, FLOOR(LOG10(MAX(r, 1.0e-30_dp))) + 12))
      pc_res(b) = pc_res(b) + 1
    ENDDO
    DEALLOCATE(c)
  END SUBROUTINE cb_check_evq
  !
  SUBROUTINE cb_denoms(hl)
    !! pre-check (M6-A referees: near-E_F denominators, both channels), per column (k, n <= nock(k)) of this q:
    !! unocc: min_{nock(k+q) < m <= nbnd} |e_m(k+q) - e_n(k)|; occ: min_{m <= nock(k+q)} (e_m(k+q) + alpha_pv - e_n(k));
    !! full: min_j |Lambda_j - e~_n| over the whole spectrum of the column's basis
    REAL(dp), INTENT(IN) :: hl(cb_n0, *)
    INTEGER :: ik, n, m, s
    REAL(dp) :: du, dc, dl, e
    DO ik = 1, nksq
      s = h_ckq(ik)
      DO n = 1, h_nock(ik)
        du = HUGE(1.0_dp); dc = HUGE(1.0_dp)
        DO m = 1, nbnd
          IF (m <= h_nockq(ik)) THEN
            dc = MIN(dc, h_etkq(m, ik) + alpha_pv - h_etk(n, ik))
          ELSE
            du = MIN(du, ABS(h_etkq(m, ik) - h_etk(n, ik)))
          ENDIF
        ENDDO
        e = h_etk(n, ik) - cbsr_v0
        dl = ABS(hl(cb_bisect(hl(1:h_npwq(ik), s), h_npwq(ik), e), s) - e)
        pc_dun(cb_dbin(du)) = pc_dun(cb_dbin(du)) + 1
        pc_doc(cb_dbin(dc)) = pc_doc(cb_dbin(dc)) + 1
        pc_dla(cb_dbin(dl)) = pc_dla(cb_dbin(dl)) + 1
      ENDDO
    ENDDO
  END SUBROUTINE cb_denoms
  !
  INTEGER FUNCTION cb_dbin(d)
    REAL(dp), INTENT(IN) :: d
    cb_dbin = MIN(6, MAX(0, FLOOR(LOG10(MAX(ABS(d), 1.0e-30_dp))) + 6))
  END FUNCTION cb_dbin
  !
  SUBROUTINE cbsr_plan()
    !! per irrep (npert): gathered columns grouped by basis (padded to 16), bases of equal width in one batched GEMM
    INTEGER :: s, ik, ip, n, c, g, i, nb, w, istat
    INTEGER, ALLOCATABLE :: ncol(:), wid(:), ord(:), co(:), hsc(:), hgl(:), cnt(:)
    TYPE(c_devptr), ALLOCATABLE :: hA(:), hX(:), hY(:)
    INTEGER :: ncolk
    ncolk = nslot * npert
    ALLOCATE(ncol(cb_nqs), wid(cb_nqs), ord(cb_nqs), co(cb_nqs))
    ncol = 0
    DO ik = 1, nksq
      s = h_ckq(ik)
      ncol(s) = ncol(s) + npert * h_nock(ik)
    ENDDO
    ! width of each basis' column block: padded to a multiple of RTSC_STH_CBSR_PAD (default 8) so that bases of equal
    ! width share one batched GEMM (zero padding columns: no effect on the valid ones)
    IF (cb_pad == 0) THEN
      cb_pad = 8
      BLOCK
        CHARACTER(LEN=16) :: sv
        INTEGER :: ios, v
        CALL get_environment_variable('RTSC_STH_CBSR_PAD', sv, STATUS=ios)
        IF (ios == 0 .AND. LEN_TRIM(sv) > 0) THEN
          READ(sv, *, IOSTAT=ios) v
          IF (ios == 0 .AND. v >= 1) cb_pad = v
        ENDIF
      END BLOCK
    ENDIF
    wid = cb_pad * ((ncol + cb_pad - 1) / cb_pad)
    ! order bases by width (counting sort over the distinct widths)
    w = MAXVAL(wid)
    ALLOCATE(cnt(0:w / cb_pad))
    cnt = 0
    DO s = 1, cb_nqs
      cnt(wid(s) / cb_pad) = cnt(wid(s) / cb_pad) + 1
    ENDDO
    cb_ngrp = COUNT(cnt(1:) > 0)
    IF (ALLOCATED(cb_gN)) DEALLOCATE(cb_gN, cb_gnb, cb_goff)
    ALLOCATE(cb_gN(cb_ngrp), cb_gnb(cb_ngrp), cb_goff(cb_ngrp))
    g = 0; i = 0
    DO w = 1, UBOUND(cnt, 1)
      IF (cnt(w) == 0) CYCLE
      g = g + 1
      cb_gN(g) = cb_pad * w; cb_gnb(g) = cnt(w); cb_goff(g) = i
      DO s = 1, cb_nqs
        IF (wid(s) == cb_pad * w) THEN
          i = i + 1
          ord(i) = s
        ENDIF
      ENDDO
    ENDDO
    nb = i     ! bases with at least one column
    cb_ctot = 0
    DO i = 1, nb
      co(ord(i)) = cb_ctot
      cb_ctot = cb_ctot + wid(ord(i))
    ENDDO
    ALLOCATE(hsc(cb_ctot), hgl(cb_ctot))
    hsc = 0
    DO i = 1, nb
      hgl(co(ord(i)) + 1:co(ord(i)) + wid(ord(i))) = ord(i)
    ENDDO
    ! columns: for each basis, its member k+q in k order, then pert, then band
    ncol = 0
    DO ik = 1, nksq
      s = h_ckq(ik)
      DO ip = 1, npert
        DO n = 1, h_nock(ik)
          ncol(s) = ncol(s) + 1
          hsc(co(s) + ncol(s)) = n + nslot * ((ip - 1) + npert * (ik - 1))
        ENDDO
      ENDDO
    ENDDO
    IF (ALLOCATED(d_cgsc)) DEALLOCATE(d_cgsc, d_cgl)
    ALLOCATE(d_cgsc(cb_ctot), d_cgl(cb_ctot))
    d_cgsc = hsc; d_cgl = hgl
    IF (ALLOCATED(d_cx)) THEN
      IF (SIZE(d_cx, 2) < cb_ctot) DEALLOCATE(d_cx, d_cy)
    ENDIF
    IF (.NOT. ALLOCATED(d_cx)) ALLOCATE(d_cx(cb_n0, cb_ctot), d_cy(cb_n0, cb_ctot))
    IF (ALLOCATED(d_pA)) DEALLOCATE(d_pA, d_pX, d_pY)
    ALLOCATE(d_pA(nb), d_pX(nb), d_pY(nb), hA(nb), hX(nb), hY(nb))
    DO i = 1, nb
      s = ord(i)
      IF (cb_qs(s) > 0) THEN
        hA(i) = c_devloc(cb_rep(cb_qs(s))%u)
      ELSE
        hA(i) = c_devloc(cb_tmp(-cb_qs(s))%u)
      ENDIF
      hX(i) = c_devloc(d_cx(1, co(s) + 1))
      hY(i) = c_devloc(d_cy(1, co(s) + 1))
    ENDDO
    d_pA = hA; d_pX = hX; d_pY = hY
    IF (sth_timing) THEN
      ! batch shapes (for the H100 projection): width x number of bases, per group
      WRITE(*, '(A,I3,A,I8,A,I8,A)', ADVANCE='NO') '     rtsc cbsr plan: npert ', npert, ', columns ', nvalid, ', gathered ', &
           cb_ctot, ', groups (width x bases):'
      DO g = 1, cb_ngrp
        WRITE(*, '(1X,I0,A,I0)', ADVANCE='NO') cb_gN(g), 'x', cb_gnb(g)
      ENDDO
      WRITE(*, *)
    ENDIF
    DEALLOCATE(ncol, wid, ord, co, hsc, hgl, cnt, hA, hX, hY)
    cb_npl = npert
    istat = 0
  END SUBROUTINE cbsr_plan
  !
  SUBROUTINE cbsr_solve()
    !! d_dpsi(:, s) = U diag(1/(Lambda - e~_n)) U^H d_d0(:, s) on every valid slot, batched per shared basis
    INTEGER :: g, istat, ncolk, ny
    REAL(dp) :: ta
    CALL tsync(); ta = wtime()
    IF (cb_npl /= npert) CALL cbsr_plan()
    ncolk = nslot * npert
    ny = MIN(cb_ctot, 65535)
    CALL k_cb_gather<<<dim3(nblk(cb_n0, TPB), ny, 1), TPB, 0, strm>>>(d_cx, d_d0, d_cgsc, d_ckmap, d_ckph, d_cktr, &
         d_npwq, cb_n0, npwx, ncolk, cb_ctot)
    CALL kerr('cb_gather')
    DO g = 1, cb_ngrp
      istat = cublasZgemmBatched(hblas, CUBLAS_OP_C, CUBLAS_OP_N, cb_n0, cb_gN(g), cb_n0, (1.0_dp, 0.0_dp), &
           d_pA(cb_goff(g) + 1:), cb_n0, d_pX(cb_goff(g) + 1:), cb_n0, (0.0_dp, 0.0_dp), d_pY(cb_goff(g) + 1:), cb_n0, cb_gnb(g))
      CALL chk(istat, 'cublasZgemmBatched (cbsr U^H b)')
    ENDDO
    CALL k_cb_scale<<<dim3(nblk(cb_n0, TPB), ny, 1), TPB, 0, strm>>>(d_cy, d_cqlam, d_cgl, d_cgsc, d_etk, cbsr_v0, &
         cb_n0, nbnd, nslot, ncolk, cb_ctot)
    CALL kerr('cb_scale')
    DO g = 1, cb_ngrp
      istat = cublasZgemmBatched(hblas, CUBLAS_OP_N, CUBLAS_OP_N, cb_n0, cb_gN(g), cb_n0, (1.0_dp, 0.0_dp), &
           d_pA(cb_goff(g) + 1:), cb_n0, d_pY(cb_goff(g) + 1:), cb_n0, (0.0_dp, 0.0_dp), d_pX(cb_goff(g) + 1:), cb_n0, cb_gnb(g))
      CALL chk(istat, 'cublasZgemmBatched (cbsr U c)')
    ENDDO
    CALL zero_c(d_dpsi, INT(npwx, 8) * nsl)
    CALL k_cb_scatter<<<dim3(nblk(cb_n0, TPB), ny, 1), TPB, 0, strm>>>(d_dpsi, d_cx, d_cgsc, d_ckmap, d_ckph, d_cktr, &
         d_npwq, cb_n0, npwx, ncolk, cb_ctot)
    CALL kerr('cb_scatter')
    CALL tsync()
    sth_t_csolve = wtime() - ta
    IF (sth_timing) WRITE(*, '(A,F9.4,A,I8,A,I8,A,I5,A,I5,A,F8.1)') '     rtsc cbsr solve ', sth_t_csolve, ' s, ', nvalid, &
         ' valid columns, ', cb_ctot, ' gathered, ', cb_nqs, ' bases, ', cb_ngrp, ' GEMM groups, mean width ', &
         REAL(cb_ctot, dp) / MAX(1, cb_nqs)
  END SUBROUTINE cbsr_solve
  !
  SUBROUTINE cbsr_plan_rf()
    !! rf: per irrep (npert), the gathered columns grouped per basis (padded to cb_pad) as in cbsr_plan, cut into chunks of
    !! whole bases (stars) of at most cb_cmax columns; inside a chunk, consecutive bases of equal width are one batched GEMM
    INTEGER :: s, ik, ip, n, i, nb, ch, b, w, cw, ncolk
    INTEGER, ALLOCATABLE :: ncol(:), wid(:), ord(:), co(:), hsc(:), hgl(:), chi(:), cst(:)
    TYPE(c_devptr), ALLOCATABLE :: hA(:), hX(:), hY(:), hA2(:), hX2(:), hY2(:)
    ncolk = nslot * npert
    IF (cb_pad == 0) THEN
      cb_pad = 8
      BLOCK
        CHARACTER(LEN=16) :: sv
        INTEGER :: ios, v
        CALL get_environment_variable('RTSC_STH_CBSR_PAD', sv, STATUS=ios)
        IF (ios == 0 .AND. LEN_TRIM(sv) > 0) THEN
          READ(sv, *, IOSTAT=ios) v
          IF (ios == 0 .AND. v >= 1) cb_pad = v
        ENDIF
      END BLOCK
    ENDIF
    ALLOCATE(ncol(cb_nqs), wid(cb_nqs), ord(cb_nqs), co(cb_nqs), chi(cb_nqs), cst(cb_nqs + 1))
    ncol = 0
    DO ik = 1, nksq
      s = h_ckq(ik)
      ncol(s) = ncol(s) + npert * h_nock(ik)
    ENDDO
    wid = cb_pad * ((ncol + cb_pad - 1) / cb_pad)
    ! bases with columns, ordered by width (stable)
    nb = 0
    DO w = cb_pad, MAXVAL(wid), cb_pad
      DO s = 1, cb_nqs
        IF (wid(s) == w) THEN
          nb = nb + 1
          ord(nb) = s
        ENDIF
      ENDDO
    ENDDO
    ! chunks of whole bases
    cb_nch = 0; cw = 0; cb_cw = 0; cb_ctot = 0
    DO i = 1, nb
      s = ord(i)
      IF (cb_nch == 0 .OR. cw + wid(s) > cb_cmax) THEN
        cb_nch = cb_nch + 1
        cst(cb_nch) = cb_ctot
        cw = 0
      ENDIF
      chi(i) = cb_nch
      co(s) = cb_ctot
      cb_ctot = cb_ctot + wid(s)
      cw = cw + wid(s)
      cb_cw = MAX(cb_cw, cw)
    ENDDO
    cst(cb_nch + 1) = cb_ctot
    ! batches: runs of equal width inside a chunk
    IF (ALLOCATED(cb_chb0)) DEALLOCATE(cb_chb0, cb_chb1, cb_chc0, cb_chnc, cb_bN, cb_bnb, cb_boff)
    ALLOCATE(cb_chb0(cb_nch), cb_chb1(cb_nch), cb_chc0(cb_nch), cb_chnc(cb_nch), cb_bN(nb), cb_bnb(nb), cb_boff(nb))
    cb_nbt = 0
    DO i = 1, nb
      s = ord(i)
      IF (i == 1) THEN
        b = 0
      ELSE IF (chi(i) /= chi(i - 1) .OR. wid(s) /= wid(ord(i - 1))) THEN
        b = 0
      ELSE
        b = 1
      ENDIF
      IF (b == 0) THEN
        cb_nbt = cb_nbt + 1
        cb_bN(cb_nbt) = wid(s); cb_bnb(cb_nbt) = 0; cb_boff(cb_nbt) = i - 1
        IF (i == 1) THEN
          cb_chb0(chi(i)) = cb_nbt
        ELSE IF (chi(i) /= chi(i - 1)) THEN
          cb_chb0(chi(i)) = cb_nbt
        ENDIF
      ENDIF
      cb_bnb(cb_nbt) = cb_bnb(cb_nbt) + 1
      cb_chb1(chi(i)) = cb_nbt
    ENDDO
    DO ch = 1, cb_nch
      cb_chc0(ch) = cst(ch)
      cb_chnc(ch) = cst(ch + 1) - cst(ch)
    ENDDO
    ALLOCATE(hsc(cb_ctot), hgl(cb_ctot))
    hsc = 0
    DO i = 1, nb
      hgl(co(ord(i)) + 1:co(ord(i)) + wid(ord(i))) = ord(i)
    ENDDO
    ncol = 0
    DO ik = 1, nksq
      s = h_ckq(ik)
      DO ip = 1, npert
        DO n = 1, h_nock(ik)
          ncol(s) = ncol(s) + 1
          hsc(co(s) + ncol(s)) = n + nslot * ((ip - 1) + npert * (ik - 1))
        ENDDO
      ENDDO
    ENDDO
    IF (ALLOCATED(d_cgsc)) DEALLOCATE(d_cgsc, d_cgl)
    ALLOCATE(d_cgsc(cb_ctot), d_cgl(cb_ctot))
    d_cgsc = hsc; d_cgl = hgl
    IF (ALLOCATED(d_cx)) THEN
      IF (SIZE(d_cx, 2) < cb_cw) DEALLOCATE(d_cx, d_cy, d_cx32, d_cy32)
    ENDIF
    IF (.NOT. ALLOCATED(d_cx)) ALLOCATE(d_cx(cb_n0, cb_cw), d_cy(cb_m0, cb_cw), d_cx32(cb_n0, cb_cw), &
                                        d_cy32(cb_n0 - cb_m0, cb_cw))
    IF (ALLOCATED(d_pA)) DEALLOCATE(d_pA, d_pX, d_pY)
    IF (ALLOCATED(d_pA32)) DEALLOCATE(d_pA32, d_pX32, d_pY32)
    ALLOCATE(d_pA(nb), d_pX(nb), d_pY(nb), d_pA32(nb), d_pX32(nb), d_pY32(nb))
    ALLOCATE(hA(nb), hX(nb), hY(nb), hA2(nb), hX2(nb), hY2(nb))
    DO i = 1, nb
      s = ord(i)
      IF (cb_qs(s) > 0) THEN
        hA(i) = c_devloc(cb_rep(cb_qs(s))%u)
        hA2(i) = c_devloc(cb_rep(cb_qs(s))%u32)
      ELSE
        hA(i) = c_devloc(cb_tmp(-cb_qs(s))%u)
        hA2(i) = c_devloc(cb_tmp(-cb_qs(s))%u32)
      ENDIF
      w = co(s) - cst(chi(i)) + 1
      hX(i) = c_devloc(d_cx(1, w)); hY(i) = c_devloc(d_cy(1, w))
      hX2(i) = c_devloc(d_cx32(1, w)); hY2(i) = c_devloc(d_cy32(1, w))
    ENDDO
    d_pA = hA; d_pX = hX; d_pY = hY; d_pA32 = hA2; d_pX32 = hX2; d_pY32 = hY2
    IF (sth_timing) WRITE(*, '(A,I3,A,I8,A,I8,A,I5,A,I4,A,I6,A,I5)') '     rtsc cbsr rf plan: npert ', npert, ', columns ', nvalid, &
         ', gathered ', cb_ctot, ', bases ', nb, ', chunks ', cb_nch, ' (max ', cb_cw, ' columns), GEMM batches ', cb_nbt
    DEALLOCATE(ncol, wid, ord, co, hsc, hgl, chi, cst, hA, hX, hY, hA2, hX2, hY2)
    cb_npl = npert
  END SUBROUTINE cbsr_plan_rf
  !
  SUBROUTINE cbsr_apply_rf(gs, dst)
    !! dst -= R~ gs on every planned column, R~ = U0 D0 U0^H (FP64) + U1 D1 U1^H (FP32), chunk by chunk
    COMPLEX(dp), DEVICE :: gs(npwx, *), dst(npwx, *)
    INTEGER :: ch, b, c0, nc, ny, istat, ncolk, n1, o
    REAL(dp) :: ta
    IF (cb_npl /= npert) CALL cbsr_plan_rf()
    CALL tsync(); ta = wtime()
    ncolk = nslot * npert
    n1 = cb_n0 - cb_m0
    IF (cb_tf32) THEN
      istat = cublasSetMathMode(hblas, CUBLAS_TF32_TENSOR_OP_MATH)
      CALL chk(istat, 'cublasSetMathMode (tf32)')
    ENDIF
    DO ch = 1, cb_nch
      c0 = cb_chc0(ch); nc = cb_chnc(ch); ny = MIN(nc, 65535)
      CALL k_cb_gather<<<dim3(nblk(cb_n0, TPB), ny, 1), TPB, 0, strm>>>(d_cx, gs, d_cgsc(c0 + 1), d_ckmap, d_ckph, d_cktr, &
           d_npwq, cb_n0, npwx, ncolk, nc)
      CALL kerr('cb_gather (rf)')
      CALL k_cb_z2c<<<4096, TPB, 0, strm>>>(d_cx32, d_cx, INT(cb_n0, 8) * nc)
      CALL kerr('cb_z2c')
      DO b = cb_chb0(ch), cb_chb1(ch)
        o = cb_boff(b) + 1
        istat = cublasZgemmBatched(hblas, CUBLAS_OP_C, CUBLAS_OP_N, cb_m0, cb_bN(b), cb_n0, (1.0_dp, 0.0_dp), &
             d_pA(o:), cb_n0, d_pX(o:), cb_n0, (0.0_dp, 0.0_dp), d_pY(o:), cb_m0, cb_bnb(b))
        CALL chk(istat, 'cublasZgemmBatched (rf U0^H g)')
        istat = cublasCgemmBatched(hblas, CUBLAS_OP_C, CUBLAS_OP_N, n1, cb_bN(b), cb_n0, (1.0_sp, 0.0_sp), &
             d_pA32(o:), cb_n0, d_pX32(o:), cb_n0, (0.0_sp, 0.0_sp), d_pY32(o:), n1, cb_bnb(b))
        CALL chk(istat, 'cublasCgemmBatched (rf U1^H g)')
      ENDDO
      CALL k_cb_scale_rf<<<dim3(nblk(cb_n0, TPB), ny, 1), TPB, 0, strm>>>(d_cy, d_cy32, d_cqlam, d_cgl(c0 + 1), &
           d_cgsc(c0 + 1), d_etk, cbsr_v0, cb_n0, cb_m0, nbnd, nslot, ncolk, nc)
      CALL kerr('cb_scale_rf')
      DO b = cb_chb0(ch), cb_chb1(ch)
        o = cb_boff(b) + 1
        istat = cublasZgemmBatched(hblas, CUBLAS_OP_N, CUBLAS_OP_N, cb_n0, cb_bN(b), cb_m0, (1.0_dp, 0.0_dp), &
             d_pA(o:), cb_n0, d_pY(o:), cb_m0, (0.0_dp, 0.0_dp), d_pX(o:), cb_n0, cb_bnb(b))
        CALL chk(istat, 'cublasZgemmBatched (rf U0 c)')
        istat = cublasCgemmBatched(hblas, CUBLAS_OP_N, CUBLAS_OP_N, cb_n0, cb_bN(b), n1, (1.0_sp, 0.0_sp), &
             d_pA32(o:), cb_n0, d_pY32(o:), n1, (0.0_sp, 0.0_sp), d_pX32(o:), cb_n0, cb_bnb(b))
        CALL chk(istat, 'cublasCgemmBatched (rf U1 c)')
      ENDDO
      CALL k_cb_addc<<<4096, TPB, 0, strm>>>(d_cx, d_cx32, INT(cb_n0, 8) * nc)
      CALL kerr('cb_addc')
      CALL k_cb_scatter_sub<<<dim3(nblk(cb_n0, TPB), ny, 1), TPB, 0, strm>>>(dst, d_cx, d_cgsc(c0 + 1), d_ckmap, d_ckph, &
           d_cktr, d_npwq, cb_n0, npwx, ncolk, nc)
      CALL kerr('cb_scatter_sub')
    ENDDO
    IF (cb_tf32) THEN
      istat = cublasSetMathMode(hblas, CUBLAS_DEFAULT_MATH)
      CALL chk(istat, 'cublasSetMathMode (default)')
    ENDIF
    rf_gcols = rf_gcols + cb_ctot
    CALL tsync()
    sth_t_crfg = sth_t_crfg + wtime() - ta
  END SUBROUTINE cbsr_apply_rf
  !
  SUBROUTINE cbsr_apply_rf_act(gs, dst, nact, done)
    !! rf, later steps: dst -= R~ gs on the nact active columns only (d_act), grouped per basis into one small plan;
    !! done = .FALSE. (nothing applied) if the gathered width exceeds the chunk buffers
    COMPLEX(dp), DEVICE :: gs(npwx, *), dst(npwx, *)
    INTEGER, INTENT(IN) :: nact
    LOGICAL, INTENT(OUT) :: done
    INTEGER, ALLOCATABLE :: hact(:), cnt(:), wid(:), co(:), ord(:), hsc(:), hgl(:), bN(:), bnb(:), boff(:)
    TYPE(c_devptr), ALLOCATABLE :: hA(:), hX(:), hY(:), hA2(:), hX2(:), hY2(:)
    TYPE(c_devptr), DEVICE, ALLOCATABLE :: pA(:), pX(:), pY(:), pA2(:), pX2(:), pY2(:)
    INTEGER, DEVICE, ALLOCATABLE :: dsc(:), dgl(:)
    INTEGER :: i, sc, k, sb, nb, ctot, w, nbt, b, o, istat, ncolk, n1, ny
    REAL(dp) :: ta
    done = .FALSE.
    CALL tsync(); ta = wtime()
    ncolk = nslot * npert
    n1 = cb_n0 - cb_m0
    ALLOCATE(hact(nact), cnt(cb_nqs), wid(cb_nqs), co(cb_nqs), ord(cb_nqs))
    hact = d_act(1:nact)
    cnt = 0
    DO i = 1, nact
      k = (hact(i) - 1) / ncolk + 1
      cnt(h_ckq(k)) = cnt(h_ckq(k)) + 1
    ENDDO
    wid = cb_pad * ((cnt + cb_pad - 1) / cb_pad)
    nb = 0
    DO w = cb_pad, MAXVAL(wid), cb_pad
      DO sb = 1, cb_nqs
        IF (wid(sb) == w) THEN
          nb = nb + 1
          ord(nb) = sb
        ENDIF
      ENDDO
    ENDDO
    ctot = 0
    DO i = 1, nb
      co(ord(i)) = ctot
      ctot = ctot + wid(ord(i))
    ENDDO
    IF (ctot > SIZE(d_cx, 2) .OR. nb == 0) THEN
      DEALLOCATE(hact, cnt, wid, co, ord)
      RETURN
    ENDIF
    ALLOCATE(hsc(ctot), hgl(ctot), bN(nb), bnb(nb), boff(nb))
    hsc = 0
    DO i = 1, nb
      hgl(co(ord(i)) + 1:co(ord(i)) + wid(ord(i))) = ord(i)
    ENDDO
    cnt = 0
    DO i = 1, nact
      k = (hact(i) - 1) / ncolk + 1
      sb = h_ckq(k)
      cnt(sb) = cnt(sb) + 1
      hsc(co(sb) + cnt(sb)) = hact(i)
    ENDDO
    nbt = 0
    DO i = 1, nb
      IF (i == 1) THEN
        nbt = 1; bN(1) = wid(ord(1)); bnb(1) = 0; boff(1) = 0
      ELSE IF (wid(ord(i)) /= wid(ord(i - 1))) THEN
        nbt = nbt + 1; bN(nbt) = wid(ord(i)); bnb(nbt) = 0; boff(nbt) = i - 1
      ENDIF
      bnb(nbt) = bnb(nbt) + 1
    ENDDO
    ALLOCATE(hA(nb), hX(nb), hY(nb), hA2(nb), hX2(nb), hY2(nb), pA(nb), pX(nb), pY(nb), pA2(nb), pX2(nb), pY2(nb))
    DO i = 1, nb
      sb = ord(i)
      IF (cb_qs(sb) > 0) THEN
        hA(i) = c_devloc(cb_rep(cb_qs(sb))%u); hA2(i) = c_devloc(cb_rep(cb_qs(sb))%u32)
      ELSE
        hA(i) = c_devloc(cb_tmp(-cb_qs(sb))%u); hA2(i) = c_devloc(cb_tmp(-cb_qs(sb))%u32)
      ENDIF
      hX(i) = c_devloc(d_cx(1, co(sb) + 1)); hY(i) = c_devloc(d_cy(1, co(sb) + 1))
      hX2(i) = c_devloc(d_cx32(1, co(sb) + 1)); hY2(i) = c_devloc(d_cy32(1, co(sb) + 1))
    ENDDO
    pA = hA; pX = hX; pY = hY; pA2 = hA2; pX2 = hX2; pY2 = hY2
    ALLOCATE(dsc(ctot), dgl(ctot))
    dsc = hsc; dgl = hgl
    IF (cb_tf32) THEN
      istat = cublasSetMathMode(hblas, CUBLAS_TF32_TENSOR_OP_MATH)
      CALL chk(istat, 'cublasSetMathMode (tf32)')
    ENDIF
    ny = MIN(ctot, 65535)
    CALL k_cb_gather<<<dim3(nblk(cb_n0, TPB), ny, 1), TPB, 0, strm>>>(d_cx, gs, dsc, d_ckmap, d_ckph, d_cktr, &
         d_npwq, cb_n0, npwx, ncolk, ctot)
    CALL kerr('cb_gather (rf act)')
    CALL k_cb_z2c<<<4096, TPB, 0, strm>>>(d_cx32, d_cx, INT(cb_n0, 8) * ctot)
    DO b = 1, nbt
      o = boff(b) + 1
      istat = cublasZgemmBatched(hblas, CUBLAS_OP_C, CUBLAS_OP_N, cb_m0, bN(b), cb_n0, (1.0_dp, 0.0_dp), &
           pA(o:), cb_n0, pX(o:), cb_n0, (0.0_dp, 0.0_dp), pY(o:), cb_m0, bnb(b))
      CALL chk(istat, 'cublasZgemmBatched (rf act U0^H g)')
      istat = cublasCgemmBatched(hblas, CUBLAS_OP_C, CUBLAS_OP_N, n1, bN(b), cb_n0, (1.0_sp, 0.0_sp), &
           pA2(o:), cb_n0, pX2(o:), cb_n0, (0.0_sp, 0.0_sp), pY2(o:), n1, bnb(b))
      CALL chk(istat, 'cublasCgemmBatched (rf act U1^H g)')
    ENDDO
    CALL k_cb_scale_rf<<<dim3(nblk(cb_n0, TPB), ny, 1), TPB, 0, strm>>>(d_cy, d_cy32, d_cqlam, dgl, dsc, d_etk, cbsr_v0, &
         cb_n0, cb_m0, nbnd, nslot, ncolk, ctot)
    CALL kerr('cb_scale_rf (act)')
    DO b = 1, nbt
      o = boff(b) + 1
      istat = cublasZgemmBatched(hblas, CUBLAS_OP_N, CUBLAS_OP_N, cb_n0, bN(b), cb_m0, (1.0_dp, 0.0_dp), &
           pA(o:), cb_n0, pY(o:), cb_m0, (0.0_dp, 0.0_dp), pX(o:), cb_n0, bnb(b))
      CALL chk(istat, 'cublasZgemmBatched (rf act U0 c)')
      istat = cublasCgemmBatched(hblas, CUBLAS_OP_N, CUBLAS_OP_N, cb_n0, bN(b), n1, (1.0_sp, 0.0_sp), &
           pA2(o:), cb_n0, pY2(o:), n1, (0.0_sp, 0.0_sp), pX2(o:), cb_n0, bnb(b))
      CALL chk(istat, 'cublasCgemmBatched (rf act U1 c)')
    ENDDO
    CALL k_cb_addc<<<4096, TPB, 0, strm>>>(d_cx, d_cx32, INT(cb_n0, 8) * ctot)
    CALL k_cb_scatter_sub<<<dim3(nblk(cb_n0, TPB), ny, 1), TPB, 0, strm>>>(dst, d_cx, dsc, d_ckmap, d_ckph, &
         d_cktr, d_npwq, cb_n0, npwx, ncolk, ctot)
    CALL kerr('cb_scatter_sub (act)')
    IF (cb_tf32) THEN
      istat = cublasSetMathMode(hblas, CUBLAS_DEFAULT_MATH)
      CALL chk(istat, 'cublasSetMathMode (default)')
    ENDIF
    CALL tsync()
    rf_gcols = rf_gcols + ctot
    sth_t_crfg = sth_t_crfg + wtime() - ta
    DEALLOCATE(hact, cnt, wid, co, ord, hsc, hgl, bN, bnb, boff, hA, hX, hY, hA2, hX2, hY2, pA, pX, pY, pA2, pX2, pY2, dsc, dgl)
    done = .TRUE.
  END SUBROUTINE cbsr_apply_rf_act
  !
  SUBROUTINE cbsr_rf_solve(first_iter, thresh)
    !! rf: dpsi refined from its previous value under QE's own test on the true FP64 residual, every column certified
    LOGICAL, INTENT(IN) :: first_iter
    REAL(dp), INTENT(IN) :: thresh
    INTEGER :: istat, nact, n0a, st, ist(0:8)
    INTEGER(8) :: gp_call, gp0
    LOGICAL :: act_done
    REAL(dp) :: ta
    CALL tsync(); ta = wtime()
    sth_t_crfg = 0.0_dp
    gp_call = 0; gp0 = rf_gcols
    IF (first_iter) THEN
      CALL zero_c(d_dpsi, INT(npwx, 8) * nsl)
      CALL zero_c(d_h, INT(npwx, 8) * nsl)
    ENDIF
    ! g = A dpsi - b; A dpsi of the current dpsi is kept in d_h (zero at the first call of the irrep)
    CALL k_copy<<<4096, TPB, 0, strm>>>(d_g, d_h, INT(npwx, 8) * nsl)
    CALL k_sub<<<4096, TPB, 0, strm>>>(d_g, d_d0, INT(npwx, 8) * nsl)
    CALL kerr('rf residual')
    d_conv = 0
    d_cnt = 0
    d_kter = 0.0_dp
    CALL k_icopy<<<nblk(nvalid, TPB), TPB, 0, strm>>>(d_act, d_valid, nvalid)
    nact = nvalid
    sth_ncolapp = 0
    ist = 0
    st = 0
    DO
      n0a = nact
      CALL k_check_rf<<<nact, sth_rb, 0, strm>>>(d_act, nslot, npert, npwx, nksq, d_npwq, d_hd, d_g, d_rho, d_conv, d_cnt, &
           thresh, MERGE(1, 0, st == 0), d_r0, d_nst)
      CALL kerr('check_rf')
      CALL k_kter<<<nblk(ngrp, TPB), TPB, 0, strm>>>(ngrp, npert, nksq, d_nock, d_cnt, d_kter)
      CALL kerr('kter (rf)')
      IF (st == 0) THEN
        sth_nact0 = nact
        IF (sth_timing) CALL r0_profile(thresh)
      ENDIF
      CALL k_compact<<<1, 1024, 0, strm>>>(d_act, d_act2, nact, d_conv, d_nact)
      CALL kerr('compact (rf)')
      istat = cudaMemcpyAsync(h_nact, d_nact, 1, strm)
      CALL tsync()
      nact = h_nact(1)
      ist(st) = n0a - nact
      rf_conv(st) = rf_conv(st) + (n0a - nact)
      IF (nact == 0 .OR. st == cb_rfmax) EXIT
      st = st + 1
      ! one step: the correction is computed on every planned column; converged columns have g = 0 and do not move
      CALL k_zero_conv<<<nvalid, TPB, 0, strm>>>(d_g, d_valid, d_conv, npwx, nvalid)
      CALL kerr('zero_conv')
      ! later steps on the few columns left: a per-call plan of the active columns only (RTSC_STH_CBSR_ACT=0: full plan)
      act_done = .FALSE.
      IF (st >= 2 .AND. cb_act .AND. 4 * nact < nvalid) CALL cbsr_apply_rf_act(d_g, d_dpsi, nact, act_done)
      IF (.NOT. act_done) CALL cbsr_apply_rf(d_g, d_dpsi)
      IF (act_done) gp_call = gp_call + rf_gcols - gp0
      IF (.NOT. act_done) gp_call = gp_call + cb_ctot
      gp0 = rf_gcols
      CALL apply_op(d_dpsi, d_t, d_act, nact)
      sth_ncolapp = sth_ncolapp + nact
      CALL k_copy_list<<<nact, TPB, 0, strm>>>(d_h, d_t, d_act, npwx)
      CALL k_sub_list<<<nact, TPB, 0, strm>>>(d_g, d_h, d_d0, d_act, npwx)
      CALL kerr('rf update')
    ENDDO
    IF (nact > 0) rf_conv(8) = rf_conv(8) + nact
    sth_nsteps = st
    sth_t_crf = wtime() - ta
    IF (sth_timing) WRITE(*, '(A,ES9.2,A,I8,A,I2,A,9I8,A,I8,A,F8.3,A,F8.3,A,I9)') '     rtsc cbsr rf: thr ', thresh, ' columns ', &
         nvalid, ' steps ', st, ' converged per step 0..', ist(0:8), ' unconverged ', nact, ' | ', sth_t_crf, ' s (GEMM part ', &
         sth_t_crfg, ' s) gathered GEMM columns', gp_call
  END SUBROUTINE cbsr_rf_solve
  !
  SUBROUTINE apply_op(x, y, list, nl)
    ! y = (H_kq - e_n + alpha_pv P_v) x on the nl columns list(1:nl); other slots of y are garbage
    COMPLEX(dp), DEVICE :: x(npwx, *), y(npwx, *)
    INTEGER, DEVICE :: list(*)
    INTEGER, INTENT(IN) :: nl
    INTEGER :: s0, nb, ip, np
    REAL(dp) :: scale
    REAL(dp) :: ta, tb
    LOGICAL :: use_p, use_f, chk
    scale = 1.0_dp / REAL(nr(1) * nr(2) * nr(3), dp)
    IF (sth_prof) THEN
      CALL tsync(); ta = wtime()
    ENDIF
    CALL gemm_proj(x, d_cf)
    CALL k_coef_h<<<nsl, 32 * ((nproj + 31) / 32), 0, strm>>>(d_cf2, d_cf, d_dmat, nproj, nkb, nslot, &
         npert, nksq, d_nockq, alpha_pv)
    CALL kerr('coef_h')
    IF (sth_prof) THEN
      CALL tsync(); tb = wtime(); sth_t_gemm = sth_t_gemm + tb - ta; ta = tb
    ENDIF
    use_f = fdx_on .AND. sth_opvar == 0
    use_p = pfft_ok() .AND. sth_opvar == 0 .AND. .NOT. use_f
    chk = .NOT. pfft_checked .AND. ((use_p .AND. sth_pfft_check) .OR. (use_f .AND. sth_fftdx_check))
    IF (.NOT. (use_p .OR. use_f) .OR. chk) THEN
      s0 = 0
      DO WHILE (s0 < nl)
        nb = MIN(sth_bmax, nl - s0)
        ip = plan_for(nb)
        nb = 2**ip
        np = 2**ip
        CALL k_scatter_full<<<dim3(nblk(nnr, TPB), nb, 1), TPB, 0, strm>>>(d_box, x, list, s0, nnr, npwx, &
             nslot, npert, nksq, d_inv)
        CALL kerr('scatter')
        CALL fft_exec(ip, d_box, CUFFT_INVERSE)
        CALL k_vmul<<<dim3(nblk(nnr, TPB), nb, 1), TPB, 0, strm>>>(d_box, d_vrs, nnr)
        CALL kerr('vmul')
        CALL fft_exec(ip, d_box, CUFFT_FORWARD)
        CALL k_gather_h<<<dim3(nblk(npwx, TPB), nb, 1), TPB, 0, strm>>>(y, x, d_box, list, s0, nnr, npwx, &
             nslot, npert, nksq, nbnd, d_npwq, d_idxkq, d_g2, d_etk, scale, sth_opvar)
        CALL kerr('gather_h')
        s0 = s0 + nb
      ENDDO
    ENDIF
    IF (chk) CALL pfft_compare(x, y, list, nl, scale, use_f)
    IF (use_f) THEN
      np = fdx_local_d(x, y, list, nl, nslot, npert, npwx, nksq, nbnd, d_vrs, d_npwq, d_g2, d_etk, scale, strm)
      IF (np /= 0) CALL errore_stop('fdx_local_d', np)
    ELSE IF (use_p) THEN
      CALL pfft_local(x, y, list, nl, nslot, npert, npwx, nksq, nbnd, d_vrs, d_npwq, d_g2, d_etk, scale, strm)
    ENDIF
    IF (sth_prof) THEN
      CALL tsync(); tb = wtime(); sth_t_fft = sth_t_fft + tb - ta; ta = tb
    ENDIF
    CALL gemm_back(d_cf2, y, (1.0_dp, 0.0_dp))
    IF (sth_prof) THEN
      CALL tsync(); tb = wtime(); sth_t_gemm = sth_t_gemm + tb - ta
    ENDIF
  END SUBROUTINE apply_op
  !
  SUBROUTINE pfft_compare(x, y, list, nl, scale, use_f)
    ! diagnostic (RTSC_STH_PFFT_CHECK=1): y holds the cuFFT local part on list(1:nl); compute the
    ! pruned one into a scratch array and print the difference over those columns
    COMPLEX(dp), DEVICE :: x(npwx, *), y(npwx, *)
    INTEGER, DEVICE :: list(*)
    INTEGER, INTENT(IN) :: nl
    REAL(dp), INTENT(IN) :: scale
    LOGICAL, INTENT(IN) :: use_f
    COMPLEX(dp), DEVICE, ALLOCATABLE :: yp(:,:)
    COMPLEX(dp), ALLOCATABLE :: a(:,:), b(:,:)
    INTEGER, ALLOCATABLE :: hl(:)
    INTEGER :: i, c
    REAL(dp) :: dmax, ymax, d2, y2
    ALLOCATE(yp(npwx, nsl), a(npwx, nsl), b(npwx, nsl), hl(nl))
    CALL tsync()
    IF (use_f) THEN
      i = fdx_local_d(x, yp, list, nl, nslot, npert, npwx, nksq, nbnd, d_vrs, d_npwq, d_g2, d_etk, scale, strm)
    ELSE
      CALL pfft_local(x, yp, list, nl, nslot, npert, npwx, nksq, nbnd, d_vrs, d_npwq, d_g2, d_etk, scale, strm)
    ENDIF
    CALL tsync()
    a = y(:, 1:nsl)
    b = yp
    hl = list(1:nl)
    dmax = 0; ymax = 0; d2 = 0; y2 = 0
    DO i = 1, nl
      c = hl(i)
      dmax = MAX(dmax, MAXVAL(ABS(a(:, c) - b(:, c)))); ymax = MAX(ymax, MAXVAL(ABS(a(:, c))))
      d2 = d2 + SUM(ABS(a(:, c) - b(:, c))**2); y2 = y2 + SUM(ABS(a(:, c))**2)
    ENDDO
    IF (ymax > 0.0_dp) THEN
      WRITE(*, '(A,A,A,I8,A,ES10.3,A,ES10.3)') '     rtsc ', MERGE('fftdx', 'pfft ', use_f), ' check: columns', nl, &
           ', max|dy|/max|y| ', dmax / ymax, &
           ', rel L2 ', SQRT(d2 / y2)
      pfft_checked = .TRUE.     ! (a zero input, e.g. dpsi = 0 at the first iteration, is not a test)
    ENDIF
    DEALLOCATE(yp, a, b, hl)
  END SUBROUTINE pfft_compare
  !
  !=================================================================================================
  ! public API
  !=================================================================================================
  !
  SUBROUTINE sth_engine_set_npmax(n)
    !! the largest irrep dimension of the coming q (sizes the irrep arrays' reserve of the partial psi(r) cache)
    INTEGER, INTENT(IN) :: n
    sth_npmax = MAX(1, n)
  END SUBROUTINE sth_engine_set_npmax
  !
  LOGICAL FUNCTION sth_engine_qready()
    sth_engine_qready = qready
  END FUNCTION sth_engine_qready
  !
  LOGICAL FUNCTION sth_engine_ready()
    sth_engine_ready = ready
  END FUNCTION sth_engine_ready
  !
  SUBROUTINE sth_engine_init(nksq_, npwx_, nbnd_, nkb_, npert_, nslot_, nr1, nr2, nr3, nr1x, nr2x, &
                             nr3x, nnr_, alpha_pv_, omega_, ierr)
    !! Allocate host staging for one q (and its first irrep, npert_). nslot_ = max over k of nocc_k.
    INTEGER, INTENT(IN) :: nksq_, npwx_, nbnd_, nkb_, npert_, nslot_, nr1, nr2, nr3, nr1x, nr2x, nr3x, nnr_
    REAL(dp), INTENT(IN) :: alpha_pv_, omega_
    INTEGER, INTENT(OUT) :: ierr
    ierr = 0
    CALL sth_engine_free()
    IF (npert_ > MAXPERT .OR. npert_ < 1) THEN
      ierr = 1
      RETURN
    ENDIF
    IF (nnr_ /= nr1x * nr2x * nr3x) THEN
      ierr = 2
      RETURN
    ENDIF
    nksq = nksq_; npwx = npwx_; nbnd = nbnd_; nkb = nkb_; npert = npert_; nslot = nslot_
    nr = (/ nr1, nr2, nr3 /); nrx = (/ nr1x, nr2x, nr3x /); nnr = nnr_
    alpha_pv = alpha_pv_; omega = omega_
    nproj = nkb + nbnd
    nbox = nslot * nksq
    ALLOCATE(h_npw(nksq), h_npwq(nksq), h_nock(nksq), h_nockq(nksq), h_wk(nksq))
    ALLOCATE(h_idxk(npwx, nksq), h_idxkq(npwx, nksq), h_g2(npwx, nksq), h_etk(nbnd, nksq), h_etkq(nbnd, nksq))
    ALLOCATE(h_wwg(nbnd, nslot, nksq), h_wg1(nslot, nksq))
    qc_q = sth_qcache
    wdev_q = .FALSE.
    IF (qc_q) THEN
      ! W and evc straight into device memory (zeroed: rows > npw(q) and bands > nocc stay zero, as h_W / h_evc)
      BLOCK
        INTEGER(8) :: freeb, totb, pre
        INTEGER :: istat
        ! clear a pending (non-sticky) error left by other code, as sth_engine_setup does, so that zero_c's kerr
        ! reports only its own launch (with RTSC_NSCF_SYMROT, nscf_engine, which used to clear it, may not run)
        istat = cudaGetLastError()
        IF (istat /= cudaSuccess) WRITE(*, '(A,A)') 'sth_engine: cleared a pending CUDA error from earlier code: ', &
             TRIM(cudaGetErrorString(istat))
        pre = 16_8 * npwx * nproj * nksq + 16_8 * npwx * nbox
        CALL sth_mem_trim()
        istat = cudaMemGetInfo(freeb, totb)
        IF (pre <= freeb - 2_8 * 1024_8**3) THEN
          ALLOCATE(d_W(npwx, nproj, nksq), d_evc(npwx, nbox))
          CALL zero_c(d_W, INT(npwx, 8) * nproj * nksq)
          CALL zero_c(d_evc, INT(npwx, 8) * nbox)
          istat = cudaDeviceSynchronize()
          wdev_q = .TRUE.
        ENDIF
      END BLOCK
    ENDIF
    IF (.NOT. wdev_q) THEN
      ALLOCATE(h_W(npwx, nproj, nksq), h_evc(npwx, nbox))
      h_W = (0.0_dp, 0.0_dp); h_evc = (0.0_dp, 0.0_dp)
    ENDIF
    h_npw = 0; h_npwq = 0; h_nock = 0; h_nockq = 0; h_wk = 0
    h_idxk = 1; h_idxkq = 1; h_g2 = 0; h_etk = 0; h_wwg = 0; h_wg1 = 0; h_etkq = 0
    CALL irrep_stage(npert_)
    staged = .TRUE.
  END SUBROUTINE sth_engine_init
  !
  SUBROUTINE sth_engine_set_global(vrs, dmat)
    !! vrs(nnr): local potential (smooth grid); dmat(nkb,nkb): dense D (block-diagonal deeq)
    REAL(dp), INTENT(IN) :: vrs(nnr)
    COMPLEX(dp), INTENT(IN) :: dmat(nkb, nkb)
    IF (ALLOCATED(d_vrs)) DEALLOCATE(d_vrs)
    IF (ALLOCATED(d_dmat)) DEALLOCATE(d_dmat)
    ALLOCATE(d_vrs(nnr), d_dmat(nkb, nkb))
    d_vrs = vrs
    d_dmat = dmat
  END SUBROUTINE sth_engine_set_global
  !
  SUBROUTINE sth_engine_set_k(ik, npw, npwq, nock, nockq, wk, idx_k, idx_kq, evc, evq, et_k, g2kin, &
                              vkb, wwg, wg1, et_kq, nowf)
    !! Per-k data. idx_k / idx_kq: 1-based FFT linear indices of k+G / k+q+G; evc, evq (npwx, nbnd);
    !! et_k (nbnd); g2kin (npwx) at k+q; vkb (npwx, nkb) at k+q; wwg (nbnd, nock), wg1 (nock).
    INTEGER, INTENT(IN) :: ik, npw, npwq, nock, nockq
    REAL(dp), INTENT(IN) :: wk
    INTEGER, INTENT(IN) :: idx_k(npw), idx_kq(npwq)
    COMPLEX(dp), INTENT(IN) :: evc(npwx, nbnd), evq(npwx, nbnd), vkb(npwx, nkb)
    REAL(dp), INTENT(IN) :: et_k(nbnd), g2kin(npwx), wwg(nbnd, nock), wg1(nock)
    REAL(dp), INTENT(IN), OPTIONAL :: et_kq(nbnd)
    LOGICAL, INTENT(IN), OPTIONAL :: nowf     ! KDEV: evc / evq are built on the device (sth_engine_kdev_build)
    INTEGER :: n
    IF (PRESENT(et_kq)) h_etkq(:, ik) = et_kq(:)
    h_npw(ik) = npw; h_npwq(ik) = npwq; h_nock(ik) = nock; h_nockq(ik) = nockq; h_wk(ik) = wk
    h_idxk(1:npw, ik) = idx_k(1:npw)
    h_idxkq(1:npwq, ik) = idx_kq(1:npwq)
    h_g2(1:npwq, ik) = g2kin(1:npwq)
    h_etk(:, ik) = et_k(:)
    h_wwg(:, 1:nock, ik) = wwg(:, 1:nock)
    h_wg1(1:nock, ik) = wg1(1:nock)
    IF (wdev_q) THEN
      ! QCACHE: vkb comes from the device (sth_engine_set_wdev); evq and evc are uploaded here
      IF (PRESENT(nowf)) THEN
        IF (nowf) RETURN
      ENDIF
      d_W(1:npwq, nkb + 1:nkb + nbnd, ik) = evq(1:npwq, 1:nbnd)
      IF (nock > 0) d_evc(1:npw, nslot * (ik - 1) + 1:nslot * (ik - 1) + nock) = evc(1:npw, 1:nock)
      RETURN
    ENDIF
    h_W(1:npwq, 1:nkb, ik) = vkb(1:npwq, 1:nkb)
    h_W(1:npwq, nkb + 1:nkb + nbnd, ik) = evq(1:npwq, 1:nbnd)
    DO n = 1, nock
      h_evc(1:npw, n + nslot * (ik - 1)) = evc(1:npw, n)
    ENDDO
  END SUBROUTINE sth_engine_set_k
  !
  LOGICAL FUNCTION sth_engine_wdev()
    !! QCACHE: does set_k expect vkb from the device (sth_engine_set_wdev) for this q?
    sth_engine_wdev = wdev_q
  END FUNCTION sth_engine_wdev
  !
  SUBROUTINE sth_engine_set_wdev(ik, npwq, vkb)
    !! QCACHE: vkb at k+q (npwx, nkb), already on the device (init_us_2), copied into d_W device-to-device
    INTEGER, INTENT(IN) :: ik, npwq
    COMPLEX(dp), DEVICE, INTENT(IN) :: vkb(npwx, *)
    INTEGER :: istat
    IF (nkb < 1) RETURN
    ! init_us_2 ran on an OpenACC queue; the next one overwrites vkb: synchronise on both sides
    istat = cudaDeviceSynchronize()
    istat = cudaMemcpy2D(d_W(1, 1, ik), npwx, vkb(1, 1), npwx, npwq, nkb, cudaMemcpyDeviceToDevice)
    CALL chk(istat, 'cudaMemcpy2D (vkb -> W)')
    istat = cudaDeviceSynchronize()
  END SUBROUTINE sth_engine_set_wdev
  !
  INTEGER FUNCTION slot(n, ipert, ik)
    INTEGER, INTENT(IN) :: n, ipert, ik
    slot = n + nslot * ((ipert - 1) + npert * (ik - 1))
  END FUNCTION slot
  !
  SUBROUTINE sth_engine_set_dvbare(ik, ipert, dvbare)
    !! dV_bare psi for (k, pert), (npwx, nbnd) as in iudvpsi; bands 1..nocc_k and rows 1..npwq used
    INTEGER, INTENT(IN) :: ik, ipert
    COMPLEX(dp), INTENT(IN) :: dvbare(npwx, nbnd)
    INTEGER :: n, nq
    IF (.NOT. ALLOCATED(h_dvb)) THEN
      ALLOCATE(h_dvb(npwx, nsl))     ! QCACHE: staged on first use (always allocated by irrep_stage otherwise)
      h_dvb = (0.0_dp, 0.0_dp)
    ENDIF
    nq = h_npwq(ik)
    DO n = 1, h_nock(ik)
      h_dvb(1:nq, slot(n, ipert, ik)) = dvbare(1:nq, n)
    ENDDO
  END SUBROUTINE sth_engine_set_dvbare
  !
  SUBROUTINE sth_engine_set_dpsi(ik, ipert, dpsi)
    !! warm start for (k, pert) (only needed when the first solve is not a first iteration)
    INTEGER, INTENT(IN) :: ik, ipert
    COMPLEX(dp), INTENT(IN) :: dpsi(npwx, nbnd)
    INTEGER :: n, nq
    IF (.NOT. ALLOCATED(h_dpsi0)) THEN
      ALLOCATE(h_dpsi0(npwx, nsl))   ! QCACHE: staged on first use
      h_dpsi0 = (0.0_dp, 0.0_dp)
    ENDIF
    nq = h_npwq(ik)
    DO n = 1, h_nock(ik)
      h_dpsi0(1:nq, slot(n, ipert, ik)) = dpsi(1:nq, n)
    ENDDO
    have_dpsi0 = .TRUE.
  END SUBROUTINE sth_engine_set_dpsi
  !
  SUBROUTINE sth_engine_set_nlk(ik, becp1, al1, al2, al3)
    !! per-k <beta_k|psi_k> and its k+G derivatives (phq_init's becp1, alphap), (nkb, nbnd): bands <= nocc_k
    INTEGER, INTENT(IN) :: ik
    COMPLEX(dp), INTENT(IN) :: becp1(nkb, *), al1(nkb, *), al2(nkb, *), al3(nkb, *)
    INTEGER :: n
    IF (.NOT. ALLOCATED(h_bp)) THEN
      ALLOCATE(h_bp(nkb, nslot, nksq), h_al(nkb, nslot, 3, nksq))
      h_bp = (0.0_dp, 0.0_dp); h_al = (0.0_dp, 0.0_dp)
    ENDIF
    DO n = 1, h_nock(ik)
      h_bp(:, n, ik) = becp1(:, n)
      h_al(:, n, 1, ik) = al1(:, n); h_al(:, n, 2, ik) = al2(:, n); h_al(:, n, 3, ik) = al3(:, n)
    ENDDO
  END SUBROUTINE sth_engine_set_nlk
  !
  SUBROUTINE nl_setup()
    ! D becp1 and D alphap_a for every k (strided-batched over k, D shared)
    COMPLEX(dp), DEVICE, ALLOCATABLE :: d_bp(:,:,:), d_al(:,:,:,:)
    INTEGER :: istat
    IF (ALLOCATED(d_db)) DEALLOCATE(d_db, d_da)
    ALLOCATE(d_db(nkb, nslot, nksq), d_da(nkb, nslot, 3, nksq), d_bp(nkb, nslot, nksq), d_al(nkb, nslot, 3, nksq))
    d_bp = h_bp
    d_al = h_al
    istat = cublasZgemmStridedBatched(hblas, CUBLAS_OP_N, CUBLAS_OP_N, nkb, nslot, nkb, (1.0_dp, 0.0_dp), &
            d_dmat, nkb, 0_8, d_bp, nkb, INT(nkb, 8) * nslot, (0.0_dp, 0.0_dp), d_db, nkb, INT(nkb, 8) * nslot, nksq)
    CALL chk(istat, 'cublasZgemmStridedBatched (D becp1)')
    istat = cublasZgemmStridedBatched(hblas, CUBLAS_OP_N, CUBLAS_OP_N, nkb, nslot, nkb, (1.0_dp, 0.0_dp), &
            d_dmat, nkb, 0_8, d_al, nkb, INT(nkb, 8) * nslot, (0.0_dp, 0.0_dp), d_da, nkb, INT(nkb, 8) * nslot, &
            3 * nksq)
    CALL chk(istat, 'cublasZgemmStridedBatched (D alphap)')
    CALL tsync()
    DEALLOCATE(d_bp, d_al)
  END SUBROUTINE nl_setup
  !
  SUBROUTINE sth_engine_build_dvb(dvl, uq, nat3, kbat, tpiba, kqg)
    !! dV_bare psi of the current irrep for every valid slot, on the device (replaces the per-(k, pert)
    !! dvqpsi_us + save_buffer loop of solve_linter and the buffer reads of the adapter), NC, no NLCC:
    !!   local:    (1/N) gather_kq( FFT( dvloc_p(r) psi_kn(r) ) )            (dvqpsi_us)
    !!   nonlocal: vkb_kq ps1 + sum_a [(k+q+G)_a vkb_kq] ps2_a,               (dvqpsi_us_only)
    !!             ps1 = D sum_a u_a alphap_a,  ps2_a = -i tpiba u_a D becp1
    !! dvl(nnr, npert): dvloc of each perturbation (compute_dvloc); uq(nat3, npert): its pattern;
    !! kbat(nkb): atom of each projector; kqg(npwx, 3, nksq): k+q+G.
    INTEGER, INTENT(IN) :: nat3, kbat(nkb)
    COMPLEX(dp), INTENT(IN) :: dvl(nnr, npert), uq(nat3, npert)
    REAL(dp), INTENT(IN) :: tpiba, kqg(npwx, 3, *)        ! (npwx, 3, nksq); not read when the per-q table is cached
    COMPLEX(dp), DEVICE, ALLOCATABLE :: d_u(:,:)
    INTEGER, DEVICE, ALLOCATABLE :: d_at(:)
    REAL(dp), DEVICE, ALLOCATABLE :: d_kqg(:,:,:)
    INTEGER :: istat, s0, nb, ip, a
    REAL(dp) :: scale, t0
    t0 = wtime()
    scale = 1.0_dp / REAL(nr(1) * nr(2) * nr(3), dp)
    IF (.NOT. dvb_q) CALL kerr('build_dvb without the per-q NL state (sth_dvb_req not set before init)')
    CALL zero_c(d_dvb, INT(npwx, 8) * nsl)
    ! local part, as the dpot step with dV = dvloc
    d_dv = dvl
    istat = cudaDeviceSynchronize()
    s0 = 0
    DO WHILE (s0 < nvalid)
      IF (s0 < psir_nv) THEN
        nb = MIN(sth_bmax, psir_nv - s0)
      ELSE
        nb = MIN(sth_bmax, nvalid - s0)
      ENDIF
      ip = plan_for(nb)
      nb = 2**ip
      IF (dp_on) THEN
        CALL dpot_fdx(d_dvb, s0, nb, ip, scale)
      ELSE
      IF (s0 < psir_nv) THEN
        CALL k_dvpsir<<<dim3(nblk(nnr, TPB), nb, 1), TPB, 0, strm>>>(d_box, d_psir, d_dv, d_valid, s0, &
             nnr, nslot, npert, 0)
      ELSE
        CALL psir_chunk(s0, nb, ip)
        CALL k_dvpsir<<<dim3(nblk(nnr, TPB), nb, 1), TPB, 0, strm>>>(d_box, d_box2, d_dv, d_valid, s0, &
             nnr, nslot, npert, 1)
      ENDIF
      CALL kerr('dvpsir (dvb)')
      CALL fft_exec(ip, d_box, CUFFT_FORWARD)
      CALL k_gather_add<<<dim3(nblk(npwx, TPB), nb, 1), TPB, 0, strm>>>(d_dvb, d_box, d_valid, s0, nnr, &
           npwx, nslot, npert, nksq, d_npwq, d_idxkq, scale)
      CALL kerr('gather_add (dvb)')
      ENDIF
      s0 = s0 + nb
    ENDDO
    IF (dp_on .AND. sth_fdxdp_check) CALL dp_compare(d_dvb, d_dvb, .TRUE., scale, 'dvb local')
    IF (sth_timing) THEN
      CALL tsync(); sth_t_dvb(1) = wtime() - t0; t0 = wtime()
    ENDIF
    IF (nkb > 0) THEN
      ALLOCATE(d_u(nat3, npert), d_at(nkb))
      d_u = uq
      d_at = kbat
      IF (.NOT. kqg_q) THEN
        ALLOCATE(d_kqg(npwx, 3, nksq))
        d_kqg = kqg(:, :, 1:nksq)
      ENDIF
      istat = cudaDeviceSynchronize()
      ! dvb += vkb_kq ps1   (ps in d_cf, leading dimension nproj)
      CALL k_ps<<<nsl, 32 * ((nkb + 31) / 32), 0, strm>>>(d_cf, d_da, d_db, d_u, d_at, 0, nkb, nproj, nslot, &
           npert, nksq, nat3, CMPLX(tpiba, 0.0_dp, dp))
      CALL kerr('ps1')
      istat = cublasZgemmStridedBatched(hblas, CUBLAS_OP_N, CUBLAS_OP_N, npwx, nslot * npert, nkb, &
              (1.0_dp, 0.0_dp), d_W, npwx, INT(npwx, 8) * nproj, d_cf, nproj, INT(nproj, 8) * nslot * npert, &
              (1.0_dp, 0.0_dp), d_dvb, npwx, INT(npwx, 8) * nslot * npert, nksq)
      CALL chk(istat, 'cublasZgemmStridedBatched (dvb ps1)')
      DO a = 1, 3
        ! t = vkb_kq ps2_a ; dvb += (k+q+G)_a t
        CALL k_ps<<<nsl, 32 * ((nkb + 31) / 32), 0, strm>>>(d_cf, d_da, d_db, d_u, d_at, a, nkb, nproj, nslot, &
             npert, nksq, nat3, CMPLX(tpiba, 0.0_dp, dp))
        CALL kerr('ps2')
        istat = cublasZgemmStridedBatched(hblas, CUBLAS_OP_N, CUBLAS_OP_N, npwx, nslot * npert, nkb, &
                (1.0_dp, 0.0_dp), d_W, npwx, INT(npwx, 8) * nproj, d_cf, nproj, INT(nproj, 8) * nslot * npert, &
                (0.0_dp, 0.0_dp), d_t, npwx, INT(npwx, 8) * nslot * npert, nksq)
        CALL chk(istat, 'cublasZgemmStridedBatched (dvb ps2)')
        IF (kqg_q) THEN
          CALL k_kqg_axpy<<<4096, TPB, 0, strm>>>(d_dvb, d_t, d_kqgq, a, npwx, nslot, npert, nksq, INT(npwx, 8) * nsl)
        ELSE
          CALL k_kqg_axpy<<<4096, TPB, 0, strm>>>(d_dvb, d_t, d_kqg, a, npwx, nslot, npert, nksq, INT(npwx, 8) * nsl)
        ENDIF
        CALL kerr('kqg_axpy')
      ENDDO
      CALL tsync()
      DEALLOCATE(d_u, d_at)
      IF (ALLOCATED(d_kqg)) DEALLOCATE(d_kqg)
      ! the CG temporaries used here start from zero in the solve
      CALL zero_c(d_t, INT(npwx, 8) * nsl)
      CALL zero_c(d_cf, INT(nproj, 8) * nsl)
    ENDIF
    CALL tsync()
    IF (sth_timing) sth_t_dvb(2) = wtime() - t0
  END SUBROUTINE sth_engine_build_dvb
  !
  LOGICAL FUNCTION sth_engine_kqg_ready()
    !! QCACHE: is the per-q k+q+G table on the device?
    sth_engine_kqg_ready = kqg_q
  END FUNCTION sth_engine_kqg_ready
  !
  SUBROUTINE sth_engine_set_kqg(kqg)
    !! QCACHE: keep the k+q+G table (npwx, 3, nksq) on the device for this q, if it fits with 1 GiB to spare
    !! (RTSC_STH_QCACHE_NOKQG=1 forces the per-call path, to test the fallback)
    REAL(dp), INTENT(IN) :: kqg(npwx, 3, nksq)
    INTEGER(8) :: freeb, totb, need
    INTEGER :: istat, ios
    CHARACTER(LEN=8) :: sv
    IF (.NOT. qc_q .OR. kqg_q) RETURN
    CALL get_environment_variable('RTSC_STH_QCACHE_NOKQG', sv, STATUS=ios)
    need = 8_8 * npwx * 3 * nksq
    istat = cudaMemGetInfo(freeb, totb)
    IF ((ios == 0 .AND. TRIM(sv) == '1') .OR. need > freeb - 1024_8**3) THEN
      ! printed once per q
      IF (.NOT. kqg_said) WRITE(*, '(A,F8.3,A,F8.2,A)') '     rtsc qcache: k+q+G table (', REAL(need) / 1024.0**3, &
           ' GiB) not cached (free ', REAL(freeb) / 1024.0**3, ' GiB): built per call'
      kqg_said = .TRUE.
      RETURN
    ENDIF
    ALLOCATE(d_kqgq(npwx, 3, nksq))
    d_kqgq = kqg
    istat = cudaDeviceSynchronize()
    CALL sth_mem_sample()
    kqg_q = .TRUE.
  END SUBROUTINE sth_engine_set_kqg
  !
  SUBROUTINE sth_engine_dvb_check(ik, ipert, dvpsi, dmax, ymax)
    !! diagnostic: compare the device dV_bare psi of (k, pert) with vanilla's (npwx, nbnd); accumulates
    !! the max |difference| and max |value| over bands <= nocc_k, rows <= npwq. h_dvb is the download.
    INTEGER, INTENT(IN) :: ik, ipert
    COMPLEX(dp), INTENT(IN) :: dvpsi(npwx, *)
    REAL(dp), INTENT(INOUT) :: dmax, ymax
    INTEGER :: n, nq, s
    IF (ik == 1 .AND. ipert == 1) THEN
      IF (.NOT. ALLOCATED(h_dvb)) ALLOCATE(h_dvb(npwx, nsl))
      h_dvb = d_dvb
    ENDIF
    nq = h_npwq(ik)
    DO n = 1, h_nock(ik)
      s = slot(n, ipert, ik)
      dmax = MAX(dmax, MAXVAL(ABS(h_dvb(1:nq, s) - dvpsi(1:nq, n))))
      ymax = MAX(ymax, MAXVAL(ABS(dvpsi(1:nq, n))))
    ENDDO
  END SUBROUTINE sth_engine_dvb_check
  !
  SUBROUTINE sth_engine_setup(ierr)
    !! Upload the staged data, build h_diag, psi_k(r) and the cuFFT plans.
    INTEGER, INTENT(OUT) :: ierr
    INTEGER :: istat, ik, n, ip, nb, b0
    INTEGER(8) :: need, freeb, totb
    REAL(dp) :: t0, tq(0:7)
    t0 = wtime()
    tq = 0; tq(0) = t0
    ierr = 0
    ! clear a pending (non-sticky) error left by other code, so that kerr reports only the
    ! engine's own launches; it is printed, not hidden
    istat = cudaGetLastError()
    IF (istat /= cudaSuccess) WRITE(*, '(A,A)') 'sth_engine: cleared a pending CUDA error from earlier code: ', &
         TRIM(cudaGetErrorString(istat))
    IF (.NOT. staged) THEN
      ierr = 1
      RETURN
    ENDIF
    IF (.NOT. have_handles) THEN
      istat = cudaStreamCreate(strm)
      CALL chk(istat, 'cudaStreamCreate')
      istat = cublasCreate(hblas)
      CALL chk(istat, 'cublasCreate')
      istat = cublasSetStream(hblas, strm)
      CALL chk(istat, 'cublasSetStream')
      ALLOCATE(h_nact(1))
      have_handles = .TRUE.
    ENDIF
    ! memory check (bytes): column arrays, psi(r) cache, FFT box, W
    need = 16_8 * npwx * nsl * (7 + MIN(sth_xp_m, 1)) + 16_8 * nnr * nbox + 16_8 * nnr * sth_bmax * 3 &
           + 16_8 * npwx * nproj * nksq + 4_8 * nnr * nksq
    ! sth_mp: FP32 W, h_diag, g2kin, box (+ C2C plan work areas) and the irrep's FP32 column arrays
    IF (sth_mp) need = need + 8_8 * npwx * nproj * nksq + 4_8 * npwx * (nslot + 1) * nksq + 8_8 * nnr * sth_bmax * 3 &
                            + (8_8 * npwx * 4 + 16_8 * nproj + 12_8) * nsl
    ! QCACHE: W and evc are already on the device (allocated at init): free memory is lower by exactly that, so
    ! take it off the need too and every decision below is the same as with the host stage
    IF (wdev_q) need = need - 16_8 * npwx * nproj * nksq - 16_8 * npwx * nbox
    CALL sth_mem_trim()
    istat = cudaMemGetInfo(freeb, totb)
    BLOCK
      CHARACTER(LEN=8) :: sv
      INTEGER :: ios
      INTEGER(8) :: need_nc
      CALL get_environment_variable('RTSC_STH_NOPSIR', sv, STATUS=ios)
      ! without the psi(r) cache: evc stays on the device and one more FFT box holds a chunk's psi(r)
      need_nc = need - 16_8 * nnr * nbox + 16_8 * npwx * nbox + 16_8 * nnr * sth_bmax
      psir_cached = .NOT. (ios == 0 .AND. TRIM(sv) == '1')
      CALL get_environment_variable('RTSC_STH_PSIR_PART', sv, STATUS=ios)
      psir_part = psir_cached .AND. ios == 0 .AND. TRIM(sv) == '1'
      IF (psir_part) psir_cached = .FALSE.     ! start without the cache; the partial cache is sized at the end of setup
      CALL get_environment_variable('RTSC_STH_NOPSIR', sv, STATUS=ios)
      IF (psir_cached .AND. need > freeb - 1024_8**3 .AND. need_nc <= freeb - 1024_8**3) THEN
        WRITE(*, '(A,F8.2,A,F8.2,A,F8.2,A)') 'sth_engine: psi(r) cache needs ', REAL(need) / 1024.0**3, &
             ' GiB, free ', REAL(freeb) / 1024.0**3, ' GiB: psi(r) recomputed per chunk (', &
             REAL(need_nc) / 1024.0**3, ' GiB)'
        psir_cached = .FALSE.
      ENDIF
      IF (.NOT. psir_cached) THEN
        need = need_nc
        IF (ios == 0 .AND. TRIM(sv) == '1') WRITE(*, '(A,F8.2,A)') &
             'sth_engine: psi(r) recomputed per chunk (RTSC_STH_NOPSIR=1), needs ', REAL(need) / 1024.0**3, ' GiB'
      ENDIF
    END BLOCK
    CALL sth_mem_log('setup', need)
    IF (need > freeb - 1024_8**3) THEN
      WRITE(*, '(A,F8.2,A,F8.2,A)') 'sth_engine: needs ', REAL(need) / 1024.0**3, ' GiB, free ', &
           REAL(freeb) / 1024.0**3, ' GiB'
      IF (wdev_q) THEN
        DEALLOCATE(d_W, d_evc)
        wdev_q = .FALSE.
      ENDIF
      ierr = 2
      RETURN
    ENDIF
    !
    ALLOCATE(d_npw(nksq), d_npwq(nksq), d_nock(nksq), d_nockq(nksq), d_idxk(npwx, nksq), d_idxkq(npwx, nksq))
    ALLOCATE(d_g2(npwx, nksq), d_etk(nbnd, nksq), d_hd(npwx, nslot, nksq), d_wwg(nbnd, nslot, nksq))
    ALLOCATE(d_wg1(nslot, nksq), d_wgt(nksq))
    IF (.NOT. wdev_q) ALLOCATE(d_W(npwx, nproj, nksq))
    d_npw = h_npw; d_npwq = h_npwq; d_nock = h_nock; d_nockq = h_nockq
    d_idxk = h_idxk; d_idxkq = h_idxkq; d_g2 = h_g2; d_etk = h_etk; d_wwg = h_wwg; d_wg1 = h_wg1
    IF (ALLOCATED(d_etkq)) DEALLOCATE(d_etkq)
    ALLOCATE(d_etkq(nbnd, nksq))
    d_etkq = h_etkq
    d_wgt = 2.0_dp * h_wk / omega
    IF (.NOT. wdev_q) d_W = h_W
    ! KDEV: evq / evc of the SYMROT-mapped k on the device (before the psi(r) cache reads d_evc)
    IF (ALLOCATED(h_kdrep)) THEN
      CALL sth_engine_kdev_build(sth_kdev_nrot)
      IF (sth_timing .AND. sth_kdev_nrot > 0) WRITE(*, '(A,I8,A)') '     rtsc kdev: ', sth_kdev_nrot, &
           ' k / k+q wavefunctions rotated on the device'
    ENDIF
    IF (sth_timing) THEN
      CALL tsync(); tq(1) = wtime()
    ENDIF
    ! rtsc amdahl: under RTSC_STH_FDXDP the FFT-index -> k+q map d_inv is used only by the box path (checks), so it
    ! is built after the fftdx decision, and only when that path can run (sth_fdxdp_req unset: unchanged order)
    IF (.NOT. sth_fdxdp_req .AND. qc_q) THEN
      CALL inv_build()
    ELSE IF (.NOT. sth_fdxdp_req) THEN
    BLOCK
      INTEGER, ALLOCATABLE :: hinv(:,:)
      INTEGER :: ig
      ALLOCATE(hinv(nnr, nksq))
      hinv = 0
      DO ik = 1, nksq
        DO ig = 1, h_npwq(ik)
          hinv(h_idxkq(ig, ik), ik) = ig
        ENDDO
      ENDDO
      IF (ALLOCATED(d_inv)) DEALLOCATE(d_inv)
      ALLOCATE(d_inv(nnr, nksq))
      d_inv = hinv
      DEALLOCATE(hinv)
    END BLOCK
    ENDIF
    !
    IF (sth_timing) THEN
      CALL tsync(); tq(2) = wtime()
    ENDIF
    ALLOCATE(d_box(nnr, sth_bmax))
    IF (psir_cached) THEN
      ALLOCATE(d_psir(nnr, nbox))
    ELSE
      ALLOCATE(d_box2(nnr, sth_bmax))
    ENDIF
    istat = cudaDeviceSynchronize()
    CALL sth_mem_sample()
    !
    CALL make_plans()
    !
    CALL zero_c(d_box, INT(nnr, 8) * sth_bmax)
    !
    ! h_diag
    CALL k_hprec<<<nbox, RB, 0, strm>>>(d_hd, d_W, d_g2, d_npwq, d_nock, npwx, nproj, nkb, nslot, nksq)
    CALL kerr('hprec')
    !
    ! sth_mp: FP32 copies of the operator data
    mp_q = sth_mp
    IF (mp_q) THEN
      ALLOCATE(s_W(npwx, nproj, nksq), s_hd(npwx, nslot, nksq), s_g2(npwx, nksq), s_vrs(nnr), s_dmat(nkb, nkb))
      ALLOCATE(s_box(nnr, sth_bmax))
      CALL d2s_c(s_W, d_W, INT(npwx, 8) * nproj * nksq)
      CALL d2s_r(s_hd, d_hd, INT(npwx, 8) * nslot * nksq)
      CALL d2s_r(s_g2, d_g2, INT(npwx, 8) * nksq)
      CALL d2s_r(s_vrs, d_vrs, INT(nnr, 8))
      CALL d2s_c(s_dmat, d_dmat, INT(nkb, 8) * nkb)
      CALL zero_s(s_box, INT(nnr, 8) * sth_bmax)
      CALL make_plans32()
      CALL tsync()
      CALL sth_mem_sample()
    ENDIF
    !
    IF (sth_timing) THEN
      CALL tsync(); tq(3) = wtime()
    ENDIF
    ! psi_k(r) for n <= nocc_k: scatter evc at k+G, inverse FFT in place (no cache: keep evc only)
    IF (.NOT. wdev_q) THEN
      ALLOCATE(d_evc(npwx, nbox))
      d_evc = h_evc
    ENDIF
    IF (psir_cached) THEN
      CALL zero_c(d_psir, INT(nnr, 8) * nbox)
      CALL k_scatter_evc<<<dim3(nblk(npwx, TPB), nbox, 1), TPB, 0, strm>>>(d_psir, d_evc, nnr, npwx, &
           nslot, nksq, d_npw, d_nock, d_idxk)
      CALL kerr('scatter_evc')
      b0 = 0
      DO WHILE (b0 < nbox)
        nb = MIN(sth_bmax, nbox - b0)
        ! largest plan not exceeding nb, so the in-place transform never runs past the array
        ip = nplan - 1
        DO WHILE (2**ip > nb)
          ip = ip - 1
        ENDDO
        CALL fft_exec(ip, d_psir(:, b0 + 1:b0 + 2**ip), CUFFT_INVERSE)
        b0 = b0 + 2**ip
      ENDDO
      CALL tsync()
      DEALLOCATE(d_evc)
      psir_kc = nksq
    ELSE
      psir_kc = 0
    ENDIF
    ! LIFT-F factor (needs h_W; the box is free here)
    IF (sth_lift > 0.0_dp .AND. .NOT. mp_q) THEN
      CALL lift_build()
    ELSE
      CALL lift_free()
      IF (sth_lift > 0.0_dp) WRITE(*, '(A)') '     rtsc lift: ignored under RTSC_STH_MP (FP64 CG path only)'
    ENDIF
    ! CBSR canary (needs W; the box is free here); exclusive with LIFT-F and MP
    IF (sth_cbsr .AND. .NOT. mp_q .AND. .NOT. lift_q) THEN
      CALL cbsr_build()
    ELSE
      CALL cbsr_free()
      IF (sth_cbsr) WRITE(*, '(A)') '     rtsc cbsr: ignored (RTSC_STH_MP or LIFT-F on)'
    ENDIF
    IF (psir_part) CALL psir_part_build()
    IF (ALLOCATED(h_evc)) DEALLOCATE(h_evc)
    IF (ALLOCATED(h_W)) DEALLOCATE(h_W)
    IF (sth_timing) THEN
      CALL tsync(); tq(4) = wtime()
    ENDIF
    dvb_q = sth_dvb_req .AND. ALLOCATED(h_bp)
    IF (dvb_q .AND. nkb > 0) CALL nl_setup()
    IF (ALLOCATED(h_bp)) DEALLOCATE(h_bp, h_al)
    pfft_checked = .FALSE.
    fdx_on = .FALSE.
    IF (sth_fftdx_req) THEN
      BLOCK
        INTEGER :: ier, ns_, nx_, pc_
        sth_t_fdxsetup = wtime()
        ier = fdx_setup(nr, nrx, nksq, npwx, h_npwq, h_idxkq, 1024)
        sth_t_fdxsetup = wtime() - sth_t_fdxsetup
        IF (ier == 0) THEN
          fdx_on = .TRUE.
          CALL fdx_info(ns_, nx_, pc_)
          WRITE(*, '(A,I6,A,F6.3,A,I4,A,I5)') '     rtsc fftdx: on, sticks', ns_, ' (', &
               REAL(ns_) / REAL(nr(1) * nr(2)), ' of the z-columns), occupied x-columns', nx_, ', chunk', pc_
        ELSE
          WRITE(*, '(A,I3,A)') '     rtsc fftdx: unsupported grid/box or not built (', ier, '), pfft/cuFFT path kept'
        ENDIF
        IF (sth_timing) WRITE(*, '(A,F8.3,A)') '     rtsc fftdx: fdx_setup ', sth_t_fdxsetup, ' s'
      END BLOCK
    ENDIF
    dp_on = fdx_on .AND. sth_fdxdp_req .AND. nnr == nr(1) * nr(2) * nr(3)
    IF (sth_fdxdp_req) THEN
      IF (ALLOCATED(d_inv)) DEALLOCATE(d_inv)
      IF ((.NOT. dp_on .OR. sth_fdxdp_check .OR. sth_fftdx_check .OR. sth_pfft_check) .AND. qc_q) THEN
        CALL inv_build()
      ELSE IF (.NOT. dp_on .OR. sth_fdxdp_check .OR. sth_fftdx_check .OR. sth_pfft_check) THEN
        BLOCK
          INTEGER, ALLOCATABLE :: hinv(:,:)
          INTEGER :: ig
          ALLOCATE(hinv(nnr, nksq))
          hinv = 0
          DO ik = 1, nksq
            DO ig = 1, h_npwq(ik)
              hinv(h_idxkq(ig, ik), ik) = ig
            ENDDO
          ENDDO
          IF (ALLOCATED(d_inv)) DEALLOCATE(d_inv)
          ALLOCATE(d_inv(nnr, nksq))
          d_inv = hinv
          DEALLOCATE(hinv)
        END BLOCK
      ENDIF
    ENDIF
    dp_checked = .FALSE.
    dr_checked = .FALSE.
    IF (sth_fdxdp_req) WRITE(*, '(A,L2)') '     rtsc fdxdp: dpot/drho/DVB-local on the pruned path:', dp_on
    IF (sth_pfft_req .AND. .NOT. fdx_on) THEN
      BLOCK
        INTEGER :: ier, ns_, nx_, pc_
        CALL pfft_setup(nr, nrx, nksq, npwx, h_npwq, h_idxkq, 1024, ier)
        IF (ier == 0) THEN
          CALL pfft_info(ns_, nx_, pc_)
          WRITE(*, '(A,I6,A,F6.3,A,I4,A,I5)') '     rtsc pfft: on, sticks', ns_, ' (', &
               REAL(ns_) / REAL(nr(1) * nr(2)), ' of the z-columns), occupied x-columns', nx_, ', chunk', pc_
        ELSE
          WRITE(*, '(A,I3,A)') '     rtsc pfft: unsupported grid/box (', ier, '), cuFFT path kept'
        ENDIF
        IF (ier == 0 .AND. mp_q) THEN
          CALL pfs_setup(nr, nrx, nksq, npwx, h_npwq, h_idxkq, 1024, ier)
          IF (ier /= 0) WRITE(*, '(A,I3,A)') '     rtsc pfft: FP32 twin unsupported (', ier, '), C2C cuFFT kept'
        ENDIF
      END BLOCK
    ENDIF
    istat = cudaDeviceSynchronize()
    CALL chk(istat, 'setup')
    tq(5) = wtime()
    qready = .TRUE.
    CALL sth_engine_irrep_commit(ierr)
    sth_t_setup = wtime() - t0
    IF (sth_timing) WRITE(*, '(A,7F8.3)') '     rtsc setup t: upload/W, d_inv, plans+hd, psi(r), nl+fdx, commit, total', &
         tq(1) - tq(0), tq(2) - tq(1), tq(3) - tq(2), tq(4) - tq(3), tq(5) - tq(4), wtime() - tq(5), sth_t_setup
  END SUBROUTINE sth_engine_setup
  !
  SUBROUTINE inv_build()
    !! QCACHE: d_inv(idx_kq(ig, k), k) = ig on the device (the host loop of the default path, same integers)
    IF (ALLOCATED(d_inv)) DEALLOCATE(d_inv)
    ALLOCATE(d_inv(nnr, nksq))
    CALL k_izero<<<4096, TPB, 0, strm>>>(d_inv, INT(nnr, 8) * nksq)
    CALL kerr('izero (inv)')
    CALL k_inv<<<dim3(nblk(npwx, TPB), nksq, 1), TPB, 0, strm>>>(d_inv, d_idxkq, d_npwq, nnr, npwx)
    CALL kerr('inv')
    CALL tsync()
  END SUBROUTINE inv_build
  !
  SUBROUTINE irrep_free()
    IF (ALLOCATED(d_xp)) DEALLOCATE(d_xp)
    CALL irrep_free_stage()
    IF (ALLOCATED(h_dpsi)) DEALLOCATE(h_dpsi)
    IF (ALLOCATED(h_valid)) DEALLOCATE(h_valid)
    IF (ALLOCATED(d_dvb)) DEALLOCATE(d_dvb, d_d0, d_dpsi, d_g, d_h, d_t, d_cf, d_cf2, d_ps, d_rho, d_conv, &
                                     d_cnt, d_kter, d_nact, d_av, d_cv, d_dv, d_drho, d_r0, d_nst)
    IF (ALLOCATED(d_valid)) DEALLOCATE(d_valid, d_act, d_act2)
    CALL irrep_free_mp()
  END SUBROUTINE irrep_free
  !
  SUBROUTINE irrep_free_stage()
    IF (ALLOCATED(h_dvb)) DEALLOCATE(h_dvb)
    IF (ALLOCATED(h_dpsi0)) DEALLOCATE(h_dpsi0)
  END SUBROUTINE irrep_free_stage
  !
  SUBROUTINE irrep_free_mp()
    IF (ALLOCATED(s_e)) DEALLOCATE(s_e, s_g, s_h, s_t, s_cf, s_cf2, d_tol, d_iconv, d_oact)
    ax_valid = .FALSE.
  END SUBROUTINE irrep_free_mp
  !
  SUBROUTINE irrep_stage(npert_)
    INTEGER, INTENT(IN) :: npert_
    npert = npert_
    nsl = nslot * npert * nksq
    ngrp = npert * nksq
    ! QCACHE: h_dvb / h_dpsi0 are allocated (zeroed) by their first set_* call; never, with DVB on a first iteration
    IF (.NOT. qc_q) THEN
      ALLOCATE(h_dvb(npwx, nsl), h_dpsi0(npwx, nsl))
      h_dvb = (0.0_dp, 0.0_dp)
      h_dpsi0 = (0.0_dp, 0.0_dp)
    ENDIF
    have_dpsi0 = .FALSE.
  END SUBROUTINE irrep_stage
  !
  SUBROUTINE sth_engine_irrep(npert_, ierr)
    !! Start a new irrep on the per-q state already set up: keeps k/k+q data, W, h_diag, psi(r) and
    !! the plans; then set_dvbare (and set_dpsi for a warm start) and sth_engine_irrep_commit.
    !! QCACHE: the irrep's device work arrays are kept here; the commit reuses them if npert is unchanged.
    INTEGER, INTENT(IN) :: npert_
    INTEGER, INTENT(OUT) :: ierr
    ierr = 0
    ready = .FALSE.
    IF (.NOT. qready) THEN
      ierr = 1
      RETURN
    ENDIF
    IF (npert_ > MAXPERT .OR. npert_ < 1) THEN
      ierr = 3
      RETURN
    ENDIF
    IF (qc_q) THEN
      CALL irrep_free_stage()
    ELSE
      CALL irrep_free()
    ENDIF
    CALL irrep_stage(npert_)
  END SUBROUTINE sth_engine_irrep
  !
  SUBROUTINE sth_engine_irrep_commit(ierr)
    !! upload the staged dV_bare psi (and warm start) of the current irrep; allocate its work arrays
    INTEGER, INTENT(OUT) :: ierr
    INTEGER :: istat, ik, n, ip
    INTEGER(8) :: need, freeb, totb
    LOGICAL :: reuse
    REAL(dp) :: t0, t1
    t0 = wtime()
    ierr = 0
    ! QCACHE: same shape as the previous irrep of this q -> keep its device arrays (no free / trim / cudaMalloc)
    reuse = .FALSE.
    IF (qc_q .AND. ALLOCATED(d_dvb) .AND. ALLOCATED(d_valid) .AND. ALLOCATED(h_dpsi)) THEN
      reuse = SIZE(d_dvb, 2) == nsl .AND. SIZE(d_dv, 2) == npert .AND. (ALLOCATED(d_xp) .EQV. (sth_xp_m > 0)) &
              .AND. (ALLOCATED(s_e) .EQV. mp_q)
    ENDIF
    IF (.NOT. reuse) THEN
    IF (ALLOCATED(d_dvb)) DEALLOCATE(d_dvb, d_d0, d_dpsi, d_g, d_h, d_t, d_cf, d_cf2, d_ps, d_rho, d_conv, &
                                     d_cnt, d_kter, d_nact, d_av, d_cv, d_dv, d_drho, d_r0, d_nst)
    IF (ALLOCATED(d_valid)) DEALLOCATE(d_valid, d_act, d_act2)
    IF (ALLOCATED(h_valid)) DEALLOCATE(h_valid)
    IF (ALLOCATED(h_dpsi)) DEALLOCATE(h_dpsi)
    CALL irrep_free_mp()
    IF (ALLOCATED(d_xp)) DEALLOCATE(d_xp)
    need = 16_8 * npwx * nsl * (7 + MIN(sth_xp_m, 1))
    IF (mp_q) need = need + (8_8 * npwx * 4 + 16_8 * nproj + 12_8) * nsl
    CALL sth_mem_trim()
    istat = cudaMemGetInfo(freeb, totb)
    IF (need > freeb - 512_8 * 1024**2) THEN
      WRITE(*, '(A,F8.2,A,F8.2,A)') 'sth_engine: irrep needs ', REAL(need) / 1024.0**3, ' GiB, free ', &
           REAL(freeb) / 1024.0**3, ' GiB'
      ierr = 2
      RETURN
    ENDIF
    IF (sth_timing) THEN
      t1 = wtime(); sth_t_commit(1) = t1 - t0; t0 = t1
    ENDIF
    ALLOCATE(d_dvb(npwx, nsl), d_d0(npwx, nsl), d_dpsi(npwx, nsl), d_g(npwx, nsl), d_h(npwx, nsl))
    ALLOCATE(d_t(npwx, nsl), d_cf(nproj, nsl), d_cf2(nproj, nsl), d_ps(nbnd, nsl))
    ALLOCATE(d_rho(nsl), d_conv(nsl), d_cnt(ngrp), d_kter(ngrp), d_nact(1), d_av(nsl), d_cv(nsl), d_r0(nsl), d_nst(nsl))
    ALLOCATE(d_dv(nnr, npert), d_drho(nnr, npert))
    ALLOCATE(h_dpsi(npwx, nsl))
    ELSE
      IF (sth_timing) THEN
        t1 = wtime(); sth_t_commit(1) = t1 - t0; t0 = t1
      ENDIF
    ENDIF
    h_dpsi_valid = .FALSE.
    xp_nc = 0
    xp_rec = 0
    IF (sth_xp_m > 0 .AND. .NOT. reuse) THEN
      ALLOCATE(d_xp(npwx, nsl))
      IF (ALLOCATED(h_xh)) THEN
        IF (SIZE(h_xh, 1) /= npwx .OR. SIZE(h_xh, 2) /= nsl .OR. SIZE(h_xh, 3) /= sth_xp_m) DEALLOCATE(h_xh)
      ENDIF
      IF (.NOT. ALLOCATED(h_xh)) ALLOCATE(h_xh(npwx, nsl, sth_xp_m))
    ENDIF
    IF (sth_timing) THEN
      istat = cudaDeviceSynchronize(); t1 = wtime(); sth_t_commit(2) = t1 - t0; t0 = t1
    ENDIF
    ! QCACHE: an unstaged array is all zeros: zero it on the device instead of uploading zeros (same values)
    IF (ALLOCATED(h_dvb)) THEN
      d_dvb = h_dvb
    ELSE
      CALL zero_c(d_dvb, INT(npwx, 8) * nsl)
    ENDIF
    IF (ALLOCATED(h_dpsi0)) THEN
      d_dpsi = h_dpsi0
    ELSE
      CALL zero_c(d_dpsi, INT(npwx, 8) * nsl)
    ENDIF
    IF (sth_timing) THEN
      istat = cudaDeviceSynchronize(); t1 = wtime(); sth_t_commit(3) = t1 - t0; t0 = t1
    ENDIF
    IF (.NOT. reuse) THEN
    ! valid slot list (k-major, then pert, then band)
    nvalid = 0
    DO ik = 1, nksq
      nvalid = nvalid + npert * h_nock(ik)
    ENDDO
    ALLOCATE(h_valid(nvalid), d_valid(nvalid), d_act(nsl), d_act2(nsl))
    nvalid = 0
    DO ik = 1, nksq
      DO ip = 1, npert
        DO n = 1, h_nock(ik)
          nvalid = nvalid + 1
          h_valid(nvalid) = slot(n, ip, ik)
        ENDDO
      ENDDO
    ENDDO
    d_valid = h_valid
    ENDIF
    psir_nv = 0
    DO ik = 1, MIN(psir_kc, nksq)
      psir_nv = psir_nv + npert * h_nock(ik)
    ENDDO
    istat = cudaDeviceSynchronize()
    ! zero the work arrays (invalid slots must stay zero; stale values must be finite)
    CALL zero_c(d_d0, INT(npwx, 8) * nsl)
    CALL zero_c(d_g, INT(npwx, 8) * nsl)
    CALL zero_c(d_h, INT(npwx, 8) * nsl)
    CALL zero_c(d_t, INT(npwx, 8) * nsl)
    CALL zero_c(d_cf, INT(nproj, 8) * nsl)
    CALL zero_c(d_cf2, INT(nproj, 8) * nsl)
    IF (mp_q) THEN
      IF (.NOT. reuse) THEN
        ALLOCATE(s_e(npwx, nsl), s_g(npwx, nsl), s_h(npwx, nsl), s_t(npwx, nsl), s_cf(nproj, nsl), s_cf2(nproj, nsl))
        ALLOCATE(d_tol(nsl), d_iconv(nsl), d_oact(nsl))
      ENDIF
      CALL zero_s(s_e, INT(npwx, 8) * nsl)
      CALL zero_s(s_g, INT(npwx, 8) * nsl)
      CALL zero_s(s_h, INT(npwx, 8) * nsl)
      CALL zero_s(s_t, INT(npwx, 8) * nsl)
      CALL zero_s(s_cf, INT(nproj, 8) * nsl)
      CALL zero_s(s_cf2, INT(nproj, 8) * nsl)
      d_iconv = 0
      d_tol = 0.0_dp
    ENDIF
    ax_valid = .FALSE.
    CALL tsync()
    CALL sth_mem_sample()
    CALL sth_mem_log('commit')
    IF (sth_timing) sth_t_commit(4) = wtime() - t0
    ready = .TRUE.
  END SUBROUTINE sth_engine_irrep_commit
  !
  SUBROUTINE sth_engine_reset_dpsi()
    !! restore the staged warm start (for repeated timing runs of the same call)
    IF (ALLOCATED(h_dpsi0)) THEN
      d_dpsi = h_dpsi0
    ELSE
      CALL zero_c(d_dpsi, INT(npwx, 8) * nsl)
    ENDIF
    h_dpsi_valid = .FALSE.
    ax_valid = .FALSE.
  END SUBROUTINE sth_engine_reset_dpsi
  !
  SUBROUTINE sth_engine_solve(first_iter, thresh, dvscf, drho, tot_num_iter, tot_cg_calls, nfail)
    !! One sternheimer_kernel call. dvscf(nnr, npert): dV_ind (ignored when first_iter);
    !! drho(nnr, npert): incremented by this call's density response (as drhoout).
    !! tot_num_iter / tot_cg_calls as in sternheimer_kernel; nfail = (k, pert) solves not converged.
    LOGICAL, INTENT(IN) :: first_iter
    REAL(dp), INTENT(IN) :: thresh
    COMPLEX(dp), INTENT(IN) :: dvscf(nnr, npert)
    COMPLEX(dp), INTENT(INOUT) :: drho(nnr, npert)
    INTEGER, INTENT(OUT) :: tot_num_iter, tot_cg_calls, nfail
    INTEGER :: istat, iter, nact, s0, nb, ip, np, first, i, k, nres
    REAL(dp) :: scale, t0, t1
    REAL(dp), ALLOCATABLE :: kter(:)
    INTEGER, ALLOCATABLE :: conv(:)
    !
    t0 = wtime()
    istat = cudaGetLastError()   ! see sth_engine_setup
    scale = 1.0_dp / REAL(nr(1) * nr(2) * nr(3), dp)
    h_dpsi_valid = .FALSE.
    sth_nvalid = nvalid
    sth_t_gemm = 0; sth_t_fft = 0; sth_t_lapply = 0
    d_drho = drho
    IF (.NOT. first_iter) d_dv = dvscf
    istat = cudaDeviceSynchronize()
    !
    ! dvpsi = dvbare (+ dvscf psi)
    CALL k_copy<<<4096, TPB, 0, strm>>>(d_d0, d_dvb, INT(npwx, 8) * nsl)
    CALL kerr('copy')
    IF (.NOT. first_iter) THEN
      s0 = 0
      DO WHILE (s0 < nvalid)
        ! chunks never straddle the cached / recomputed boundary psir_nv (= nvalid or 0 unless RTSC_STH_PSIR_PART)
        IF (s0 < psir_nv) THEN
          nb = MIN(sth_bmax, psir_nv - s0)
        ELSE
          nb = MIN(sth_bmax, nvalid - s0)
        ENDIF
        ip = plan_for(nb)
        nb = 2**ip
        IF (dp_on) THEN
          CALL dpot_fdx(d_d0, s0, nb, ip, scale)
        ELSE
        IF (s0 < psir_nv) THEN
          CALL k_dvpsir<<<dim3(nblk(nnr, TPB), nb, 1), TPB, 0, strm>>>(d_box, d_psir, d_dv, d_valid, s0, &
               nnr, nslot, npert, 0)
        ELSE
          CALL psir_chunk(s0, nb, ip)
          CALL k_dvpsir<<<dim3(nblk(nnr, TPB), nb, 1), TPB, 0, strm>>>(d_box, d_box2, d_dv, d_valid, s0, &
               nnr, nslot, npert, 1)
        ENDIF
        CALL kerr('dvpsir')
        CALL fft_exec(ip, d_box, CUFFT_FORWARD)
        CALL k_gather_add<<<dim3(nblk(npwx, TPB), nb, 1), TPB, 0, strm>>>(d_d0, d_box, d_valid, s0, nnr, &
             npwx, nslot, npert, nksq, d_npwq, d_idxkq, scale)
        CALL kerr('gather_add')
        ENDIF
        s0 = s0 + nb
      ENDDO
      IF (dp_on .AND. sth_fdxdp_check .AND. .NOT. dp_checked) THEN
        CALL dp_compare(d_d0, d_dvb, .FALSE., scale, 'dpot')
        dp_checked = .TRUE.
      ENDIF
    ENDIF
    IF (sth_timing) THEN
      CALL tsync(); t1 = wtime(); sth_t_dpot = t1 - t0; t0 = t1
    ENDIF
    !
    ! orthogonalize (lgauss): ps = evq^H dvpsi ; ps *= wwg ; dvpsi *= wg1 ; dvpsi = evq ps - dvpsi.
    ! One ZGEMM pair per (k, pert) with exactly the shapes of orthogonalize.f90 (m = nbnd, n = nocc_k,
    ! k = npwq), so that the right-hand side matches vanilla's rounding as closely as possible: the
    ! screened, projected dvpsi can be a strong cancellation, and its rounding feeds every CG step.
    IF (mp_q .OR. sth_obatch) THEN
      CALL ortho_batched()
    ELSE
      CALL ortho_exact()
    ENDIF
    IF (sth_timing) THEN
      CALL tsync(); t1 = wtime(); sth_t_ortho = t1 - t0; t0 = t1
    ENDIF
    !
    ! CG (cgsolve_all), all columns at once
    IF (mp_q) THEN
      CALL mp_solve(first_iter, thresh)
    ELSEIF (cbsr_q .AND. cbsr_rf) THEN
      ! rtsc b2r: CBSR in refinement form under QE's threshold (RTSC_STH_CBSR_RF)
      CALL cbsr_rf_solve(first_iter, thresh)
      sth_n64 = sth_ncolapp; sth_n32 = 0; sth_nround = 0
    ELSEIF (cbsr_q) THEN
      ! CBSR canary: exact spectral solve of the engine's system (no CG, no thresh)
      CALL cbsr_solve()
      d_conv = 1
      d_kter = 0.0_dp
      sth_nsteps = 0; sth_ncolapp = 0; sth_nact0 = 0
      sth_n64 = 0; sth_n32 = 0; sth_nround = 0
    ELSE
    ax_valid = .FALSE.
    IF (first_iter) CALL zero_c(d_dpsi, INT(npwx, 8) * nsl)
    ! dpsi extrapolation (sth_engine_xp_set): keep the start dpsi(k-1), then add sum_j c_j delta_j
    IF (ALLOCATED(d_xp) .AND. xp_rec > 0) THEN
      CALL k_copy<<<4096, TPB, 0, strm>>>(d_xp, d_dpsi, INT(npwx, 8) * nsl)
      CALL kerr('copy (xp)')
    ENDIF
    IF (ALLOCATED(d_xp)) THEN
      DO i = 1, xp_nc
        istat = cudaMemcpyAsync(d_g, h_xh(:, :, xp_slot(i)), INT(npwx, 8) * nsl, strm)
        CALL k_axpyc<<<4096, TPB, 0, strm>>>(d_dpsi, d_g, xp_c(i), INT(npwx, 8) * nsl)
        CALL kerr('axpy (xp)')
      ENDDO
    ENDIF
    d_conv = 0
    d_cnt = 0
    d_kter = 0.0_dp
    d_rho = 0.0_dp
    nres = 0
    IF (sth_trn < 0) THEN
      BLOCK
        CHARACTER(LEN=8) :: sv
        INTEGER :: ios
        CALL get_environment_variable('RTSC_STH_TRUERES', sv, STATUS=ios)
        sth_trn = 0
        IF (ios == 0 .AND. TRIM(sv) == '1') sth_trn = 1
        IF (sth_trn == 1) WRITE(*, '(A,L2)') '     rtsc trueres: stop on ||g||_2 + true-residual recheck; DEFL path', sth_defl > 0
      END BLOCK
    ENDIF
    istat = cudaDeviceSynchronize()
    CALL k_icopy<<<nblk(nvalid, TPB), TPB, 0, strm>>>(d_act, d_valid, nvalid)
    nact = nvalid
    ! g = A dpsi - dvpsi
    CALL apply_op(d_dpsi, d_g, d_act, nact)
    CALL k_sub<<<4096, TPB, 0, strm>>>(d_g, d_d0, INT(npwx, 8) * nsl)
    CALL kerr('sub')
    sth_nsteps = 0
    sth_ncolapp = nact
    DO iter = 1, maxter
      first = 0
      IF (iter == 1) first = 1
      IF (lift_q) THEN
        CALL lift_apply()
        CALL k_check_lift<<<nact, sth_rb, 0, strm>>>(d_act, nslot, npert, npwx, nksq, d_npwq, d_hd, d_g, d_h, &
             d_rho, d_conv, d_cnt, thresh, first, sth_eta, d_r0, d_nst, d_lz, d_lt, d_etk, nbnd, lift_v0, d_lok)
      ELSEIF (sth_defl > 0) THEN
        CALL k_check_defl<<<nact, sth_rb, 0, strm>>>(d_act, nslot, npert, npwx, nksq, d_npwq, d_hd, d_g, d_h, &
             d_rho, d_conv, d_cnt, thresh, first, sth_eta, d_r0, d_nst, d_W, nproj, nkb, nbnd, d_nockq, d_etk, &
             d_etkq, sth_dfloor, MAX(sth_trn, 0))
      ELSE
        CALL k_check<<<nact, sth_rb, 0, strm>>>(d_act, nslot, npert, npwx, nksq, d_npwq, d_hd, d_g, d_h, d_rho, &
             d_conv, d_cnt, thresh, first, sth_eta, d_r0, d_nst)
      ENDIF
      CALL kerr('check')
      CALL k_kter<<<nblk(ngrp, TPB), TPB, 0, strm>>>(ngrp, npert, nksq, d_nock, d_cnt, d_kter)
      CALL kerr('kter')
      CALL k_compact<<<1, 1024, 0, strm>>>(d_act, d_act2, nact, d_conv, d_nact)
      CALL kerr('compact')
      istat = cudaMemcpyAsync(h_nact, d_nact, 1, strm)
      CALL tsync()
      nact = h_nact(1)
      IF (iter == 1) THEN
        sth_nact0 = nact
        IF (sth_timing) CALL r0_profile(thresh)
      ENDIF
      sth_nsteps = iter
      IF (nact == 0 .AND. sth_trn == 1 .AND. sth_defl > 0 .AND. nres < 3) THEN
        ! true-residual recheck: g = A dpsi - rhs on every valid column; failing columns restart the CG (first = 1)
        nres = nres + 1
        CALL k_icopy<<<nblk(nvalid, TPB), TPB, 0, strm>>>(d_act, d_valid, nvalid)
        nact = nvalid
        CALL apply_op(d_dpsi, d_g, d_act, nact)
        CALL k_sub<<<4096, TPB, 0, strm>>>(d_g, d_d0, INT(npwx, 8) * nsl)
        CALL kerr('sub (trueres)')
        sth_ncolapp = sth_ncolapp + nact
        d_conv = 0
        CALL k_check_defl<<<nact, sth_rb, 0, strm>>>(d_act, nslot, npert, npwx, nksq, d_npwq, d_hd, d_g, d_h, &
             d_rho, d_conv, d_cnt, thresh, 1, sth_eta, d_r0, d_nst, d_W, nproj, nkb, nbnd, d_nockq, d_etk, &
             d_etkq, sth_dfloor, 1)
        CALL kerr('check (trueres)')
        CALL k_compact<<<1, 1024, 0, strm>>>(d_act, d_act2, nact, d_conv, d_nact)
        CALL kerr('compact (trueres)')
        istat = cudaMemcpyAsync(h_nact, d_nact, 1, strm)
        CALL tsync()
        nact = h_nact(1)
        trn_rechk = trn_rechk + nvalid
        trn_fail = trn_fail + nact
      ENDIF
      IF (nact == 0) EXIT
      CALL apply_op(d_h, d_t, d_act, nact)
      sth_ncolapp = sth_ncolapp + nact
      CALL k_update<<<nact, sth_rb, 0, strm>>>(d_act, nslot, npert, npwx, nksq, d_npwq, d_h, d_g, d_t, d_dpsi, d_av, d_cv)
      CALL kerr('update')
      IF (sth_debug_slot > 0) CALL debug_step(iter)
    ENDDO
    IF (sth_timing) CALL band_profile(thresh)
    IF (sth_trn == 1 .AND. sth_timing) WRITE(*, '(A,I2,A,I12,A,I12)') '     rtsc trueres: rechecks this call', nres, &
         ', cumulative columns rechecked', trn_rechk, ', failing (restarted)', trn_fail
    ! dpsi extrapolation: delta = dpsi(k) - dpsi(k-1) into history slot xp_rec (d_g is free after CG)
    IF (ALLOCATED(d_xp) .AND. xp_rec > 0) THEN
      CALL k_copy<<<4096, TPB, 0, strm>>>(d_g, d_dpsi, INT(npwx, 8) * nsl)
      CALL k_sub<<<4096, TPB, 0, strm>>>(d_g, d_xp, INT(npwx, 8) * nsl)
      CALL kerr('delta (xp)')
      istat = cudaMemcpyAsync(h_xh(:, :, xp_rec), d_g, INT(npwx, 8) * nsl, strm)
      CALL tsync()
    ENDIF
    sth_n64 = sth_ncolapp; sth_n32 = 0; sth_nround = 0
    ENDIF
    xp_nc = 0
    xp_rec = 0
    IF (sth_timing) THEN
      CALL tsync(); t1 = wtime(); sth_t_cg = t1 - t0; t0 = t1
      IF (lift_q) WRITE(*, '(A,F9.3,A,I10)') '     rtsc lift apply ', sth_t_lapply, ' s, colapp ', sth_ncolapp
    ENDIF
    !
    ! incdrhoscf
    s0 = 0
    DO WHILE (s0 < nvalid)
      IF (s0 < psir_nv) THEN
        nb = MIN(sth_bmax, psir_nv - s0)
      ELSE
        nb = MIN(sth_bmax, nvalid - s0)
      ENDIF
      ip = plan_for(nb)
      nb = 2**ip
      np = 2**ip
      IF (dp_on) THEN
        istat = fdx_inv_d(d_box, d_dpsi, d_valid, s0, nb, nslot, npert, npwx, nnr, strm)
        IF (istat /= 0) CALL errore_stop('fdx_inv_d', istat)
      ELSE
      CALL k_scatter_full<<<dim3(nblk(nnr, TPB), nb, 1), TPB, 0, strm>>>(d_box, d_dpsi, d_valid, s0, nnr, &
           npwx, nslot, npert, nksq, d_inv)
      CALL kerr('scatter (drho)')
      CALL fft_exec(ip, d_box, CUFFT_INVERSE)
      ENDIF
      IF (s0 < psir_nv) THEN
        CALL k_drho<<<nblk(nnr, TPB), TPB, 0, strm>>>(d_drho, d_box, d_psir, d_valid, s0, nb, nnr, nslot, &
             npert, d_wgt, 0)
      ELSE
        CALL psir_chunk(s0, nb, ip)
        CALL k_drho<<<nblk(nnr, TPB), TPB, 0, strm>>>(d_drho, d_box, d_box2, d_valid, s0, nb, nnr, nslot, &
             npert, d_wgt, 1)
      ENDIF
      CALL kerr('drho')
      s0 = s0 + nb
    ENDDO
    CALL tsync()
    IF (dp_on .AND. sth_fdxdp_check .AND. .NOT. dr_checked .AND. .NOT. first_iter) THEN
      CALL dr_compare(drho)
      dr_checked = .TRUE.
    ENDIF
    drho = d_drho
    !
    ! iteration count, exactly as sternheimer_kernel: sum over (k, pert) of INT(kter_eff)
    ALLOCATE(kter(ngrp), conv(nsl))
    kter = d_kter
    conv = d_conv
    tot_num_iter = 0
    DO i = 1, ngrp
      tot_num_iter = tot_num_iter + INT(kter(i))
    ENDDO
    tot_cg_calls = ngrp
    nfail = 0
    DO i = 1, ngrp
      k = (i - 1) / npert + 1
      IF (ANY(conv(nslot * (i - 1) + 1:nslot * (i - 1) + h_nock(k)) == 0)) nfail = nfail + 1
    ENDDO
    DEALLOCATE(kter, conv)
    IF (sth_timing) THEN
      t1 = wtime(); sth_t_drho = t1 - t0
      sth_t_total = sth_t_dpot + sth_t_ortho + sth_t_cg + sth_t_drho
    ENDIF
  END SUBROUTINE sth_engine_solve
  !
  SUBROUTINE sth_engine_xp_set(nc, slots, c, rec)
    !! for the next sth_engine_solve only: start from dpsi + sum_i c(i) history(slots(i)); after the
    !! solve store dpsi(end) - dpsi(start, before the extrapolation) in history slot rec (0 = do not)
    INTEGER, INTENT(IN) :: nc, slots(*), rec
    COMPLEX(dp), INTENT(IN) :: c(*)
    xp_nc = 0
    xp_rec = 0
    IF (sth_xp_m <= 0 .OR. .NOT. ALLOCATED(h_xh)) RETURN
    xp_nc = MIN(nc, sth_xp_m)
    xp_slot(1:xp_nc) = slots(1:xp_nc)
    xp_c(1:xp_nc) = c(1:xp_nc)
    xp_rec = rec
  END SUBROUTINE sth_engine_xp_set
  !
  SUBROUTINE sth_engine_becs(kqg, bec, alp)
    !! For drhodv, after the last solve of the irrep: bec(:, n, p, k) = vkb_kq^H dpsi_(k,p,n) and
    !! alp(:, n, ipol, p, k) = vkb_kq^H [ (k+q+G)_ipol dpsi_(k,p,n) ], as calbec in drhodv (before its
    !! i tpiba scaling). Bands n > nocc_k are zero. kqg(npwx, 3, nksq): k+q+G (rows > npwq unused).
    REAL(dp), INTENT(IN) :: kqg(npwx, 3, *)      ! (npwx, 3, nksq); not read when the per-q table is cached
    COMPLEX(dp), INTENT(OUT) :: bec(nkb, nbnd, npert, nksq), alp(nkb, nbnd, 3, npert, nksq)
    REAL(dp), DEVICE, ALLOCATABLE :: d_kqg(:,:,:)
    COMPLEX(dp), DEVICE, ALLOCATABLE :: d_b(:,:,:)
    COMPLEX(dp), ALLOCATABLE :: hb(:,:,:)
    INTEGER :: istat, ipol, ik, ip, n, s
    IF (kqg_q) THEN
      ALLOCATE(d_b(nkb, nsl, 4), hb(nkb, nsl, 4))
    ELSE
      ALLOCATE(d_kqg(npwx, 3, nksq), d_b(nkb, nsl, 4), hb(nkb, nsl, 4))
    ENDIF
    CALL sth_mem_sample()
    IF (.NOT. kqg_q) d_kqg = kqg(:, :, 1:nksq)
    istat = cudaDeviceSynchronize()
    DO ipol = 0, 3
      IF (ipol == 0) THEN
        istat = cublasZgemmStridedBatched(hblas, CUBLAS_OP_C, CUBLAS_OP_N, nkb, nslot * npert, npwx, &
                (1.0_dp, 0.0_dp), d_W, npwx, INT(npwx, 8) * nproj, d_dpsi, npwx, INT(npwx, 8) * nslot * npert, &
                (0.0_dp, 0.0_dp), d_b(1, 1, 1), nkb, INT(nkb, 8) * nslot * npert, nksq)
      ELSE
        IF (kqg_q) THEN
          CALL k_kqg_scale<<<4096, TPB, 0, strm>>>(d_t, d_dpsi, d_kqgq, ipol, npwx, nslot, npert, nksq, &
               INT(npwx, 8) * nsl)
        ELSE
          CALL k_kqg_scale<<<4096, TPB, 0, strm>>>(d_t, d_dpsi, d_kqg, ipol, npwx, nslot, npert, nksq, &
               INT(npwx, 8) * nsl)
        ENDIF
        CALL kerr('kqg_scale')
        istat = cublasZgemmStridedBatched(hblas, CUBLAS_OP_C, CUBLAS_OP_N, nkb, nslot * npert, npwx, &
                (1.0_dp, 0.0_dp), d_W, npwx, INT(npwx, 8) * nproj, d_t, npwx, INT(npwx, 8) * nslot * npert, &
                (0.0_dp, 0.0_dp), d_b(1, 1, 1 + ipol), nkb, INT(nkb, 8) * nslot * npert, nksq)
      ENDIF
      CALL chk(istat, 'cublasZgemmStridedBatched (becs)')
    ENDDO
    CALL tsync()
    hb = d_b
    bec = (0.0_dp, 0.0_dp)
    alp = (0.0_dp, 0.0_dp)
    DO ik = 1, nksq
      DO ip = 1, npert
        DO n = 1, h_nock(ik)
          s = slot(n, ip, ik)
          bec(:, n, ip, ik) = hb(:, s, 1)
          DO ipol = 1, 3
            alp(:, n, ipol, ip, ik) = hb(:, s, 1 + ipol)
          ENDDO
        ENDDO
      ENDDO
    ENDDO
    DEALLOCATE(d_b, hb)
    IF (ALLOCATED(d_kqg)) DEALLOCATE(d_kqg)
  END SUBROUTINE sth_engine_becs
  !
  SUBROUTINE sth_engine_get_dpsi(ik, ipert, dpsi)
    !! dpsi of (k, pert) after the last solve, (npwx, nbnd): bands > nocc_k and rows > npwq are zero
    INTEGER, INTENT(IN) :: ik, ipert
    COMPLEX(dp), INTENT(OUT) :: dpsi(npwx, nbnd)
    INTEGER :: n
    IF (.NOT. h_dpsi_valid) THEN
      h_dpsi = d_dpsi
      h_dpsi_valid = .TRUE.
    ENDIF
    dpsi = (0.0_dp, 0.0_dp)
    DO n = 1, h_nock(ik)
      dpsi(:, n) = h_dpsi(:, slot(n, ipert, ik))
    ENDDO
  END SUBROUTINE sth_engine_get_dpsi
  !
  SUBROUTINE sth_engine_get_hdiag(ik, hd)
    INTEGER, INTENT(IN) :: ik
    REAL(dp), INTENT(OUT) :: hd(npwx, nslot)
    hd = d_hd(:, :, ik)
  END SUBROUTINE sth_engine_get_hdiag
  !
  SUBROUTINE sth_engine_set_hdiag(ik, hd)
    !! override h_diag of k (rows 1..npwq, bands 1..nocc_k); for testing against a dump
    INTEGER, INTENT(IN) :: ik
    REAL(dp), INTENT(IN) :: hd(npwx, nbnd)
    REAL(dp) :: tmp(npwx, nslot)
    INTEGER :: n
    tmp = 0.0_dp
    DO n = 1, h_nock(ik)
      tmp(1:h_npwq(ik), n) = hd(1:h_npwq(ik), n)
    ENDDO
    d_hd(:, :, ik) = tmp
  END SUBROUTINE sth_engine_set_hdiag
  !
  SUBROUTINE sth_engine_free()
    !! release the per-q state (keeps the stream, cuBLAS handle and FFT plans)
    ready = .FALSE.
    staged = .FALSE.
    qready = .FALSE.
    CALL irrep_free()
    IF (ALLOCATED(h_npw)) DEALLOCATE(h_npw, h_npwq, h_nock, h_nockq, h_wk, h_idxk, h_idxkq, h_g2, &
                                     h_etk, h_wwg, h_wg1, h_etkq)
    IF (ALLOCATED(h_W)) DEALLOCATE(h_W)
    IF (ALLOCATED(h_evc)) DEALLOCATE(h_evc)
    IF (ALLOCATED(d_npw)) DEALLOCATE(d_npw, d_npwq, d_nock, d_nockq, d_idxk, d_idxkq, d_g2, d_etk, &
                                     d_hd, d_wwg, d_wg1, d_wgt)
    IF (ALLOCATED(d_W)) DEALLOCATE(d_W)
    IF (ALLOCATED(d_kqgq)) DEALLOCATE(d_kqgq)
    kqg_q = .FALSE.
    kqg_said = .FALSE.
    wdev_q = .FALSE.
    IF (ALLOCATED(d_box)) DEALLOCATE(d_box)
    IF (ALLOCATED(d_psir)) DEALLOCATE(d_psir)
    IF (ALLOCATED(d_box2)) DEALLOCATE(d_box2)
    IF (ALLOCATED(d_evc)) DEALLOCATE(d_evc)
    IF (ALLOCATED(d_inv)) DEALLOCATE(d_inv)
    IF (ALLOCATED(s_W)) DEALLOCATE(s_W, s_hd, s_g2, s_vrs, s_dmat, s_box)
    mp_q = .FALSE.
    IF (ALLOCATED(d_etkq)) DEALLOCATE(d_etkq)
    IF (ALLOCATED(h_xh)) DEALLOCATE(h_xh)
    CALL pfft_free()
    CALL lift_free()
    CALL cbsr_free()
    CALL pfs_free()
    IF (fdx_on) CALL fdx_free()
    fdx_on = .FALSE.
    IF (ALLOCATED(d_db)) DEALLOCATE(d_db, d_da)
    IF (ALLOCATED(h_bp)) DEALLOCATE(h_bp, h_al)
    dvb_q = .FALSE.
    CALL sth_mem_trim()
  END SUBROUTINE sth_engine_free
  !
  SUBROUTINE sth_engine_shutdown()
    INTEGER :: istat
    CALL sth_engine_free()
    CALL cbsr_release()
    IF (ALLOCATED(d_vrs)) DEALLOCATE(d_vrs, d_dmat)
    CALL destroy_plans()
    IF (have_handles) THEN
      istat = cublasDestroy(hblas)
      istat = cudaStreamDestroy(strm)
      DEALLOCATE(h_nact)
      have_handles = .FALSE.
    ENDIF
  END SUBROUTINE sth_engine_shutdown
  !
  !=================================================================================================
  ! orthogonalize weights (lgauss branch of orthogonalize.f90), and wgauss / w0gauss (Modules/)
  !=================================================================================================
  !
  SUBROUTINE sth_weights(nbnd_, nock, nockq, etk, etkq, ef, degauss, ngauss, alpha, wwg, wg1)
    INTEGER, INTENT(IN) :: nbnd_, nock, nockq, ngauss
    REAL(dp), INTENT(IN) :: etk(nbnd_), etkq(nbnd_), ef, degauss, alpha
    REAL(dp), INTENT(OUT) :: wwg(nbnd_, nock), wg1(nock)
    INTEGER :: ibnd, jbnd
    REAL(dp) :: w0g, wgp, deltae, theta
    DO ibnd = 1, nock
      wg1(ibnd) = wgauss_((ef - etk(ibnd)) / degauss, ngauss)
      w0g = w0gauss_((ef - etk(ibnd)) / degauss, ngauss) / degauss
      DO jbnd = 1, nbnd_
        wgp = wgauss_((ef - etkq(jbnd)) / degauss, ngauss)
        deltae = etkq(jbnd) - etk(ibnd)
        theta = wgauss_(deltae / degauss, 0)
        wwg(jbnd, ibnd) = wg1(ibnd) * (1.d0 - theta) + wgp * theta
        IF (jbnd <= nockq) THEN
          IF (ABS(deltae) > 1.0d-5) THEN
            wwg(jbnd, ibnd) = wwg(jbnd, ibnd) + alpha * theta * (wgp - wg1(ibnd)) / deltae
          ELSE
            wwg(jbnd, ibnd) = wwg(jbnd, ibnd) - alpha * theta * w0g
          ENDIF
        ENDIF
      ENDDO
    ENDDO
  END SUBROUTINE sth_weights
  !
  REAL(dp) FUNCTION wgauss_(x, n)
    REAL(dp), INTENT(IN) :: x
    INTEGER, INTENT(IN) :: n
    REAL(dp), PARAMETER :: pi = 3.14159265358979323846_dp, maxarg = 200.d0
    REAL(dp) :: a, hp, arg, hd, xp
    INTEGER :: i, ni
    IF (n == -99) THEN
      IF (x < -maxarg) THEN
        wgauss_ = 0.d0
      ELSEIF (x > maxarg) THEN
        wgauss_ = 1.d0
      ELSE
        wgauss_ = 1.0d0 / (1.0d0 + EXP(-x))
      ENDIF
      RETURN
    ENDIF
    IF (n == -1) THEN
      xp = x - 1.0d0 / SQRT(2.0d0)
      arg = MIN(maxarg, xp**2)
      wgauss_ = 0.5d0 * ERF(xp) + 1.0d0 / SQRT(2.0d0 * pi) * EXP(-arg) + 0.5d0
      RETURN
    ENDIF
    arg = -x
    IF (arg < SQRT(maxarg)) THEN
      wgauss_ = 0.5_dp * ERFC(arg)
    ELSE
      wgauss_ = 0._dp
    ENDIF
    IF (n == 0) RETURN
    hd = 0.d0
    arg = MIN(maxarg, x**2)
    hp = EXP(-arg)
    ni = 0
    a = 1.d0 / SQRT(pi)
    DO i = 1, n
      hd = 2.0d0 * x * hp - 2.0d0 * DBLE(ni) * hd
      ni = ni + 1
      a = -a / (DBLE(i) * 4.0d0)
      wgauss_ = wgauss_ - a * hd
      hp = 2.0d0 * x * hd - 2.0d0 * DBLE(ni) * hp
      ni = ni + 1
    ENDDO
  END FUNCTION wgauss_
  !
  REAL(dp) FUNCTION w0gauss_(x, n)
    REAL(dp), INTENT(IN) :: x
    INTEGER, INTENT(IN) :: n
    REAL(dp), PARAMETER :: sqrtpm1 = 1.0_dp / 1.77245385090551602729_dp
    REAL(dp) :: a, arg, hp, hd
    INTEGER :: i, ni
    IF (n == -99) THEN
      IF (ABS(x) <= 36.0) THEN
        w0gauss_ = 1.0d0 / (2.0d0 + EXP(-x) + EXP(+x))
      ELSE
        w0gauss_ = 0.d0
      ENDIF
      RETURN
    ENDIF
    IF (n == -1) THEN
      arg = MIN(200.d0, (x - 1.0d0 / SQRT(2.0d0))**2)
      w0gauss_ = sqrtpm1 * EXP(-arg) * (2.0d0 - SQRT(2.0d0) * x)
      RETURN
    ENDIF
    arg = MIN(200.d0, x**2)
    w0gauss_ = EXP(-arg) * sqrtpm1
    IF (n == 0) RETURN
    hd = 0.0d0
    hp = EXP(-arg)
    ni = 0
    a = sqrtpm1
    DO i = 1, n
      hd = 2.0d0 * x * hp - 2.0d0 * DBLE(ni) * hd
      ni = ni + 1
      a = -a / (DBLE(i) * 4.0d0)
      hp = 2.0d0 * x * hd - 2.0d0 * DBLE(ni) * hp
      ni = ni + 1
      w0gauss_ = w0gauss_ + a * hp
    ENDDO
  END FUNCTION w0gauss_
  !
END MODULE sth_engine
