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
  IMPLICIT NONE
  PRIVATE
  !
  INTEGER, PARAMETER, PUBLIC :: sth_dp = SELECTED_REAL_KIND(14, 200)
  INTEGER, PARAMETER :: dp = sth_dp
  INTEGER, PARAMETER :: TPB = 256      ! threads per block, element-wise kernels
  INTEGER, PARAMETER :: RB = 256       ! max threads per block, per-column reductions (power of 2)
  INTEGER, PARAMETER :: MAXPERT = 6
  INTEGER, PARAMETER :: MAXPLAN = 12   ! FFT batch sizes 2**0 .. 2**(MAXPLAN-1)
  INTEGER, PARAMETER :: maxter = 400   ! as cgsolve_all
  !
  PUBLIC :: sth_engine_init, sth_engine_set_global, sth_engine_set_k, sth_engine_set_dvbare, &
            sth_engine_set_dpsi, sth_engine_setup, sth_engine_solve, sth_engine_get_dpsi, &
            sth_engine_get_hdiag, sth_engine_set_hdiag, sth_engine_reset_dpsi, sth_engine_free, sth_engine_shutdown, &
            sth_weights, sth_engine_ready, sth_engine_irrep, sth_engine_irrep_commit, sth_engine_qready, &
            sth_engine_becs, sth_mem_trim, sth_mem_sample, sth_mem_take_hw
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
  INTEGER(8), SAVE :: mem_hw = 0
  !
  ! run statistics of the last solve (seconds; timers need sth_timing = .true.)
  LOGICAL, PUBLIC, SAVE :: sth_timing = .FALSE.
  REAL(dp), PUBLIC, SAVE :: sth_t_dpot = 0, sth_t_ortho = 0, sth_t_cg = 0, sth_t_drho = 0, &
                            sth_t_total = 0, sth_t_setup = 0
  INTEGER, PUBLIC, SAVE :: sth_nsteps = 0
  INTEGER(8), PUBLIC, SAVE :: sth_ncolapp = 0      ! column-operator applications in the CG
  INTEGER, PUBLIC, SAVE :: sth_bmax = 256          ! FFT chunk (largest batch); set before init
  INTEGER, PUBLIC, SAVE :: sth_debug_slot = 0      ! >0: print the CG history of this slot
  INTEGER, PUBLIC, SAVE :: sth_opvar = 0           ! testing: 1 = (g2 - e) x + Vx (same operator, other rounding)
  INTEGER, PUBLIC, SAVE :: sth_rb = 128            ! threads per column reduction (power of 2, <= 256)
  !
  LOGICAL, SAVE :: ready = .FALSE., staged = .FALSE., qready = .FALSE.
  INTEGER, SAVE :: nksq, npwx, nbnd, nkb, npert, nslot, nproj, nsl, ngrp, nnr, nvalid, nbox
  INTEGER, SAVE :: nr(3), nrx(3)
  REAL(dp), SAVE :: alpha_pv, omega
  !
  ! host staging (filled by set_* calls, uploaded by setup)
  INTEGER, ALLOCATABLE, SAVE :: h_npw(:), h_npwq(:), h_nock(:), h_nockq(:), h_idxk(:,:), h_idxkq(:,:)
  REAL(dp), ALLOCATABLE, SAVE :: h_wk(:), h_g2(:,:), h_etk(:,:), h_wwg(:,:,:), h_wg1(:,:)
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
                                        cnt, ethr, first)
    ! cgsolve_all, per active column: h = h_diag*g ; rho = <h|g> ; converged if sqrt(rho) < ethr;
    ! otherwise h = -h + (rho/rhoold) hold  (hold is the previous h, kept in place).
    ! cnt(group) counts the columns that were active at this test (kter_eff).
    INTEGER, VALUE :: nslot, npert, npwx, nksq, first
    REAL(dp), VALUE :: ethr
    INTEGER :: list(*), npwq(nksq), conv(*), cnt(*)
    REAL(dp) :: hd(npwx, nslot, nksq), rho(*)
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
      IF (SQRT(rn) < ethr) THEN
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
  ATTRIBUTES(GLOBAL) SUBROUTINE k_dvpsir(box, psir, dv, list, s0, nnr, nslot, npert)
    ! apply_dpot: box(:, j) = dvscf(:, p) * psi_kn(r)
    INTEGER, VALUE :: s0, nnr, nslot, npert
    COMPLEX(dp) :: box(nnr, *), psir(nnr, *), dv(nnr, *)
    INTEGER :: list(*)
    INTEGER :: j, s, k, n, p, b, ir
    j = blockIdx%y
    s = list(s0 + j)
    k = (s - 1) / (nslot * npert) + 1
    n = MOD(s - 1, nslot) + 1
    p = MOD((s - 1) / nslot, npert) + 1
    b = n + nslot * (k - 1)
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
  ATTRIBUTES(GLOBAL) SUBROUTINE k_drho(drho, box, psir, list, s0, nb, nnr, nslot, npert, wgt)
    ! incdrhoscf: drho(r, p) += sum_j wgt(k_j) conj(psi_{k_j n_j}(r)) dpsi_j(r); deterministic order
    INTEGER, VALUE :: s0, nb, nnr, nslot, npert
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
  SUBROUTINE apply_op(x, y, list, nl)
    ! y = (H_kq - e_n + alpha_pv P_v) x on the nl columns list(1:nl); other slots of y are garbage
    COMPLEX(dp), DEVICE :: x(npwx, *), y(npwx, *)
    INTEGER, DEVICE :: list(*)
    INTEGER, INTENT(IN) :: nl
    INTEGER :: s0, nb, ip, np
    REAL(dp) :: scale
    scale = 1.0_dp / REAL(nr(1) * nr(2) * nr(3), dp)
    CALL gemm_proj(x, d_cf)
    CALL k_coef_h<<<nsl, 32 * ((nproj + 31) / 32), 0, strm>>>(d_cf2, d_cf, d_dmat, nproj, nkb, nslot, &
         npert, nksq, d_nockq, alpha_pv)
    CALL kerr('coef_h')
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
    CALL gemm_back(d_cf2, y, (1.0_dp, 0.0_dp))
  END SUBROUTINE apply_op
  !
  !=================================================================================================
  ! public API
  !=================================================================================================
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
    ALLOCATE(h_idxk(npwx, nksq), h_idxkq(npwx, nksq), h_g2(npwx, nksq), h_etk(nbnd, nksq))
    ALLOCATE(h_wwg(nbnd, nslot, nksq), h_wg1(nslot, nksq))
    ALLOCATE(h_W(npwx, nproj, nksq), h_evc(npwx, nbox))
    h_npw = 0; h_npwq = 0; h_nock = 0; h_nockq = 0; h_wk = 0
    h_idxk = 1; h_idxkq = 1; h_g2 = 0; h_etk = 0; h_wwg = 0; h_wg1 = 0
    h_W = (0.0_dp, 0.0_dp); h_evc = (0.0_dp, 0.0_dp)
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
                              vkb, wwg, wg1)
    !! Per-k data. idx_k / idx_kq: 1-based FFT linear indices of k+G / k+q+G; evc, evq (npwx, nbnd);
    !! et_k (nbnd); g2kin (npwx) at k+q; vkb (npwx, nkb) at k+q; wwg (nbnd, nock), wg1 (nock).
    INTEGER, INTENT(IN) :: ik, npw, npwq, nock, nockq
    REAL(dp), INTENT(IN) :: wk
    INTEGER, INTENT(IN) :: idx_k(npw), idx_kq(npwq)
    COMPLEX(dp), INTENT(IN) :: evc(npwx, nbnd), evq(npwx, nbnd), vkb(npwx, nkb)
    REAL(dp), INTENT(IN) :: et_k(nbnd), g2kin(npwx), wwg(nbnd, nock), wg1(nock)
    INTEGER :: n
    h_npw(ik) = npw; h_npwq(ik) = npwq; h_nock(ik) = nock; h_nockq(ik) = nockq; h_wk(ik) = wk
    h_idxk(1:npw, ik) = idx_k(1:npw)
    h_idxkq(1:npwq, ik) = idx_kq(1:npwq)
    h_g2(1:npwq, ik) = g2kin(1:npwq)
    h_etk(:, ik) = et_k(:)
    h_wwg(:, 1:nock, ik) = wwg(:, 1:nock)
    h_wg1(1:nock, ik) = wg1(1:nock)
    h_W(1:npwq, 1:nkb, ik) = vkb(1:npwq, 1:nkb)
    h_W(1:npwq, nkb + 1:nkb + nbnd, ik) = evq(1:npwq, 1:nbnd)
    DO n = 1, nock
      h_evc(1:npw, n + nslot * (ik - 1)) = evc(1:npw, n)
    ENDDO
  END SUBROUTINE sth_engine_set_k
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
    nq = h_npwq(ik)
    DO n = 1, h_nock(ik)
      h_dpsi0(1:nq, slot(n, ipert, ik)) = dpsi(1:nq, n)
    ENDDO
    have_dpsi0 = .TRUE.
  END SUBROUTINE sth_engine_set_dpsi
  !
  SUBROUTINE sth_engine_setup(ierr)
    !! Upload the staged data, build h_diag, psi_k(r) and the cuFFT plans.
    INTEGER, INTENT(OUT) :: ierr
    INTEGER :: istat, ik, n, ip, nb, b0
    INTEGER(8) :: need, freeb, totb
    REAL(dp) :: t0
    t0 = wtime()
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
    need = 16_8 * npwx * nsl * 7 + 16_8 * nnr * nbox + 16_8 * nnr * sth_bmax * 3 + 16_8 * npwx * nproj * nksq + 4_8 * nnr * nksq
    ! sth_mp: FP32 W, h_diag, g2kin, box (+ C2C plan work areas) and the irrep's FP32 column arrays
    IF (sth_mp) need = need + 8_8 * npwx * nproj * nksq + 4_8 * npwx * (nslot + 1) * nksq + 8_8 * nnr * sth_bmax * 3 &
                            + (8_8 * npwx * 4 + 16_8 * nproj + 12_8) * nsl
    CALL sth_mem_trim()
    istat = cudaMemGetInfo(freeb, totb)
    IF (need > freeb - 1024_8**3) THEN
      WRITE(*, '(A,F8.2,A,F8.2,A)') 'sth_engine: needs ', REAL(need) / 1024.0**3, ' GiB, free ', &
           REAL(freeb) / 1024.0**3, ' GiB'
      ierr = 2
      RETURN
    ENDIF
    !
    ALLOCATE(d_npw(nksq), d_npwq(nksq), d_nock(nksq), d_nockq(nksq), d_idxk(npwx, nksq), d_idxkq(npwx, nksq))
    ALLOCATE(d_g2(npwx, nksq), d_etk(nbnd, nksq), d_hd(npwx, nslot, nksq), d_wwg(nbnd, nslot, nksq))
    ALLOCATE(d_wg1(nslot, nksq), d_wgt(nksq), d_W(npwx, nproj, nksq))
    d_npw = h_npw; d_npwq = h_npwq; d_nock = h_nock; d_nockq = h_nockq
    d_idxk = h_idxk; d_idxkq = h_idxkq; d_g2 = h_g2; d_etk = h_etk; d_wwg = h_wwg; d_wg1 = h_wg1
    d_wgt = 2.0_dp * h_wk / omega
    d_W = h_W
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
    !
    ALLOCATE(d_box(nnr, sth_bmax))
    ALLOCATE(d_psir(nnr, nbox))
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
    ! psi_k(r) for n <= nocc_k: scatter evc at k+G, inverse FFT in place
    CALL zero_c(d_psir, INT(nnr, 8) * nbox)
      ALLOCATE(d_evc(npwx, nbox))
      d_evc = h_evc
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
    DEALLOCATE(h_evc, h_W)
    istat = cudaDeviceSynchronize()
    CALL chk(istat, 'setup')
    qready = .TRUE.
    CALL sth_engine_irrep_commit(ierr)
    sth_t_setup = wtime() - t0
  END SUBROUTINE sth_engine_setup
  !
  SUBROUTINE irrep_free()
    IF (ALLOCATED(h_dvb)) DEALLOCATE(h_dvb, h_dpsi0)
    IF (ALLOCATED(h_dpsi)) DEALLOCATE(h_dpsi)
    IF (ALLOCATED(h_valid)) DEALLOCATE(h_valid)
    IF (ALLOCATED(d_dvb)) DEALLOCATE(d_dvb, d_d0, d_dpsi, d_g, d_h, d_t, d_cf, d_cf2, d_ps, d_rho, d_conv, &
                                     d_cnt, d_kter, d_nact, d_av, d_cv, d_dv, d_drho)
    IF (ALLOCATED(d_valid)) DEALLOCATE(d_valid, d_act, d_act2)
    CALL irrep_free_mp()
  END SUBROUTINE irrep_free
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
    ALLOCATE(h_dvb(npwx, nsl), h_dpsi0(npwx, nsl))
    h_dvb = (0.0_dp, 0.0_dp)
    h_dpsi0 = (0.0_dp, 0.0_dp)
    have_dpsi0 = .FALSE.
  END SUBROUTINE irrep_stage
  !
  SUBROUTINE sth_engine_irrep(npert_, ierr)
    !! Start a new irrep on the per-q state already set up: keeps k/k+q data, W, h_diag, psi(r) and
    !! the plans; then set_dvbare (and set_dpsi for a warm start) and sth_engine_irrep_commit.
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
    CALL irrep_free()
    CALL irrep_stage(npert_)
  END SUBROUTINE sth_engine_irrep
  !
  SUBROUTINE sth_engine_irrep_commit(ierr)
    !! upload the staged dV_bare psi (and warm start) of the current irrep; allocate its work arrays
    INTEGER, INTENT(OUT) :: ierr
    INTEGER :: istat, ik, n, ip
    INTEGER(8) :: need, freeb, totb
    ierr = 0
    IF (ALLOCATED(d_dvb)) DEALLOCATE(d_dvb, d_d0, d_dpsi, d_g, d_h, d_t, d_cf, d_cf2, d_ps, d_rho, d_conv, &
                                     d_cnt, d_kter, d_nact, d_av, d_cv, d_dv, d_drho)
    IF (ALLOCATED(d_valid)) DEALLOCATE(d_valid, d_act, d_act2)
    IF (ALLOCATED(h_valid)) DEALLOCATE(h_valid)
    IF (ALLOCATED(h_dpsi)) DEALLOCATE(h_dpsi)
    CALL irrep_free_mp()
    need = 16_8 * npwx * nsl * 7
    IF (mp_q) need = need + (8_8 * npwx * 4 + 16_8 * nproj + 12_8) * nsl
    CALL sth_mem_trim()
    istat = cudaMemGetInfo(freeb, totb)
    IF (need > freeb - 512_8 * 1024**2) THEN
      WRITE(*, '(A,F8.2,A,F8.2,A)') 'sth_engine: irrep needs ', REAL(need) / 1024.0**3, ' GiB, free ', &
           REAL(freeb) / 1024.0**3, ' GiB'
      ierr = 2
      RETURN
    ENDIF
    ALLOCATE(d_dvb(npwx, nsl), d_d0(npwx, nsl), d_dpsi(npwx, nsl), d_g(npwx, nsl), d_h(npwx, nsl))
    ALLOCATE(d_t(npwx, nsl), d_cf(nproj, nsl), d_cf2(nproj, nsl), d_ps(nbnd, nsl))
    ALLOCATE(d_rho(nsl), d_conv(nsl), d_cnt(ngrp), d_kter(ngrp), d_nact(1), d_av(nsl), d_cv(nsl))
    ALLOCATE(d_dv(nnr, npert), d_drho(nnr, npert))
    ALLOCATE(h_dpsi(npwx, nsl))
    h_dpsi_valid = .FALSE.
    d_dvb = h_dvb
    d_dpsi = h_dpsi0
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
    istat = cudaDeviceSynchronize()
    ! zero the work arrays (invalid slots must stay zero; stale values must be finite)
    CALL zero_c(d_d0, INT(npwx, 8) * nsl)
    CALL zero_c(d_g, INT(npwx, 8) * nsl)
    CALL zero_c(d_h, INT(npwx, 8) * nsl)
    CALL zero_c(d_t, INT(npwx, 8) * nsl)
    CALL zero_c(d_cf, INT(nproj, 8) * nsl)
    CALL zero_c(d_cf2, INT(nproj, 8) * nsl)
    IF (mp_q) THEN
      ALLOCATE(s_e(npwx, nsl), s_g(npwx, nsl), s_h(npwx, nsl), s_t(npwx, nsl), s_cf(nproj, nsl), s_cf2(nproj, nsl))
      ALLOCATE(d_tol(nsl), d_iconv(nsl), d_oact(nsl))
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
    ready = .TRUE.
  END SUBROUTINE sth_engine_irrep_commit
  !
  SUBROUTINE sth_engine_reset_dpsi()
    !! restore the staged warm start (for repeated timing runs of the same call)
    d_dpsi = h_dpsi0
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
    INTEGER :: istat, iter, nact, s0, nb, ip, np, first, i, k
    REAL(dp) :: scale, t0, t1
    REAL(dp), ALLOCATABLE :: kter(:)
    INTEGER, ALLOCATABLE :: conv(:)
    !
    t0 = wtime()
    istat = cudaGetLastError()   ! see sth_engine_setup
    scale = 1.0_dp / REAL(nr(1) * nr(2) * nr(3), dp)
    h_dpsi_valid = .FALSE.
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
        nb = MIN(sth_bmax, nvalid - s0)
        ip = plan_for(nb)
        nb = 2**ip
        CALL k_dvpsir<<<dim3(nblk(nnr, TPB), nb, 1), TPB, 0, strm>>>(d_box, d_psir, d_dv, d_valid, s0, &
             nnr, nslot, npert)
        CALL kerr('dvpsir')
        CALL fft_exec(ip, d_box, CUFFT_FORWARD)
        CALL k_gather_add<<<dim3(nblk(npwx, TPB), nb, 1), TPB, 0, strm>>>(d_d0, d_box, d_valid, s0, nnr, &
             npwx, nslot, npert, nksq, d_npwq, d_idxkq, scale)
        CALL kerr('gather_add')
        s0 = s0 + nb
      ENDDO
    ENDIF
    IF (sth_timing) THEN
      CALL tsync(); t1 = wtime(); sth_t_dpot = t1 - t0; t0 = t1
    ENDIF
    !
    ! orthogonalize (lgauss): ps = evq^H dvpsi ; ps *= wwg ; dvpsi *= wg1 ; dvpsi = evq ps - dvpsi.
    ! One ZGEMM pair per (k, pert) with exactly the shapes of orthogonalize.f90 (m = nbnd, n = nocc_k,
    ! k = npwq), so that the right-hand side matches vanilla's rounding as closely as possible: the
    ! screened, projected dvpsi can be a strong cancellation, and its rounding feeds every CG step.
    CALL ortho_exact()
    IF (sth_timing) THEN
      CALL tsync(); t1 = wtime(); sth_t_ortho = t1 - t0; t0 = t1
    ENDIF
    !
    ! CG (cgsolve_all), all columns at once
    IF (mp_q) THEN
      CALL mp_solve(first_iter, thresh)
    ELSE
    ax_valid = .FALSE.
    IF (first_iter) CALL zero_c(d_dpsi, INT(npwx, 8) * nsl)
    d_conv = 0
    d_cnt = 0
    d_kter = 0.0_dp
    d_rho = 0.0_dp
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
      CALL k_check<<<nact, sth_rb, 0, strm>>>(d_act, nslot, npert, npwx, nksq, d_npwq, d_hd, d_g, d_h, d_rho, &
           d_conv, d_cnt, thresh, first)
      CALL kerr('check')
      CALL k_kter<<<nblk(ngrp, TPB), TPB, 0, strm>>>(ngrp, npert, nksq, d_nock, d_cnt, d_kter)
      CALL kerr('kter')
      CALL k_compact<<<1, 1024, 0, strm>>>(d_act, d_act2, nact, d_conv, d_nact)
      CALL kerr('compact')
      istat = cudaMemcpyAsync(h_nact, d_nact, 1, strm)
      CALL tsync()
      nact = h_nact(1)
      sth_nsteps = iter
      IF (nact == 0) EXIT
      CALL apply_op(d_h, d_t, d_act, nact)
      sth_ncolapp = sth_ncolapp + nact
      CALL k_update<<<nact, sth_rb, 0, strm>>>(d_act, nslot, npert, npwx, nksq, d_npwq, d_h, d_g, d_t, d_dpsi, d_av, d_cv)
      CALL kerr('update')
      IF (sth_debug_slot > 0) CALL debug_step(iter)
    ENDDO
    sth_n64 = sth_ncolapp; sth_n32 = 0; sth_nround = 0
    ENDIF
    IF (sth_timing) THEN
      CALL tsync(); t1 = wtime(); sth_t_cg = t1 - t0; t0 = t1
    ENDIF
    !
    ! incdrhoscf
    s0 = 0
    DO WHILE (s0 < nvalid)
      nb = MIN(sth_bmax, nvalid - s0)
      ip = plan_for(nb)
      nb = 2**ip
      np = 2**ip
      CALL k_scatter_full<<<dim3(nblk(nnr, TPB), nb, 1), TPB, 0, strm>>>(d_box, d_dpsi, d_valid, s0, nnr, &
           npwx, nslot, npert, nksq, d_inv)
      CALL kerr('scatter (drho)')
      CALL fft_exec(ip, d_box, CUFFT_INVERSE)
      CALL k_drho<<<nblk(nnr, TPB), TPB, 0, strm>>>(d_drho, d_box, d_psir, d_valid, s0, nb, nnr, nslot, &
           npert, d_wgt)
      CALL kerr('drho')
      s0 = s0 + nb
    ENDDO
    CALL tsync()
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
  SUBROUTINE sth_engine_becs(kqg, bec, alp)
    !! For drhodv, after the last solve of the irrep: bec(:, n, p, k) = vkb_kq^H dpsi_(k,p,n) and
    !! alp(:, n, ipol, p, k) = vkb_kq^H [ (k+q+G)_ipol dpsi_(k,p,n) ], as calbec in drhodv (before its
    !! i tpiba scaling). Bands n > nocc_k are zero. kqg(npwx, 3, nksq): k+q+G (rows > npwq unused).
    REAL(dp), INTENT(IN) :: kqg(npwx, 3, nksq)
    COMPLEX(dp), INTENT(OUT) :: bec(nkb, nbnd, npert, nksq), alp(nkb, nbnd, 3, npert, nksq)
    REAL(dp), DEVICE, ALLOCATABLE :: d_kqg(:,:,:)
    COMPLEX(dp), DEVICE, ALLOCATABLE :: d_b(:,:,:)
    COMPLEX(dp), ALLOCATABLE :: hb(:,:,:)
    INTEGER :: istat, ipol, ik, ip, n, s
    ALLOCATE(d_kqg(npwx, 3, nksq), d_b(nkb, nsl, 4), hb(nkb, nsl, 4))
    CALL sth_mem_sample()
    d_kqg = kqg
    istat = cudaDeviceSynchronize()
    DO ipol = 0, 3
      IF (ipol == 0) THEN
        istat = cublasZgemmStridedBatched(hblas, CUBLAS_OP_C, CUBLAS_OP_N, nkb, nslot * npert, npwx, &
                (1.0_dp, 0.0_dp), d_W, npwx, INT(npwx, 8) * nproj, d_dpsi, npwx, INT(npwx, 8) * nslot * npert, &
                (0.0_dp, 0.0_dp), d_b(1, 1, 1), nkb, INT(nkb, 8) * nslot * npert, nksq)
      ELSE
        CALL k_kqg_scale<<<4096, TPB, 0, strm>>>(d_t, d_dpsi, d_kqg, ipol, npwx, nslot, npert, nksq, &
             INT(npwx, 8) * nsl)
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
    DEALLOCATE(d_kqg, d_b, hb)
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
                                     h_etk, h_wwg, h_wg1)
    IF (ALLOCATED(h_W)) DEALLOCATE(h_W)
    IF (ALLOCATED(h_evc)) DEALLOCATE(h_evc)
    IF (ALLOCATED(d_npw)) DEALLOCATE(d_npw, d_npwq, d_nock, d_nockq, d_idxk, d_idxkq, d_g2, d_etk, &
                                     d_hd, d_wwg, d_wg1, d_wgt, d_W)
    IF (ALLOCATED(d_box)) DEALLOCATE(d_box, d_psir)
    IF (ALLOCATED(d_inv)) DEALLOCATE(d_inv)
    IF (ALLOCATED(s_W)) DEALLOCATE(s_W, s_hd, s_g2, s_vrs, s_dmat, s_box)
    mp_q = .FALSE.
    CALL sth_mem_trim()
  END SUBROUTINE sth_engine_free
  !
  SUBROUTINE sth_engine_shutdown()
    INTEGER :: istat
    CALL sth_engine_free()
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
