!
! rtsc: batched block-Davidson for the ph.x nscf (groups/compute/STH-LOG.md, Session 2).
!
! All k-points of the nscf (or a chunk of them) are diagonalized together on the GPU with the
! algorithm of cegterg (KS_Solvers/Davidson/cegterg.f90, norm-conserving: S = 1):
!   basis psi(:, 1:nbase), hpsi = H psi; reduced hc = psi^H hpsi, sc = psi^H psi; Ritz pairs (ew, vc)
!   of the lowest nbnd; for the bands not converged, correction vectors
!     t = g_psi( hpsi vc - ew psi vc ),  g_psi: divide by (h_diag - ew), |.| >= 1e-4,
!   normalized and appended; convergence |ew - e_previous| < ethr for every band;
!   restart (psi <- psi vc, hpsi <- hpsi vc) when the basis would exceed nvx = david * nbnd.
! The differences from cegterg are batching only: the H application runs over the new columns of all
! k at once (batched cuFFT, cuBLAS), the reduced matrices are rebuilt in full every step
! (strided-batched ZGEMM over k), and the small generalized eigenproblems are solved on the host
! (LAPACK zhegv, OpenMP over k). k-points that fail (zhegv error or no convergence within maxit)
! are reported to the caller, which runs vanilla's Davidson for them.
!
! H x = g2kin x + Vloc x (FFT on the smooth grid) + vkb D vkb^H x, as h_psi for NC pseudopotentials.
! No QE modules: plain arrays in, plain arrays out.
!
MODULE nscf_engine
  USE cudafor
  USE cufft
  USE cublas_v2
  IMPLICIT NONE
  PRIVATE
  INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(14, 200)
  INTEGER, PARAMETER :: TPB = 256, RB = 128, MAXPLAN = 12
  !
  PUBLIC :: nscf_eng_init, nscf_eng_set_k, nscf_eng_solve, nscf_eng_get_k, nscf_eng_free, &
            nscf_eng_kmax
  ! statistics of the last solve: iterations, wall time in the host eigensolver and in total (s)
  INTEGER, PUBLIC, SAVE :: nscf_eng_nit = 0
  REAL(dp), PUBLIC, SAVE :: nscf_eng_t_eig = 0, nscf_eng_t_tot = 0
  !
  INTEGER, SAVE :: nk = 0, npwx, nbnd, nkb, nvx, nnr, nr(3), nrx(3)
  REAL(dp), SAVE :: v0
  INTEGER, ALLOCATABLE, SAVE :: h_npw(:), h_idx(:,:)
  REAL(dp), ALLOCATABLE, SAVE :: h_g2(:,:), h_e(:,:)
  COMPLEX(dp), ALLOCATABLE, SAVE :: h_vkb(:,:,:), h_psi0(:,:,:), h_evc(:,:,:)
  LOGICAL, ALLOCATABLE, SAVE :: h_ok(:)
  INTEGER, ALLOCATABLE, SAVE :: h_iter(:)
  !
  INTEGER, DEVICE, ALLOCATABLE :: d_npw(:), d_inv(:,:), d_idx(:,:), d_list(:), d_off(:), d_ncol(:), d_flag(:)
  REAL(dp), DEVICE, ALLOCATABLE :: d_g2(:,:), d_hd(:,:), d_vrs(:), d_ewc(:,:), d_nrm(:,:)
  COMPLEX(dp), DEVICE, ALLOCATABLE :: d_vkb(:,:,:), d_dmat(:,:), d_psi(:,:,:), d_hpsi(:,:,:), d_new(:,:,:)
  COMPLEX(dp), DEVICE, ALLOCATABLE :: d_hnew(:,:,:), d_hc(:,:,:), d_sc(:,:,:), d_A(:,:,:), d_B(:,:,:)
  COMPLEX(dp), DEVICE, ALLOCATABLE :: d_cf(:,:,:), d_cf2(:,:,:), d_t1(:,:,:), d_t2(:,:,:), d_box(:,:)
  !
  INTEGER(KIND=cuda_stream_kind), SAVE :: strm
  TYPE(cublasHandle), SAVE :: hb
  LOGICAL, SAVE :: have_h = .FALSE.
  INTEGER, SAVE :: plans(0:MAXPLAN-1) = 0, nplan = 0, bmax = 256
  !
CONTAINS
  !
  ! ---------------------------------------------------------------------------------------------
  ! kernels
  ! ---------------------------------------------------------------------------------------------
  ATTRIBUTES(GLOBAL) SUBROUTINE k_scat(box, x, list, s0, nnr, npwx, nbnd, nk, inv)
    INTEGER, VALUE :: s0, nnr, npwx, nbnd, nk
    COMPLEX(dp) :: box(nnr, *), x(npwx, *)
    INTEGER :: list(*), inv(nnr, nk)
    INTEGER :: j, c, k, ir, ig
    j = blockIdx%y
    c = list(s0 + j)
    k = (c - 1) / nbnd + 1
    ir = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ir <= nnr) THEN
      ig = inv(ir, k)
      IF (ig > 0) THEN
        box(ir, j) = x(ig, c)
      ELSE
        box(ir, j) = (0.0_dp, 0.0_dp)
      ENDIF
    ENDIF
  END SUBROUTINE k_scat
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
  ATTRIBUTES(GLOBAL) SUBROUTINE k_gath(y, x, box, list, s0, nnr, npwx, nbnd, nk, npw, idx, g2, scale)
    ! y = g2kin x + (Vloc x) for G <= npw; rows > npw zero (the nonlocal term is added by ZGEMM)
    INTEGER, VALUE :: s0, nnr, npwx, nbnd, nk
    REAL(dp), VALUE :: scale
    COMPLEX(dp) :: y(npwx, *), x(npwx, *), box(nnr, *)
    INTEGER :: list(*), npw(nk), idx(npwx, nk)
    REAL(dp) :: g2(npwx, nk)
    INTEGER :: j, c, k, ig
    j = blockIdx%y
    c = list(s0 + j)
    k = (c - 1) / nbnd + 1
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ig <= npwx) THEN
      IF (ig <= npw(k)) THEN
        y(ig, c) = g2(ig, k) * x(ig, c) + box(idx(ig, k), j) * scale
      ELSE
        y(ig, c) = (0.0_dp, 0.0_dp)
      ENDIF
    ENDIF
  END SUBROUTINE k_gath
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_hdiag(hd, vkb, dmat, g2, v0, npw, npwx, nkb, nk)
    ! usnldiag (NC, collinear): h_diag = g2kin + v_of_0 + sum_ij D_ij Re(vkb_i conj(vkb_j))
    INTEGER, VALUE :: npwx, nkb, nk
    REAL(dp), VALUE :: v0
    REAL(dp) :: hd(npwx, nk), g2(npwx, nk)
    COMPLEX(dp) :: vkb(npwx, nkb, nk), dmat(nkb, nkb)
    INTEGER :: npw(nk)
    INTEGER :: ig, k, i, j
    REAL(dp) :: s
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    k = blockIdx%y
    IF (ig > npwx) RETURN
    IF (ig > npw(k)) THEN
      hd(ig, k) = 0.0_dp
      RETURN
    ENDIF
    s = g2(ig, k) + v0
    DO i = 1, nkb
      DO j = 1, nkb
        IF (dmat(i, j) /= (0.0_dp, 0.0_dp)) s = s + REAL(dmat(i, j)) * REAL(vkb(ig, i, k) * CONJG(vkb(ig, j, k)))
      ENDDO
    ENDDO
    hd(ig, k) = s
  END SUBROUTINE k_hdiag
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_prec(x, hd, ewc, ncol, npw, npwx, nbnd, nk)
    ! g_psi: x(:, n, k) /= (h_diag - ew), |denominator| >= 1e-4, n <= ncol(k); rows > npw zeroed
    INTEGER, VALUE :: npwx, nbnd, nk
    COMPLEX(dp) :: x(npwx, nbnd, nk)
    REAL(dp) :: hd(npwx, nk), ewc(nbnd, nk)
    INTEGER :: ncol(nk), npw(nk)
    INTEGER :: ig, n, k
    REAL(dp) :: denm
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    n = blockIdx%y
    k = blockIdx%z
    IF (ig > npwx .OR. n > ncol(k)) RETURN
    IF (ig > npw(k)) THEN
      x(ig, n, k) = (0.0_dp, 0.0_dp)
      RETURN
    ENDIF
    denm = hd(ig, k) - ewc(n, k)
    IF (ABS(denm) < 1.0d-4) denm = SIGN(1.0d-4, denm)
    x(ig, n, k) = x(ig, n, k) / denm
  END SUBROUTINE k_prec
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_norm(x, nrm, ncol, npwx, nbnd, nk)
    ! nrm(n, k) = <x|x> (one block per column)
    INTEGER, VALUE :: npwx, nbnd, nk
    COMPLEX(dp) :: x(npwx, nbnd, nk)
    REAL(dp) :: nrm(nbnd, nk)
    INTEGER :: ncol(nk)
    REAL(dp), SHARED :: sh(RB)
    INTEGER :: n, k, ig, t, st
    REAL(dp) :: s
    n = blockIdx%x
    k = blockIdx%y
    IF (n > ncol(k)) RETURN
    t = threadIdx%x
    s = 0.0_dp
    DO ig = t, npwx, RB
      s = s + REAL(x(ig, n, k))**2 + AIMAG(x(ig, n, k))**2
    ENDDO
    sh(t) = s
    CALL syncthreads()
    st = RB / 2
    DO WHILE (st > 0)
      IF (t <= st) sh(t) = sh(t) + sh(t + st)
      CALL syncthreads()
      st = st / 2
    ENDDO
    IF (t == 1) nrm(n, k) = sh(1)
  END SUBROUTINE k_norm
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_scal(x, nrm, ncol, npwx, nbnd, nk)
    INTEGER, VALUE :: npwx, nbnd, nk
    COMPLEX(dp) :: x(npwx, nbnd, nk)
    REAL(dp) :: nrm(nbnd, nk)
    INTEGER :: ncol(nk)
    INTEGER :: ig, n, k
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    n = blockIdx%y
    k = blockIdx%z
    IF (ig > npwx .OR. n > ncol(k)) RETURN
    x(ig, n, k) = x(ig, n, k) / SQRT(nrm(n, k))
  END SUBROUTINE k_scal
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_append(psi, hpsi, xn, hxn, off, ncol, npwx, nvx, nbnd, nk)
    ! psi(:, off(k)+n, k) = xn(:, n, k), hpsi likewise, n <= ncol(k)
    INTEGER, VALUE :: npwx, nvx, nbnd, nk
    COMPLEX(dp) :: psi(npwx, nvx, nk), hpsi(npwx, nvx, nk), xn(npwx, nbnd, nk), hxn(npwx, nbnd, nk)
    INTEGER :: off(nk), ncol(nk)
    INTEGER :: ig, n, k
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    n = blockIdx%y
    k = blockIdx%z
    IF (ig > npwx .OR. n > ncol(k)) RETURN
    psi(ig, off(k) + n, k) = xn(ig, n, k)
    hpsi(ig, off(k) + n, k) = hxn(ig, n, k)
  END SUBROUTINE k_append
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_restart(psi, hpsi, t1, t2, flag, npwx, nvx, nbnd, nk)
    ! for k with flag(k) = 1: psi(:, 1:nbnd) = t1, hpsi(:, 1:nbnd) = t2
    INTEGER, VALUE :: npwx, nvx, nbnd, nk
    COMPLEX(dp) :: psi(npwx, nvx, nk), hpsi(npwx, nvx, nk), t1(npwx, nbnd, nk), t2(npwx, nbnd, nk)
    INTEGER :: flag(nk)
    INTEGER :: ig, n, k
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    n = blockIdx%y
    k = blockIdx%z
    IF (ig > npwx .OR. flag(k) /= 1) RETURN
    psi(ig, n, k) = t1(ig, n, k)
    hpsi(ig, n, k) = t2(ig, n, k)
  END SUBROUTINE k_restart
  !
  ! ---------------------------------------------------------------------------------------------
  ! helpers
  ! ---------------------------------------------------------------------------------------------
  SUBROUTINE chk(istat, what)
    INTEGER, INTENT(IN) :: istat
    CHARACTER(LEN=*), INTENT(IN) :: what
    IF (istat /= 0) THEN
      WRITE(*, '(A,A,A,I0)') 'nscf_engine: ', what, ' failed, status ', istat
      STOP 1
    ENDIF
  END SUBROUTINE chk
  !
  SUBROUTINE kerr(what)
    CHARACTER(LEN=*), INTENT(IN) :: what
    INTEGER :: istat
    istat = cudaGetLastError()
    IF (istat /= cudaSuccess) THEN
      WRITE(*, '(A,A,A,A)') 'nscf_engine: kernel ', what, ': ', TRIM(cudaGetErrorString(istat))
      STOP 1
    ENDIF
  END SUBROUTINE kerr
  !
  INTEGER FUNCTION nblk(n, b)
    INTEGER, INTENT(IN) :: n, b
    nblk = MAX(1, (n + b - 1) / b)
  END FUNCTION nblk
  !
  INTEGER FUNCTION nscf_eng_kmax(npwx_, nbnd_, nkb_, david, nnr_)
    !! largest k chunk that fits in the free device memory (minus 2 GiB)
    INTEGER, INTENT(IN) :: npwx_, nbnd_, nkb_, david, nnr_
    INTEGER(8) :: per, freeb, totb
    INTEGER :: istat, nv
    nv = david * nbnd_
    per = 16_8 * npwx_ * (2 * nv + 4 * nbnd_ + nkb_) + 16_8 * nv * (2 * nv + 2 * nbnd_) + 4_8 * nnr_ + 32_8 * npwx_
    istat = cudaMemGetInfo(freeb, totb)
    nscf_eng_kmax = INT(MAX(1_8, (freeb - 16_8 * nnr_ * 256 - 2_8 * 1024**3) / per))
  END FUNCTION nscf_eng_kmax
  !
  SUBROUTINE make_plans()
    INTEGER :: ip, istat, n(3), emb(3)
    IF (nplan > 0) THEN
      DO ip = 0, nplan - 1
        istat = cufftDestroy(plans(ip))
      ENDDO
      nplan = 0
    ENDIF
    n = (/ nr(3), nr(2), nr(1) /)
    emb = (/ nrx(3), nrx(2), nrx(1) /)
    DO ip = 0, MAXPLAN - 1
      IF (2**ip > bmax) EXIT
      istat = cufftPlanMany(plans(ip), 3, n, emb, 1, nnr, emb, 1, nnr, CUFFT_Z2Z, 2**ip)
      CALL chk(istat, 'cufftPlanMany')
      istat = cufftSetStream(plans(ip), strm)
      CALL chk(istat, 'cufftSetStream')
      nplan = ip + 1
    ENDDO
  END SUBROUTINE make_plans
  !
  SUBROUTINE gemm_sb(ta, tb, m, n, kk, alpha, a, lda, sa, b, ldb, sbb, beta, c, ldc, sc, nbat)
    INTEGER, INTENT(IN) :: ta, tb, m, n, kk, lda, ldb, ldc, nbat
    INTEGER(8), INTENT(IN) :: sa, sbb, sc
    COMPLEX(dp), INTENT(IN) :: alpha, beta
    COMPLEX(dp), DEVICE :: a(*), b(*), c(*)
    INTEGER :: istat
    istat = cublasZgemmStridedBatched(hb, ta, tb, m, n, kk, alpha, a, lda, sa, b, ldb, sbb, beta, c, ldc, sc, nbat)
    CALL chk(istat, 'cublasZgemmStridedBatched')
  END SUBROUTINE gemm_sb
  !
  SUBROUTINE apply_h(x, y, list, nl)
    ! y(:, c) = H x(:, c) for the nl columns c in list (columns c = n + nbnd (k - 1) of (npwx, nbnd, nk))
    COMPLEX(dp), DEVICE :: x(npwx, nbnd, nk), y(npwx, nbnd, nk)
    INTEGER, DEVICE :: list(*)
    INTEGER, INTENT(IN) :: nl
    INTEGER :: s0, nb, ip, istat
    REAL(dp) :: scale
    scale = 1.0_dp / REAL(nr(1) * nr(2) * nr(3), dp)
    s0 = 0
    DO WHILE (s0 < nl)
      nb = MIN(bmax, nl - s0)
      ip = nplan - 1
      DO WHILE (ip > 0 .AND. 2**ip > nb)
        ip = ip - 1
      ENDDO
      nb = 2**ip
      CALL k_scat<<<dim3(nblk(nnr, TPB), nb, 1), TPB, 0, strm>>>(d_box, x, list, s0, nnr, npwx, nbnd, nk, d_inv)
      CALL kerr('scat')
      istat = cufftExecZ2Z(plans(ip), d_box, d_box, CUFFT_INVERSE)
      CALL chk(istat, 'cufftExecZ2Z')
      CALL k_vmul<<<dim3(nblk(nnr, TPB), nb, 1), TPB, 0, strm>>>(d_box, d_vrs, nnr)
      CALL kerr('vmul')
      istat = cufftExecZ2Z(plans(ip), d_box, d_box, CUFFT_FORWARD)
      CALL chk(istat, 'cufftExecZ2Z')
      CALL k_gath<<<dim3(nblk(npwx, TPB), nb, 1), TPB, 0, strm>>>(y, x, d_box, list, s0, nnr, npwx, nbnd, nk, &
           d_npw, d_idx, d_g2, scale)
      CALL kerr('gath')
      s0 = s0 + nb
    ENDDO
    ! nonlocal: y += vkb D vkb^H x on all columns (inactive columns of x are zero or ignored)
    CALL gemm_sb(CUBLAS_OP_C, CUBLAS_OP_N, nkb, nbnd, npwx, (1.0_dp, 0.0_dp), d_vkb, npwx, INT(npwx, 8) * nkb, &
         x, npwx, INT(npwx, 8) * nbnd, (0.0_dp, 0.0_dp), d_cf, nkb, INT(nkb, 8) * nbnd, nk)
    istat = cublasZgemm(hb, CUBLAS_OP_N, CUBLAS_OP_N, nkb, nbnd * nk, nkb, (1.0_dp, 0.0_dp), d_dmat, nkb, &
            d_cf, nkb, (0.0_dp, 0.0_dp), d_cf2, nkb)
    CALL chk(istat, 'cublasZgemm (D)')
    CALL gemm_sb(CUBLAS_OP_N, CUBLAS_OP_N, npwx, nbnd, nkb, (1.0_dp, 0.0_dp), d_vkb, npwx, INT(npwx, 8) * nkb, &
         d_cf2, nkb, INT(nkb, 8) * nbnd, (1.0_dp, 0.0_dp), y, npwx, INT(npwx, 8) * nbnd, nk)
  END SUBROUTINE apply_h
  !
  ! ---------------------------------------------------------------------------------------------
  ! public API
  ! ---------------------------------------------------------------------------------------------
  SUBROUTINE nscf_eng_init(nk_, npwx_, nbnd_, nkb_, david, nr1, nr2, nr3, nr1x, nr2x, nr3x, nnr_, vrs, dmat, v_of_0)
    !! one chunk of nk_ k-points; vrs(nnr_): smooth local potential; dmat(nkb,nkb): dense D (deeq)
    INTEGER, INTENT(IN) :: nk_, npwx_, nbnd_, nkb_, david, nr1, nr2, nr3, nr1x, nr2x, nr3x, nnr_
    REAL(dp), INTENT(IN) :: vrs(nnr_), v_of_0
    COMPLEX(dp), INTENT(IN) :: dmat(nkb_, nkb_)
    INTEGER :: istat
    CALL nscf_eng_free()
    nk = nk_; npwx = npwx_; nbnd = nbnd_; nkb = MAX(nkb_, 1); nvx = david * nbnd_
    nr = (/ nr1, nr2, nr3 /); nrx = (/ nr1x, nr2x, nr3x /); nnr = nnr_; v0 = v_of_0
    IF (.NOT. have_h) THEN
      istat = cudaStreamCreate(strm)
      CALL chk(istat, 'cudaStreamCreate')
      istat = cublasCreate(hb)
      CALL chk(istat, 'cublasCreate')
      istat = cublasSetStream(hb, strm)
      CALL chk(istat, 'cublasSetStream')
      have_h = .TRUE.
    ENDIF
    ALLOCATE(h_npw(nk), h_idx(npwx, nk), h_g2(npwx, nk), h_vkb(npwx, nkb, nk), h_psi0(npwx, nbnd, nk))
    ALLOCATE(h_e(nbnd, nk), h_evc(npwx, nbnd, nk), h_ok(nk), h_iter(nk))
    h_npw = 0; h_idx = 1; h_g2 = 0.0_dp; h_vkb = (0.0_dp, 0.0_dp); h_psi0 = (0.0_dp, 0.0_dp)
    ALLOCATE(d_vrs(nnr), d_dmat(nkb, nkb))
    d_vrs = vrs
    IF (nkb_ > 0) THEN
      d_dmat = dmat
    ELSE
      d_dmat = (0.0_dp, 0.0_dp)
    ENDIF
  END SUBROUTINE nscf_eng_init
  !
  SUBROUTINE nscf_eng_set_k(ik, npw, idx, g2kin, vkb, psi0)
    !! k-point ik of the chunk: idx(npw) 1-based FFT indices of k+G; g2kin, vkb (npwx, nkb), start psi0
    INTEGER, INTENT(IN) :: ik, npw, idx(npw)
    REAL(dp), INTENT(IN) :: g2kin(npwx)
    COMPLEX(dp), INTENT(IN) :: vkb(npwx, *), psi0(npwx, nbnd)
    h_npw(ik) = npw
    h_idx(1:npw, ik) = idx(1:npw)
    h_g2(1:npw, ik) = g2kin(1:npw)
    IF (nkb > 0) h_vkb(1:npw, 1:nkb, ik) = vkb(1:npw, 1:nkb)
    h_psi0(1:npw, :, ik) = psi0(1:npw, :)
  END SUBROUTINE nscf_eng_set_k
  !
  SUBROUTINE nscf_eng_get_k(ik, evc, e, ok, niter)
    INTEGER, INTENT(IN) :: ik
    COMPLEX(dp), INTENT(OUT) :: evc(npwx, nbnd)
    REAL(dp), INTENT(OUT) :: e(nbnd)
    LOGICAL, INTENT(OUT) :: ok
    INTEGER, INTENT(OUT) :: niter
    evc = h_evc(:, :, ik)
    e = h_e(:, ik)
    ok = h_ok(ik)
    niter = h_iter(ik)
  END SUBROUTINE nscf_eng_get_k
  !
  SUBROUTINE nscf_eng_solve(ethr, maxit, nhpsi_tot)
    !! Davidson for all k of the chunk. Results via nscf_eng_get_k.
    REAL(dp), INTENT(IN) :: ethr
    INTEGER, INTENT(IN) :: maxit
    INTEGER(8), INTENT(OUT) :: nhpsi_tot
    COMPLEX(dp), ALLOCATABLE :: hc(:,:,:), sc(:,:,:), vc(:,:,:), A(:,:,:), B(:,:,:), ha(:,:), sa(:,:), work(:)
    REAL(dp), ALLOCATABLE :: ew(:,:), e(:,:), ewc(:,:), rwork(:), w(:)
    INTEGER, ALLOCATABLE :: nbase(:), notcnv(:), list(:), off(:), ncol(:), flag(:), inv(:,:), info(:)
    LOGICAL, ALLOCATABLE :: active(:), conv(:), bconv(:,:)
    INTEGER :: istat, ik, n, m, nl, it, nb, lwork, ierr, np, nact
    INTEGER(8) :: snv, sbn, c0, c1, c2, cr
    COMPLEX(dp), PARAMETER :: ONE = (1.0_dp, 0.0_dp), ZERO = (0.0_dp, 0.0_dp)
    !
    CALL SYSTEM_CLOCK(c0, cr)
    nscf_eng_t_eig = 0
    nhpsi_tot = 0
    snv = INT(npwx, 8) * nvx
    sbn = INT(npwx, 8) * nbnd
    ! device state
    ALLOCATE(d_npw(nk), d_idx(npwx, nk), d_g2(npwx, nk), d_vkb(npwx, nkb, nk), d_hd(npwx, nk))
    ALLOCATE(d_inv(nnr, nk), d_list(nbnd * nk), d_off(nk), d_ncol(nk), d_flag(nk))
    ALLOCATE(d_psi(npwx, nvx, nk), d_hpsi(npwx, nvx, nk), d_new(npwx, nbnd, nk), d_hnew(npwx, nbnd, nk))
    ALLOCATE(d_hc(nvx, nvx, nk), d_sc(nvx, nvx, nk), d_A(nvx, nbnd, nk), d_B(nvx, nbnd, nk))
    ALLOCATE(d_cf(nkb, nbnd, nk), d_cf2(nkb, nbnd, nk), d_t1(npwx, nbnd, nk), d_t2(npwx, nbnd, nk))
    ALLOCATE(d_ewc(nbnd, nk), d_nrm(nbnd, nk), d_box(nnr, bmax))
    d_npw = h_npw; d_idx = h_idx; d_g2 = h_g2; d_vkb = h_vkb
    ALLOCATE(inv(nnr, nk))
    inv = 0
    DO ik = 1, nk
      DO n = 1, h_npw(ik)
        inv(h_idx(n, ik), ik) = n
      ENDDO
    ENDDO
    d_inv = inv
    DEALLOCATE(inv)
    d_psi = (0.0_dp, 0.0_dp)
    d_hpsi = (0.0_dp, 0.0_dp)
    d_new = h_psi0
    d_hnew = (0.0_dp, 0.0_dp)
    istat = cudaDeviceSynchronize()
    CALL make_plans()
    CALL k_hdiag<<<dim3(nblk(npwx, TPB), nk, 1), TPB, 0, strm>>>(d_hd, d_vkb, d_dmat, d_g2, v0, d_npw, npwx, nkb, nk)
    CALL kerr('hdiag')
    !
    ALLOCATE(hc(nvx, nvx, nk), sc(nvx, nvx, nk), vc(nvx, nbnd, nk), A(nvx, nbnd, nk), B(nvx, nbnd, nk))
    ALLOCATE(ew(nbnd, nk), e(nbnd, nk), ewc(nbnd, nk), nbase(nk), notcnv(nk), list(nbnd * nk), off(nk), &
             ncol(nk), flag(nk), active(nk), conv(nbnd), info(nk), bconv(nbnd, nk))
    !
    ! start: psi(:, 1:nbnd) = psi0, hpsi = H psi0
    nl = 0
    DO ik = 1, nk
      DO n = 1, nbnd
        nl = nl + 1
        list(nl) = n + nbnd * (ik - 1)
      ENDDO
    ENDDO
    d_list(1:nl) = list(1:nl)
    CALL apply_h(d_new, d_hnew, d_list, nl)
    nhpsi_tot = nhpsi_tot + nl
    off = 0
    ncol = nbnd
    d_off = off
    d_ncol = ncol
    CALL k_append<<<dim3(nblk(npwx, TPB), nbnd, nk), TPB, 0, strm>>>(d_psi, d_hpsi, d_new, d_hnew, d_off, d_ncol, &
         npwx, nvx, nbnd, nk)
    CALL kerr('append')
    nbase = nbnd
    notcnv = nbnd
    active = .TRUE.
    h_ok = .FALSE.
    h_iter = 0
    e = 0.0_dp
    ew = 0.0_dp
    ewc = 0.0_dp
    bconv = .FALSE.
    !
    DO it = 0, maxit
      ! reduced matrices over the full nvx block (entries beyond nbase are ignored)
      CALL gemm_sb(CUBLAS_OP_C, CUBLAS_OP_N, nvx, nvx, npwx, ONE, d_psi, npwx, snv, d_hpsi, npwx, snv, ZERO, &
           d_hc, nvx, INT(nvx, 8) * nvx, nk)
      CALL gemm_sb(CUBLAS_OP_C, CUBLAS_OP_N, nvx, nvx, npwx, ONE, d_psi, npwx, snv, d_psi, npwx, snv, ZERO, &
           d_sc, nvx, INT(nvx, 8) * nvx, nk)
      istat = cudaStreamSynchronize(strm)
      hc = d_hc
      sc = d_sc
      CALL SYSTEM_CLOCK(c1)
      ! Ritz pairs (host, LAPACK, OpenMP over k)
      info = 0
      vc = (0.0_dp, 0.0_dp)
      !$omp parallel do private(ik, nb, ha, sa, w, work, rwork, lwork, ierr, n, m) schedule(dynamic)
      DO ik = 1, nk
        IF (.NOT. active(ik)) CYCLE
        nb = nbase(ik)
        ALLOCATE(ha(nb, nb), sa(nb, nb), w(nb), rwork(3 * nb), work(1))
        DO m = 1, nb
          DO n = m, nb
            ha(n, m) = hc(n, m, ik)
            sa(n, m) = sc(n, m, ik)
          ENDDO
          ha(m, m) = CMPLX(REAL(ha(m, m)), 0.0_dp, KIND=dp)
          sa(m, m) = CMPLX(REAL(sa(m, m)), 0.0_dp, KIND=dp)
        ENDDO
        CALL zhegv(1, 'V', 'L', nb, ha, nb, sa, nb, w, work, -1, rwork, ierr)
        lwork = MAX(1, INT(REAL(work(1))))
        DEALLOCATE(work)
        ALLOCATE(work(lwork))
        CALL zhegv(1, 'V', 'L', nb, ha, nb, sa, nb, w, work, lwork, rwork, ierr)
        info(ik) = ierr
        IF (ierr == 0) THEN
          ew(:, ik) = w(1:nbnd)
          vc(1:nb, 1:nbnd, ik) = ha(1:nb, 1:nbnd)
        ENDIF
        DEALLOCATE(ha, sa, w, rwork, work)
      ENDDO
      !$omp end parallel do
      CALL SYSTEM_CLOCK(c2)
      nscf_eng_t_eig = nscf_eng_t_eig + REAL(c2 - c1, dp) / REAL(cr, dp)
      nscf_eng_nit = it
      !
      ! convergence, restart and expansion bookkeeping
      flag = 0
      DO ik = 1, nk
        ncol(ik) = 0
        off(ik) = 0
        A(:, :, ik) = (0.0_dp, 0.0_dp)
        B(:, :, ik) = (0.0_dp, 0.0_dp)
        IF (.NOT. active(ik)) CYCLE
        IF (info(ik) /= 0) THEN
          active(ik) = .FALSE.          ! h_ok stays false: vanilla Davidson for this k
          CYCLE
        ENDIF
        IF (it == 0) THEN
          conv = .FALSE.
        ELSE
          conv = ABS(ew(:, ik) - e(:, ik)) < ethr
        ENDIF
        notcnv(ik) = COUNT(.NOT. conv)
        bconv(:, ik) = conv
        e(:, ik) = ew(:, ik)
        h_iter(ik) = it
        IF (notcnv(ik) == 0 .OR. nbase(ik) + notcnv(ik) > nvx .OR. it == maxit) THEN
          ! evc / restart vectors = psi vc, hpsi vc  (A <- vc, flag 1)
          A(:, :, ik) = vc(:, :, ik)
          flag(ik) = 1
          IF (notcnv(ik) == 0) THEN
            flag(ik) = 2                 ! converged: final vectors only
          ELSE IF (it == maxit) THEN
            flag(ik) = 3                 ! not converged within maxit
          ENDIF
        ENDIF
      ENDDO
      !
      IF (ANY(flag /= 0)) THEN
        d_A = A
        d_flag = MERGE(1, 0, flag == 1)
        CALL gemm_sb(CUBLAS_OP_N, CUBLAS_OP_N, npwx, nbnd, nvx, ONE, d_psi, npwx, snv, d_A, nvx, &
             INT(nvx, 8) * nbnd, ZERO, d_t1, npwx, sbn, nk)
        CALL gemm_sb(CUBLAS_OP_N, CUBLAS_OP_N, npwx, nbnd, nvx, ONE, d_hpsi, npwx, snv, d_A, nvx, &
             INT(nvx, 8) * nbnd, ZERO, d_t2, npwx, sbn, nk)
        IF (ANY(flag == 2 .OR. flag == 3)) THEN
          istat = cudaStreamSynchronize(strm)
          ! final vectors of the k-points that stop now
          BLOCK
            COMPLEX(dp), ALLOCATABLE :: t(:,:,:)
            ALLOCATE(t(npwx, nbnd, nk))
            t = d_t1
            DO ik = 1, nk
              IF (flag(ik) == 2 .OR. flag(ik) == 3) THEN
                h_evc(:, :, ik) = t(:, :, ik)
                h_e(:, ik) = e(:, ik)
                h_ok(ik) = (flag(ik) == 2)
                active(ik) = .FALSE.
              ENDIF
            ENDDO
            DEALLOCATE(t)
          END BLOCK
        ENDIF
        IF (ANY(flag == 1)) THEN
          CALL k_restart<<<dim3(nblk(npwx, TPB), nbnd, nk), TPB, 0, strm>>>(d_psi, d_hpsi, d_t1, d_t2, d_flag, &
               npwx, nvx, nbnd, nk)
          CALL kerr('restart')
          DO ik = 1, nk
            IF (flag(ik) == 1) THEN
              nbase(ik) = nbnd
              ! Ritz vectors are now the basis: vc = identity on the first nbnd
              vc(:, :, ik) = (0.0_dp, 0.0_dp)
              DO n = 1, nbnd
                vc(n, n, ik) = (1.0_dp, 0.0_dp)
              ENDDO
            ENDIF
          ENDDO
        ENDIF
      ENDIF
      nact = COUNT(active)
      IF (nact == 0) EXIT
      !
      ! correction vectors for the unconverged bands: t = hpsi vc_c - psi vc_c ew_c
      nl = 0
      DO ik = 1, nk
        IF (.NOT. active(ik)) CYCLE
        np = 0
        DO n = 1, nbnd
          IF (.NOT. bconv(n, ik)) THEN
            np = np + 1
            A(:, np, ik) = vc(:, n, ik)
            B(:, np, ik) = -vc(:, n, ik) * e(n, ik)
            ewc(np, ik) = e(n, ik)
            nl = nl + 1
            list(nl) = np + nbnd * (ik - 1)
          ENDIF
        ENDDO
        ncol(ik) = np
        off(ik) = nbase(ik)
      ENDDO
      d_A = A
      d_B = B
      d_ewc = ewc
      d_ncol = ncol
      d_off = off
      d_list(1:nl) = list(1:nl)
      CALL gemm_sb(CUBLAS_OP_N, CUBLAS_OP_N, npwx, nbnd, nvx, ONE, d_psi, npwx, snv, d_B, nvx, &
           INT(nvx, 8) * nbnd, ZERO, d_new, npwx, sbn, nk)
      CALL gemm_sb(CUBLAS_OP_N, CUBLAS_OP_N, npwx, nbnd, nvx, ONE, d_hpsi, npwx, snv, d_A, nvx, &
           INT(nvx, 8) * nbnd, ONE, d_new, npwx, sbn, nk)
      CALL k_prec<<<dim3(nblk(npwx, TPB), nbnd, nk), TPB, 0, strm>>>(d_new, d_hd, d_ewc, d_ncol, d_npw, npwx, nbnd, nk)
      CALL kerr('prec')
      CALL k_norm<<<dim3(nbnd, nk, 1), RB, 0, strm>>>(d_new, d_nrm, d_ncol, npwx, nbnd, nk)
      CALL kerr('norm')
      CALL k_scal<<<dim3(nblk(npwx, TPB), nbnd, nk), TPB, 0, strm>>>(d_new, d_nrm, d_ncol, npwx, nbnd, nk)
      CALL kerr('scal')
      CALL apply_h(d_new, d_hnew, d_list, nl)
      nhpsi_tot = nhpsi_tot + nl
      CALL k_append<<<dim3(nblk(npwx, TPB), nbnd, nk), TPB, 0, strm>>>(d_psi, d_hpsi, d_new, d_hnew, d_off, d_ncol, &
           npwx, nvx, nbnd, nk)
      CALL kerr('append')
      DO ik = 1, nk
        IF (active(ik)) nbase(ik) = nbase(ik) + ncol(ik)
      ENDDO
    ENDDO
    istat = cudaDeviceSynchronize()
    CALL SYSTEM_CLOCK(c2)
    nscf_eng_t_tot = REAL(c2 - c0, dp) / REAL(cr, dp)
    DEALLOCATE(hc, sc, vc, A, B, ew, e, ewc, nbase, notcnv, list, off, ncol, flag, active, conv, info, bconv)
    DEALLOCATE(d_npw, d_idx, d_g2, d_vkb, d_hd, d_inv, d_list, d_off, d_ncol, d_flag, d_psi, d_hpsi, d_new, &
               d_hnew, d_hc, d_sc, d_A, d_B, d_cf, d_cf2, d_t1, d_t2, d_ewc, d_nrm, d_box)
  END SUBROUTINE nscf_eng_solve
  !
  SUBROUTINE nscf_eng_free()
    IF (ALLOCATED(h_npw)) DEALLOCATE(h_npw, h_idx, h_g2, h_vkb, h_psi0, h_e, h_evc, h_ok, h_iter)
    IF (ALLOCATED(d_vrs)) DEALLOCATE(d_vrs, d_dmat)
    nk = 0
  END SUBROUTINE nscf_eng_free
  !
END MODULE nscf_engine
