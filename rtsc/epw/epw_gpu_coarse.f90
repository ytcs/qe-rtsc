!
! rtsc: GPU path for the local part of the EPW coarse electron-phonon matrix elements.
!
! Vanilla (dvqpsi_us3 + ZGEMM in ep_coarse_compute), per k and per perturbation nu:
!   dvpsi(G, n) = fw_FFT[ psi_k,n(r) * V_nu(r) ](G) restricted to the k+q sphere
!   g(m, n, nu) = sum_G conj(psi_k+q,m(G)) dvpsi(G, n)
! with V_nu = dvscf_nu + dvloc_nu. By Parseval (QE: invfft unnormalised, fwfft has 1/N, and
! psi_k+q is zero outside its sphere) this is exactly
!   g(m, n, nu) = (1/N) sum_r conj(psiq_m(r)) V_nu(r) psi_n(r),   psiq = inv_FFT[psi_k+q]
! so per k we need 2*nbnd inverse FFTs (all perturbations share them) and one ZGEMM
! (nbnd x nnr) x (nnr x nbnd*npe). Same algebra, different floating-point order.
!
! No QE modules: plain arrays in and out (like rtsc/sth/sth_engine.f90).
!
MODULE epw_gpu_coarse
  USE cudafor
  USE cufft
  USE cublas_v2
  USE iso_c_binding, ONLY : c_int, c_double
  IMPLICIT NONE
  PRIVATE
  INTEGER, PARAMETER :: dp = KIND(1.d0)
  INTEGER, PARAMETER :: wchunk_bytes = 268435456   ! cap on the V*psi work buffer (256 MB)
  !
  LOGICAL :: ready = .FALSE.
  INTEGER :: nr(3), nrx(3), nnr = 0, nbnd = 0, nvmax = 0, nvset = 0, nmc = 0
  INTEGER :: plan = 0
  TYPE(cublasHandle) :: hblas
  COMPLEX(dp), DEVICE, ALLOCATABLE :: d_v(:, :), d_psi(:, :), d_psq(:, :), d_w(:, :), d_g(:, :)
  COMPLEX(dp), DEVICE, ALLOCATABLE :: d_ck(:, :), d_cq(:, :)
  INTEGER, DEVICE, ALLOCATABLE :: d_ik(:), d_iq(:)
  INTEGER :: ldc = 0
  !
  ! Nonlocal part on the GPU (groups/compute/KERNEL-LOG.md, lever 3). Vanilla (and ep_gpu_k before this):
  ! per k, host ZGEMMs becq_a = [vkb (k+q+G)_a]^H psi_k+q (a = 0: vkb itself), host loops for
  ! ps1 = D (sum_a u_a alphap_a), ps2_a = -i tpiba u_a D becp1, and g += becq_0^H ps1 + sum_a becq_a^H ps2_a.
  ! Here: D becp1 and D alphap_a by cuBLAS, then one contraction kernel for all perturbations, added to
  ! the local part on the device (one D2H of g). Same algebra (block-diagonal D = deff), other rounding.
  INTEGER :: nkbs = 0, nat3s = 0, nvs = 0
  COMPLEX(dp), DEVICE, ALLOCATABLE :: d_vkb(:,:), d_auxv(:,:), d_becq(:,:,:), d_bp(:,:), d_al(:,:,:)
  COMPLEX(dp), DEVICE, ALLOCATABLE :: d_dm(:,:), d_db(:,:), d_da(:,:,:), d_u(:,:)
  REAL(dp), DEVICE, ALLOCATABLE :: d_kqg(:,:)
  INTEGER, DEVICE, ALLOCATABLE :: d_at(:)
  !
  PUBLIC :: egc_init, egc_set_v, egc_local, egc_finalize, egc_ready, egc_nl_set, egc_local_nl
  !
  ! rtsc amdahl profile (RTSC_EPW_PROF=1; groups/compute/AMDAHL-LOG.md): synchronising wall clocks per part of
  ! egc_local_nl, cumulative; tp(1) uploads, (2) FFTs of psi_k / psi_k+q, (3) V*psi + ZGEMM, (4) nonlocal, (5) D2H,
  ! (6) host prep in ep_gpu_k (set by the caller through egc_prof_add)
  LOGICAL, SAVE :: prof = .FALSE., prof_read = .FALSE.
  REAL(dp), SAVE :: tp(6) = 0.0_dp
  INTEGER, SAVE :: np_calls = 0
  PUBLIC :: egc_prof_on, egc_prof_add, egc_prof_print, egc_wall
  !
  ! rtsc amdahl (RTSC_EPW_FDX=1; groups/compute/AMDAHL-LOG.md): the local part in G-space on the pruned cuFFTDx path
  ! (rtsc/sth/fdx/sth_fftdx.cu), i.e. vanilla's own form dvpsi = fwfft(psi_k(r) V(r)) on the k+q sphere, then
  ! g = psi_k+q^H dvpsi (one ZGEMM, K = npwq). Replaces the Parseval form, which materialises V psi for every band and
  ! mode in a full box (nnr x nbnd x nmodes) before an M = nbnd ZGEMM over nnr. Only psi_k needs an inverse FFT.
  ! RTSC_EPW_FDX_CHECK=1: at the first call also runs the Parseval form and prints the difference.
  LOGICAL, SAVE :: fdx_req = .FALSE., fdx_chk = .FALSE., fdx_read = .FALSE., fdx_ok = .TRUE.
  ! RTSC_EPW_BEC=1: becp1 and alphap of the current k on the device (egc_becs_set uploads vkb_k and (k+G)_a; they
  ! are formed in egc_local_nl once psi_k is on the device)
  LOGICAL, SAVE :: bec_dev = .FALSE.
  INTEGER, SAVE :: bec_npw = 0
  REAL(dp), SAVE :: bec_tpiba = 0.0_dp
  COMPLEX(dp), DEVICE, ALLOCATABLE :: d_vkbk(:,:), d_ckt(:,:)
  REAL(dp), DEVICE, ALLOCATABLE :: d_kgk(:,:)
  PUBLIC :: egc_becs_set
  COMPLEX(dp), DEVICE, ALLOCATABLE :: d_dvp(:,:)
  INTEGER, DEVICE, ALLOCATABLE :: d_lst(:)
  INTERFACE
    INTEGER(c_int) FUNCTION fdx_geom1(nr, nrx, npwq, idxkq, ncols) BIND(C, name='fdx_geom1')
      IMPORT :: c_int
      INTEGER(c_int) :: nr(3), nrx(3), idxkq(*)
      INTEGER(c_int), VALUE :: npwq, ncols
    END FUNCTION fdx_geom1
    INTEGER(c_int) FUNCTION fdx_dpot_d(d, psir, dv, list, s0, nb, nslot, npert, npwx, nnr, pj, scale, strm) &
                                       BIND(C, name='fdx_dpot_d')
      IMPORT :: c_int, c_double, cuda_stream_kind
      COMPLEX(c_double), DEVICE :: d(*), psir(*), dv(*)
      INTEGER(c_int), DEVICE :: list(*)
      INTEGER(c_int), VALUE :: s0, nb, nslot, npert, npwx, nnr, pj
      REAL(c_double), VALUE :: scale
      INTEGER(cuda_stream_kind), VALUE :: strm
    END FUNCTION fdx_dpot_d
  END INTERFACE
  !
CONTAINS
  !
  LOGICAL FUNCTION egc_ready()
    egc_ready = ready
  END FUNCTION egc_ready
  !
  SUBROUTINE chk(istat, what)
    INTEGER, INTENT(IN) :: istat
    CHARACTER(*), INTENT(IN) :: what
    IF (istat /= 0) THEN
      WRITE(*, '(a, a, a, i0)') 'epw_gpu_coarse: ', what, ' failed, status ', istat
      STOP 1
    ENDIF
  END SUBROUTINE chk
  !
  LOGICAL FUNCTION egc_prof_on()
    CHARACTER(LEN=8) :: ev
    IF (.NOT. prof_read) THEN
      CALL GET_ENVIRONMENT_VARIABLE('RTSC_EPW_PROF', ev)
      prof = TRIM(ev) == '1'
      prof_read = .TRUE.
    ENDIF
    egc_prof_on = prof
  END FUNCTION egc_prof_on
  !
  REAL(dp) FUNCTION egc_wall(sync)
    LOGICAL, INTENT(IN) :: sync
    INTEGER(8) :: c, r
    INTEGER :: istat
    IF (sync) istat = cudaDeviceSynchronize()
    CALL SYSTEM_CLOCK(c, r)
    egc_wall = REAL(c, dp) / REAL(r, dp)
  END FUNCTION egc_wall
  !
  SUBROUTINE egc_prof_add(i, t)
    INTEGER, INTENT(IN) :: i
    REAL(dp), INTENT(IN) :: t
    tp(i) = tp(i) + t
  END SUBROUTINE egc_prof_add
  !
  SUBROUTINE egc_prof_print()
    IF (.NOT. prof) RETURN
    WRITE(*, '(A,I7,A,6F9.3)') '     rtsc epw prof: calls', np_calls, ' s: upload fft mulv+gemm nl d2h hostprep', tp
  END SUBROUTINE egc_prof_print
  !
  SUBROUTINE egc_init(nr1, nr2, nr3, nr1x, nr2x, nr3x, nnr_, nbnd_, nvmax_, npwx)
    !! Allocate device buffers and the batched FFT plan. Idempotent for unchanged sizes.
    INTEGER, INTENT(IN) :: nr1, nr2, nr3, nr1x, nr2x, nr3x, nnr_, nbnd_, nvmax_, npwx
    INTEGER :: istat, n(3), emb(3)
    IF (ready .AND. nnr == nnr_ .AND. nbnd == nbnd_ .AND. nvmax >= nvmax_ .AND. ldc >= npwx) RETURN
    CALL egc_finalize()
    IF (nr1x * nr2x * nr3x /= nnr_) CALL chk(1, 'nnr /= nr1x*nr2x*nr3x (G-vector parallelisation?)')
    nr = (/ nr1, nr2, nr3 /); nrx = (/ nr1x, nr2x, nr3x /)
    nnr = nnr_; nbnd = nbnd_; nvmax = nvmax_; ldc = npwx
    ! modes per chunk of the V*psi buffer
    nmc = MAX(1, MIN(nvmax, INT(wchunk_bytes / (16_8 * nnr * nbnd))))
    ALLOCATE(d_v(nnr, nvmax), d_psi(nnr, nbnd), d_psq(nnr, nbnd), d_w(nnr, nbnd * nmc))
    ALLOCATE(d_g(nbnd, nbnd * nvmax), d_ck(ldc, nbnd), d_cq(ldc, nbnd), d_ik(ldc), d_iq(ldc))
    n = (/ nr(3), nr(2), nr(1) /)
    emb = (/ nrx(3), nrx(2), nrx(1) /)
    istat = cufftPlanMany(plan, 3, n, emb, 1, nnr, emb, 1, nnr, CUFFT_Z2Z, nbnd)
    CALL chk(istat, 'cufftPlanMany')
    istat = cublasCreate(hblas)
    CALL chk(istat, 'cublasCreate')
    ready = .TRUE.
  END SUBROUTINE egc_init
  !
  SUBROUTINE egc_finalize()
    INTEGER :: istat
    IF (ALLOCATED(d_vkb)) DEALLOCATE(d_vkb, d_auxv, d_becq, d_bp, d_al, d_dm, d_db, d_da, d_u, d_kqg, d_at)
    IF (ALLOCATED(d_dvp)) DEALLOCATE(d_dvp, d_lst)
    IF (ALLOCATED(d_vkbk)) DEALLOCATE(d_vkbk, d_ckt, d_kgk)
    bec_dev = .FALSE.
    nkbs = 0
    IF (.NOT. ready) RETURN
    istat = cufftDestroy(plan)
    istat = cublasDestroy(hblas)
    DEALLOCATE(d_v, d_psi, d_psq, d_w, d_g, d_ck, d_cq, d_ik, d_iq)
    ready = .FALSE.
  END SUBROUTINE egc_finalize
  !
  SUBROUTINE egc_set_v(nv, v)
    !! Upload the total local perturbing potentials V(r, nu), nu = 1..nv.
    INTEGER, INTENT(IN) :: nv
    COMPLEX(dp), INTENT(IN) :: v(nnr, nv)
    INTEGER :: istat
    IF (nv > nvmax) CALL chk(1, 'egc_set_v: nv > nvmax')
    istat = cudaMemcpy(d_v, v, nnr * nv)
    CALL chk(istat, 'cudaMemcpy V')
    nvset = nv
  END SUBROUTINE egc_set_v
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_scatter(npw, nb, ld, ind, c, nnr_, a)
    INTEGER, VALUE :: npw, nb, ld, nnr_
    INTEGER, DEVICE :: ind(ld)
    COMPLEX(dp), DEVICE :: c(ld, nb), a(nnr_, nb)
    INTEGER :: ig, ib
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    ib = blockIdx%y
    IF (ig <= npw) a(ind(ig), ib) = c(ig, ib)
  END SUBROUTINE k_scatter
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_mulv(nnr_, nb, nm, j0, nvm, v, psi, w)
    !! w(r, n + nb*(j-1)) = v(r, j0-1+j) * psi(r, n)
    INTEGER, VALUE :: nnr_, nb, nm, j0, nvm
    COMPLEX(dp), DEVICE :: v(nnr_, nvm), psi(nnr_, nb), w(nnr_, nb * nm)
    INTEGER :: ir, icol, n, j
    ir = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    icol = blockIdx%y
    IF (ir > nnr_) RETURN
    n = MOD(icol - 1, nb) + 1
    j = (icol - 1) / nb + 1
    w(ir, icol) = v(ir, j0 - 1 + j) * psi(ir, n)
  END SUBROUTINE k_mulv
  !
  SUBROUTINE egc_local(npw, indk, ck, npwq, indq, cq, g)
    !! g(m, n, nu) = (1/N) sum_r conj(psiq_m(r)) V_nu(r) psi_n(r) for nu = 1..nvset.
    !! indk/indq: FFT-grid positions (dffts%nl(igk)); ck/cq: (ldc, nbnd) G-space coefficients.
    INTEGER, INTENT(IN) :: npw, npwq
    INTEGER, INTENT(IN) :: indk(npw), indq(npwq)
    COMPLEX(dp), INTENT(IN) :: ck(ldc, nbnd), cq(ldc, nbnd)
    COMPLEX(dp), INTENT(OUT) :: g(nbnd, nbnd, nvset)
    INTEGER :: istat
    CALL egc_local_dev(npw, indk, ck, npwq, indq, cq)
    istat = cudaMemcpy(g, d_g, nbnd * nbnd * nvset); CALL chk(istat, 'D2H g')
  END SUBROUTINE egc_local
  !
  SUBROUTINE egc_local_dev(npw, indk, ck, npwq, indq, cq)
    !! the local part into d_g (device); d_cq keeps psi_k+q(G) for the nonlocal part
    INTEGER, INTENT(IN) :: npw, npwq
    INTEGER, INTENT(IN) :: indk(npw), indq(npwq)
    COMPLEX(dp), INTENT(IN) :: ck(ldc, nbnd), cq(ldc, nbnd)
    INTEGER :: istat, j0, nm
    TYPE(dim3) :: grd, blk
    REAL(dp) :: fac, ta, tb
    !
    IF (prof) ta = egc_wall(.TRUE.)
    istat = cudaMemcpy(d_ik, indk, npw);       CALL chk(istat, 'H2D indk')
    istat = cudaMemcpy(d_iq, indq, npwq);      CALL chk(istat, 'H2D indq')
    istat = cudaMemcpy(d_ck, ck, ldc * nbnd);  CALL chk(istat, 'H2D ck')
    istat = cudaMemcpy(d_cq, cq, ldc * nbnd);  CALL chk(istat, 'H2D cq')
    d_psi = (0.0_dp, 0.0_dp)
    d_psq = (0.0_dp, 0.0_dp)
    blk = dim3(256, 1, 1)
    grd = dim3((npw + 255) / 256, nbnd, 1)
    IF (prof) THEN
      tb = egc_wall(.TRUE.); tp(1) = tp(1) + tb - ta; ta = tb
    ENDIF
    CALL k_scatter<<<grd, blk>>>(npw, nbnd, ldc, d_ik, d_ck, nnr, d_psi)
    grd = dim3((npwq + 255) / 256, nbnd, 1)
    CALL k_scatter<<<grd, blk>>>(npwq, nbnd, ldc, d_iq, d_cq, nnr, d_psq)
    istat = cufftExecZ2Z(plan, d_psi, d_psi, CUFFT_INVERSE); CALL chk(istat, 'cufft psi')
    istat = cufftExecZ2Z(plan, d_psq, d_psq, CUFFT_INVERSE); CALL chk(istat, 'cufft psq')
    fac = 1.0_dp / DBLE(nr(1) * nr(2) * nr(3))
    IF (prof) THEN
      tb = egc_wall(.TRUE.); tp(2) = tp(2) + tb - ta; ta = tb
    ENDIF
    DO j0 = 1, nvset, nmc
      nm = MIN(nmc, nvset - j0 + 1)
      grd = dim3((nnr + 255) / 256, nbnd * nm, 1)
      CALL k_mulv<<<grd, blk>>>(nnr, nbnd, nm, j0, nvmax, d_v, d_psi, d_w)
      istat = cublasZgemm(hblas, CUBLAS_OP_C, CUBLAS_OP_N, nbnd, nbnd * nm, nnr, &
                          CMPLX(fac, 0.0_dp, dp), d_psq, nnr, d_w, nnr, (0.0_dp, 0.0_dp), &
                          d_g(1, nbnd * (j0 - 1) + 1), nbnd)
      CALL chk(istat, 'cublasZgemm')
    ENDDO
    IF (prof) THEN
      tb = egc_wall(.TRUE.); tp(3) = tp(3) + tb - ta
    ENDIF
  END SUBROUTINE egc_local_dev
  !
  LOGICAL FUNCTION egc_fdx_on()
    CHARACTER(LEN=8) :: ev
    IF (.NOT. fdx_read) THEN
      CALL GET_ENVIRONMENT_VARIABLE('RTSC_EPW_FDX', ev)
      fdx_req = TRIM(ev) == '1'
      CALL GET_ENVIRONMENT_VARIABLE('RTSC_EPW_FDX_CHECK', ev)
      fdx_chk = fdx_req .AND. TRIM(ev) == '1'
      fdx_read = .TRUE.
    ENDIF
    egc_fdx_on = fdx_req .AND. fdx_ok
  END FUNCTION egc_fdx_on
  !
  SUBROUTINE egc_local_fdx(npw, indk, ck, npwq, indq, cq)
    !! as egc_local_dev (local part into d_g; d_cq keeps psi_k+q(G)), in G-space on the pruned path
    INTEGER, INTENT(IN) :: npw, npwq
    INTEGER, INTENT(IN) :: indk(npw), indq(npwq)
    COMPLEX(dp), INTENT(IN) :: ck(ldc, nbnd), cq(ldc, nbnd)
    INTEGER :: istat, ncol, i
    TYPE(dim3) :: grd, blk
    REAL(dp) :: fac, ta, tb
    COMPLEX(dp), ALLOCATABLE :: ga(:,:), gb(:,:)
    ncol = nbnd * nvset
    IF (.NOT. ALLOCATED(d_dvp)) THEN
      ALLOCATE(d_dvp(ldc, nbnd * nvmax), d_lst(nbnd * nvmax))
      d_lst = (/ (i, i = 1, nbnd * nvmax) /)
    ENDIF
    IF (prof) ta = egc_wall(.TRUE.)
    istat = cudaMemcpy(d_ik, indk, npw);       CALL chk(istat, 'H2D indk')
    istat = cudaMemcpy(d_ck, ck, ldc * nbnd);  CALL chk(istat, 'H2D ck')
    istat = cudaMemcpy(d_cq, cq, ldc * nbnd);  CALL chk(istat, 'H2D cq')
    istat = fdx_geom1(nr, nrx, npwq, indq, ncol)
    IF (istat /= 0) THEN
      WRITE(*, '(a,i0,a)') 'epw_gpu_coarse: RTSC_EPW_FDX geometry unsupported (', istat, '), Parseval form kept'
      fdx_ok = .FALSE.
      CALL egc_local_dev(npw, indk, ck, npwq, indq, cq)
      RETURN
    ENDIF
    IF (prof) THEN
      tb = egc_wall(.TRUE.); tp(1) = tp(1) + tb - ta; ta = tb
    ENDIF
    d_psi = (0.0_dp, 0.0_dp)
    blk = dim3(256, 1, 1)
    grd = dim3((npw + 255) / 256, nbnd, 1)
    CALL k_scatter<<<grd, blk>>>(npw, nbnd, ldc, d_ik, d_ck, nnr, d_psi)
    istat = cufftExecZ2Z(plan, d_psi, d_psi, CUFFT_INVERSE); CALL chk(istat, 'cufft psi')
    IF (prof) THEN
      tb = egc_wall(.TRUE.); tp(2) = tp(2) + tb - ta; ta = tb
    ENDIF
    fac = 1.0_dp / DBLE(nr(1) * nr(2) * nr(3))
    d_dvp(:, 1:ncol) = (0.0_dp, 0.0_dp)
    ! column c = n + nbnd (nu - 1): band n of psi_k(r) (slot n, k = 1), perturbation nu of V
    istat = fdx_dpot_d(d_dvp, d_psi, d_v, d_lst, 0, ncol, nbnd, nvset, ldc, nnr, 0, fac, 0_cuda_stream_kind)
    IF (istat /= 0) CALL chk(istat, 'fdx_dpot_d')
    istat = cublasZgemm(hblas, CUBLAS_OP_C, CUBLAS_OP_N, nbnd, ncol, npwq, (1.0_dp, 0.0_dp), d_cq, ldc, &
                        d_dvp, ldc, (0.0_dp, 0.0_dp), d_g, nbnd)
    CALL chk(istat, 'cublasZgemm (fdx)')
    IF (prof) THEN
      tb = egc_wall(.TRUE.); tp(3) = tp(3) + tb - ta
    ENDIF
    IF (fdx_chk) THEN
      ALLOCATE(ga(nbnd, ncol), gb(nbnd, ncol))
      ga = d_g(:, 1:ncol)
      CALL egc_local_dev(npw, indk, ck, npwq, indq, cq)
      gb = d_g(:, 1:ncol)
      WRITE(*, '(a,i6,a,es10.3,a,es10.3)') '     rtsc epw fdx check: columns', ncol, ', max|d|/max ', &
           MAXVAL(ABS(ga - gb)) / MAXVAL(ABS(gb)), ', rel L2 ', SQRT(SUM(ABS(ga - gb)**2) / SUM(ABS(gb)**2))
      d_g(:, 1:ncol) = ga
      DEALLOCATE(ga, gb)
      fdx_chk = .FALSE.
    ENDIF
  END SUBROUTINE egc_local_fdx
  !
  SUBROUTINE egc_becs_set(npw, nkb_, ldv, vkbk, kgk, tpiba)
    !! RTSC_EPW_BEC: vkb at k (ldv x nkb_, first npw rows) and (k+G)_a (ldv x 3, tpiba units) for the next
    !! egc_local_nl, which then forms becp1 and alphap on the device instead of uploading them
    INTEGER, INTENT(IN) :: npw, nkb_, ldv
    COMPLEX(dp), INTENT(IN) :: vkbk(ldv, nkb_)
    REAL(dp), INTENT(IN) :: kgk(ldv, 3), tpiba
    INTEGER :: istat
    IF (ALLOCATED(d_vkbk)) THEN
      IF (SIZE(d_vkbk, 2) /= nkb_) DEALLOCATE(d_vkbk, d_ckt, d_kgk)
    ENDIF
    IF (.NOT. ALLOCATED(d_vkbk)) ALLOCATE(d_vkbk(ldc, nkb_), d_ckt(ldc, nbnd), d_kgk(ldc, 3))
    istat = cudaMemcpy2D(d_vkbk, ldc, vkbk, ldv, npw, nkb_); CALL chk(istat, 'H2D vkb_k')
    istat = cudaMemcpy2D(d_kgk, ldc, kgk, ldv, npw, 3); CALL chk(istat, 'H2D (k+G)')
    bec_npw = npw
    bec_tpiba = tpiba
    bec_dev = .TRUE.
  END SUBROUTINE egc_becs_set
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_kgpsi(t, c, kg, ld, npw, nb, a, tp)
    !! t(ig, n) = c(ig, n) * i tpiba (k+G)_a(ig)   (as the prolog's aux2)
    INTEGER, VALUE :: ld, npw, nb, a
    REAL(dp), VALUE :: tp
    COMPLEX(dp), DEVICE :: t(ld, nb), c(ld, nb)
    REAL(dp), DEVICE :: kg(ld, 3)
    INTEGER :: ig, n
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    n = blockIdx%y
    IF (ig <= npw) t(ig, n) = c(ig, n) * CMPLX(0.0_dp, tp * kg(ig, a), dp)
  END SUBROUTINE k_kgpsi
  !
  SUBROUTINE egc_nl_set(nkb_, nat3_, nv, dmat, atom, uq)
    !! Per ep_coarse_compute call: dense D (nkb x nkb, block-diagonal deff), the atom of each projector
    !! (1-based) and the perturbation patterns uq(3 nat, nv) exactly as vanilla uses them (conjugated
    !! under time reversal).
    INTEGER, INTENT(IN) :: nkb_, nat3_, nv, atom(nkb_)
    COMPLEX(dp), INTENT(IN) :: dmat(nkb_, nkb_), uq(nat3_, nv)
    IF (nkbs /= nkb_ .OR. nat3s /= nat3_ .OR. nvs < nv) THEN
      IF (ALLOCATED(d_vkb)) DEALLOCATE(d_vkb, d_auxv, d_becq, d_bp, d_al, d_dm, d_db, d_da, d_u, d_kqg, d_at)
      nkbs = nkb_; nat3s = nat3_; nvs = MAX(nv, nvmax)
      ALLOCATE(d_vkb(ldc, nkbs), d_auxv(ldc, nkbs), d_becq(nkbs, nbnd, 0:3), d_bp(nkbs, nbnd), d_al(nkbs, nbnd, 3))
      ALLOCATE(d_dm(nkbs, nkbs), d_db(nkbs, nbnd), d_da(nkbs, nbnd, 3), d_u(nat3s, nvs), d_kqg(ldc, 3), d_at(nkbs))
    ENDIF
    d_dm = dmat
    d_at = atom
    d_u(:, 1:nv) = uq(:, 1:nv)
  END SUBROUTINE egc_nl_set
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_kqg(auxv, vkb, kqg, ld, npwq, nkb, ipol)
    INTEGER, VALUE :: ld, npwq, nkb, ipol
    COMPLEX(dp), DEVICE :: auxv(ld, nkb), vkb(ld, nkb)
    REAL(dp), DEVICE :: kqg(ld, 3)
    INTEGER :: ig, kb
    ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    kb = blockIdx%y
    IF (ig <= npwq) auxv(ig, kb) = vkb(ig, kb) * kqg(ig, ipol)
  END SUBROUTINE k_kqg
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_nlc(g, becq, da, db, u, at, nkb, nb, nat3, nvm, ctp)
    !! g(m, n + nb (nu-1)) += sum_kb sum_a conj(becq_0(kb,m)) u(3(at-1)+a, nu) da_a(kb,n)
    !!                                     + ctp conj(becq_a(kb,m)) u(3(at-1)+a, nu) db(kb,n),   ctp = -i tpiba
    INTEGER, VALUE :: nkb, nb, nat3, nvm
    COMPLEX(dp), VALUE :: ctp
    COMPLEX(dp), DEVICE :: g(nb, *), becq(nkb, nb, 0:3), da(nkb, nb, 3), db(nkb, nb), u(nat3, nvm)
    INTEGER, DEVICE :: at(nkb)
    INTEGER :: m, n, nu, kb, a, mu
    COMPLEX(dp) :: acc, uu
    m = blockIdx%x; nu = blockIdx%y; n = threadIdx%x
    IF (n > nb) RETURN
    acc = (0.0_dp, 0.0_dp)
    DO kb = 1, nkb
      mu = 3 * (at(kb) - 1)
      DO a = 1, 3
        uu = u(mu + a, nu)
        acc = acc + CONJG(becq(kb, m, 0)) * uu * da(kb, n, a) + ctp * CONJG(becq(kb, m, a)) * uu * db(kb, n)
      ENDDO
    ENDDO
    g(m, n + nb * (nu - 1)) = g(m, n + nb * (nu - 1)) + acc
  END SUBROUTINE k_nlc
  !
  SUBROUTINE egc_local_nl(npw, indk, ck, npwq, indq, cq, vkb, ldv, kqg, becp1, ldb, al1, al2, al3, tpiba, g)
    !! Local + nonlocal electron-phonon matrix elements of one k for all nvset perturbations.
    !! vkb(ldv, nkb): projectors at k+q on the k+q sphere; kqg(npwq, 3) = (k+q+G)_a (tpiba units, as
    !! vanilla's auxv; leading dimension ldv); becp1, al1..3 (ldb, nbnd): <beta_k | psi_k> and its k+G derivatives.
    INTEGER, INTENT(IN) :: npw, npwq, ldv, ldb
    INTEGER, INTENT(IN) :: indk(npw), indq(npwq)
    COMPLEX(dp), INTENT(IN) :: ck(ldc, nbnd), cq(ldc, nbnd), vkb(ldv, nkbs)
    REAL(dp), INTENT(IN) :: kqg(ldv, 3), tpiba
    COMPLEX(dp), INTENT(IN) :: becp1(ldb, nbnd), al1(ldb, nbnd), al2(ldb, nbnd), al3(ldb, nbnd)
    COMPLEX(dp), INTENT(OUT) :: g(nbnd, nbnd, nvset)
    INTEGER :: istat, a
    REAL(dp) :: ta, tb
    COMPLEX(dp), PARAMETER :: one = (1.0_dp, 0.0_dp), zero = (0.0_dp, 0.0_dp)
    IF (prof) THEN
      ta = egc_wall(.TRUE.); np_calls = np_calls + 1
    ENDIF
    ! all uploads first (synchronous copies), so that only the final D2H waits for the GPU work
    IF (nkbs > 0) THEN
      istat = cudaMemcpy2D(d_vkb, ldc, vkb, ldv, npwq, nkbs); CALL chk(istat, 'H2D vkb')
      istat = cudaMemcpy2D(d_kqg, ldc, kqg, ldv, npwq, 3); CALL chk(istat, 'H2D kqg')
    ENDIF
    IF (nkbs > 0 .AND. .NOT. bec_dev) THEN
      istat = cudaMemcpy2D(d_bp, nkbs, becp1, ldb, nkbs, nbnd); CALL chk(istat, 'H2D becp1')
      istat = cudaMemcpy2D(d_al(1, 1, 1), nkbs, al1, ldb, nkbs, nbnd); CALL chk(istat, 'H2D alphap')
      istat = cudaMemcpy2D(d_al(1, 1, 2), nkbs, al2, ldb, nkbs, nbnd); CALL chk(istat, 'H2D alphap')
      istat = cudaMemcpy2D(d_al(1, 1, 3), nkbs, al3, ldb, nkbs, nbnd); CALL chk(istat, 'H2D alphap')
    ENDIF
    IF (prof) THEN
      tb = egc_wall(.TRUE.); tp(1) = tp(1) + tb - ta
    ENDIF
    IF (egc_fdx_on()) THEN
      CALL egc_local_fdx(npw, indk, ck, npwq, indq, cq)
    ELSE
      CALL egc_local_dev(npw, indk, ck, npwq, indq, cq)
    ENDIF
    IF (prof) ta = egc_wall(.TRUE.)
    IF (nkbs > 0 .AND. bec_dev) THEN
      ! RTSC_EPW_BEC: becp1 = vkb_k^H psi_k, alphap_a = vkb_k^H [i tpiba (k+G)_a psi_k]   (d_ck = psi_k(G))
      istat = cublasZgemm(hblas, CUBLAS_OP_C, CUBLAS_OP_N, nkbs, nbnd, bec_npw, one, d_vkbk, ldc, d_ck, ldc, zero, &
                          d_bp, nkbs)
      CALL chk(istat, 'cublasZgemm becp1')
      DO a = 1, 3
        CALL k_kgpsi<<<dim3((bec_npw + 255) / 256, nbnd, 1), 256>>>(d_ckt, d_ck, d_kgk, ldc, bec_npw, nbnd, a, bec_tpiba)
        istat = cublasZgemm(hblas, CUBLAS_OP_C, CUBLAS_OP_N, nkbs, nbnd, bec_npw, one, d_vkbk, ldc, d_ckt, ldc, zero, &
                            d_al(1, 1, a), nkbs)
        CALL chk(istat, 'cublasZgemm alphap')
      ENDDO
      bec_dev = .FALSE.
    ENDIF
    IF (nkbs > 0) THEN
      ! becq_0 = vkb^H psi_k+q ; becq_a = [vkb (k+q+G)_a]^H psi_k+q
      istat = cublasZgemm(hblas, CUBLAS_OP_C, CUBLAS_OP_N, nkbs, nbnd, npwq, one, d_vkb, ldc, d_cq, ldc, zero, &
                          d_becq(1, 1, 0), nkbs)
      CALL chk(istat, 'cublasZgemm becq0')
      DO a = 1, 3
        CALL k_kqg<<<dim3((npwq + 255) / 256, nkbs, 1), 256>>>(d_auxv, d_vkb, d_kqg, ldc, npwq, nkbs, a)
        istat = cublasZgemm(hblas, CUBLAS_OP_C, CUBLAS_OP_N, nkbs, nbnd, npwq, one, d_auxv, ldc, d_cq, ldc, zero, &
                            d_becq(1, 1, a), nkbs)
        CALL chk(istat, 'cublasZgemm becq')
      ENDDO
      ! D becp1 and D alphap_a
      istat = cublasZgemm(hblas, CUBLAS_OP_N, CUBLAS_OP_N, nkbs, nbnd, nkbs, one, d_dm, nkbs, d_bp, nkbs, zero, &
                          d_db, nkbs)
      CALL chk(istat, 'cublasZgemm Db')
      istat = cublasZgemmStridedBatched(hblas, CUBLAS_OP_N, CUBLAS_OP_N, nkbs, nbnd, nkbs, one, d_dm, nkbs, 0_8, &
                          d_al, nkbs, INT(nkbs, 8) * nbnd, zero, d_da, nkbs, INT(nkbs, 8) * nbnd, 3)
      CALL chk(istat, 'cublasZgemm Da')
      CALL k_nlc<<<dim3(nbnd, nvset, 1), 32 * ((nbnd + 31) / 32)>>>(d_g, d_becq, d_da, d_db, d_u, d_at, nkbs, nbnd, &
           nat3s, nvs, CMPLX(0.0_dp, -tpiba, dp))
      istat = cudaGetLastError(); CALL chk(istat, 'k_nlc')
    ENDIF
    IF (prof) THEN
      tb = egc_wall(.TRUE.); tp(4) = tp(4) + tb - ta; ta = tb
    ENDIF
    istat = cudaMemcpy(g, d_g, nbnd * nbnd * nvset); CALL chk(istat, 'D2H g')
    IF (prof) THEN
      tb = egc_wall(.FALSE.); tp(5) = tp(5) + tb - ta
    ENDIF
  END SUBROUTINE egc_local_nl
  !
END MODULE epw_gpu_coarse
