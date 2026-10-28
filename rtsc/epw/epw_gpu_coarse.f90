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
  PUBLIC :: egc_init, egc_set_v, egc_local, egc_finalize, egc_ready
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
    INTEGER :: istat, j0, nm
    TYPE(dim3) :: grd, blk
    REAL(dp) :: fac
    !
    istat = cudaMemcpy(d_ik, indk, npw);       CALL chk(istat, 'H2D indk')
    istat = cudaMemcpy(d_iq, indq, npwq);      CALL chk(istat, 'H2D indq')
    istat = cudaMemcpy(d_ck, ck, ldc * nbnd);  CALL chk(istat, 'H2D ck')
    istat = cudaMemcpy(d_cq, cq, ldc * nbnd);  CALL chk(istat, 'H2D cq')
    d_psi = (0.0_dp, 0.0_dp)
    d_psq = (0.0_dp, 0.0_dp)
    blk = dim3(256, 1, 1)
    grd = dim3((npw + 255) / 256, nbnd, 1)
    CALL k_scatter<<<grd, blk>>>(npw, nbnd, ldc, d_ik, d_ck, nnr, d_psi)
    grd = dim3((npwq + 255) / 256, nbnd, 1)
    CALL k_scatter<<<grd, blk>>>(npwq, nbnd, ldc, d_iq, d_cq, nnr, d_psq)
    istat = cufftExecZ2Z(plan, d_psi, d_psi, CUFFT_INVERSE); CALL chk(istat, 'cufft psi')
    istat = cufftExecZ2Z(plan, d_psq, d_psq, CUFFT_INVERSE); CALL chk(istat, 'cufft psq')
    fac = 1.0_dp / DBLE(nr(1) * nr(2) * nr(3))
    DO j0 = 1, nvset, nmc
      nm = MIN(nmc, nvset - j0 + 1)
      grd = dim3((nnr + 255) / 256, nbnd * nm, 1)
      CALL k_mulv<<<grd, blk>>>(nnr, nbnd, nm, j0, nvmax, d_v, d_psi, d_w)
      istat = cublasZgemm(hblas, CUBLAS_OP_C, CUBLAS_OP_N, nbnd, nbnd * nm, nnr, &
                          CMPLX(fac, 0.0_dp, dp), d_psq, nnr, d_w, nnr, (0.0_dp, 0.0_dp), &
                          d_g(1, nbnd * (j0 - 1) + 1), nbnd)
      CALL chk(istat, 'cublasZgemm')
    ENDDO
    istat = cudaMemcpy(g, d_g, nbnd * nbnd * nvset); CALL chk(istat, 'D2H g')
  END SUBROUTINE egc_local
  !
END MODULE epw_gpu_coarse
