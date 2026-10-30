! pftest: sth_pfft (pruned fused local operator) against the engine's cuFFT chain on a synthetic
! sphere (KERNEL-LOG lever 1). nvfortran -O3 -cuda -gpu=ccXX -cudalib=cufft sth_pfft.f90 pftest.f90
!   ./pftest n1 n2 n3 [ncol] [frac]   (frac: sphere radius as a fraction of n/2, default 0.5)
MODULE pft_ks
  USE cudafor
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp = KIND(1.d0), sp = KIND(1.0)
CONTAINS
  ATTRIBUTES(GLOBAL) SUBROUTINE k_scat(box, x, inv, nnr, npw)
    INTEGER, VALUE :: nnr, npw
    COMPLEX(sp) :: box(nnr, *), x(npw, *)
    INTEGER :: inv(nnr)
    INTEGER :: ir, j, ig
    j = blockIdx%y; ir = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ir <= nnr) THEN
      ig = inv(ir)
      IF (ig > 0) THEN
        box(ir, j) = x(ig, j)
      ELSE
        box(ir, j) = (0.0_sp, 0.0_sp)
      ENDIF
    ENDIF
  END SUBROUTINE k_scat
  ATTRIBUTES(GLOBAL) SUBROUTINE k_vmul(box, v, nnr)
    INTEGER, VALUE :: nnr
    COMPLEX(sp) :: box(nnr, *)
    REAL(sp) :: v(nnr)
    INTEGER :: ir, j
    j = blockIdx%y; ir = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ir <= nnr) box(ir, j) = box(ir, j) * v(ir)
  END SUBROUTINE k_vmul
  ATTRIBUTES(GLOBAL) SUBROUTINE k_gath(y, x, box, idx, nnr, npw, g2, e, scale)
    INTEGER, VALUE :: nnr, npw
    REAL(sp), VALUE :: scale
    COMPLEX(sp) :: y(npw, *), x(npw, *), box(nnr, *)
    INTEGER :: idx(npw)
    REAL(sp) :: g2(npw)
    REAL(dp) :: e(*)
    INTEGER :: ig, j
    COMPLEX(sp) :: xv
    j = blockIdx%y; ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ig <= npw) THEN
      xv = x(ig, j)
      y(ig, j) = g2(ig) * xv + box(idx(ig), j) * scale - REAL(e(j), sp) * xv
    ENDIF
  END SUBROUTINE k_gath
END MODULE pft_ks

PROGRAM pftest_s
  USE cudafor
  USE cufft
  USE pft_ks
  USE sth_pfft_s
  IMPLICIT NONE
  INTEGER :: n1, n2, n3, nb, nnr, npw, i1, i2, i3, m1, m2, m3, ig, s, istat, rep, nrep, ier, nst, nxa, pch
  CHARACTER(32) :: arg
  INTEGER, ALLOCATABLE :: inv(:), idx(:), lst(:)
  INTEGER, DEVICE, ALLOCATABLE :: d_inv(:), d_idx(:), d_list(:), d_npwq(:)
  COMPLEX(sp), DEVICE, ALLOCATABLE :: box(:,:), x(:,:), y(:,:)
  REAL(sp), DEVICE, ALLOCATABLE :: v(:), g2(:)
  REAL(dp), DEVICE, ALLOCATABLE :: et(:)
  REAL(dp), ALLOCATABLE :: hv(:), hg(:), he(:)
  COMPLEX(dp), ALLOCATABLE :: hx(:,:), y1(:,:), y2(:,:)
  REAL(dp) :: g1, gg2, g3, r2, ta, tp, frac
  REAL(sp) :: scale
  TYPE(cudaEvent) :: e0, e1
  REAL :: ms
  INTEGER :: pA, n(3), nrr(3)
  INTEGER(KIND=cuda_stream_kind) :: strm
  CALL get_command_argument(1, arg); READ(arg, *) n1
  CALL get_command_argument(2, arg); READ(arg, *) n2
  CALL get_command_argument(3, arg); READ(arg, *) n3
  nb = 256; frac = 0.5_dp
  IF (command_argument_count() >= 4) THEN
    CALL get_command_argument(4, arg); READ(arg, *) nb
  ENDIF
  IF (command_argument_count() >= 5) THEN
    CALL get_command_argument(5, arg); READ(arg, *) frac
  ENDIF
  nnr = n1 * n2 * n3
  ALLOCATE(inv(nnr)); inv = 0; npw = 0
  ! a sphere shifted by a non-lattice k (as k+q+G): |m + 0.3| <= frac * n/2 per direction
  DO i3 = 1, n3
    DO i2 = 1, n2
      DO i1 = 1, n1
        m1 = i1 - 1; IF (m1 > n1 / 2) m1 = m1 - n1
        m2 = i2 - 1; IF (m2 > n2 / 2) m2 = m2 - n2
        m3 = i3 - 1; IF (m3 > n3 / 2) m3 = m3 - n3
        g1 = (m1 + 0.3_dp) / (frac * 0.5_dp * n1); gg2 = (m2 - 0.2_dp) / (frac * 0.5_dp * n2)
        g3 = (m3 + 0.1_dp) / (frac * 0.5_dp * n3)
        r2 = g1 * g1 + gg2 * gg2 + g3 * g3
        IF (r2 <= 1.0_dp) THEN
          npw = npw + 1; inv(i1 + (i2 - 1) * n1 + (i3 - 1) * n1 * n2) = npw
        ENDIF
      ENDDO
    ENDDO
  ENDDO
  ALLOCATE(idx(npw))
  DO i3 = 1, nnr
    IF (inv(i3) > 0) idx(inv(i3)) = i3
  ENDDO
  ALLOCATE(d_inv(nnr), d_idx(npw), d_list(nb), lst(nb), d_npwq(1))
  d_inv = inv; d_idx = idx
  DO s = 1, nb
    lst(s) = s
  ENDDO
  d_list = lst; d_npwq = npw
  ALLOCATE(box(nnr, nb), x(npw, nb), y(npw, nb), v(nnr), hv(nnr), hx(npw, nb), y1(npw, nb), y2(npw, nb))
  ALLOCATE(g2(npw), et(nb), hg(npw), he(nb))
  CALL RANDOM_NUMBER(hv); v = hv - 0.5_dp
  CALL RANDOM_NUMBER(hg); g2 = 5.0_dp * hg
  CALL RANDOM_NUMBER(he); et = he
  DO s = 1, nb
    DO ig = 1, npw
      CALL RANDOM_NUMBER(g1); CALL RANDOM_NUMBER(gg2); hx(ig, s) = CMPLX(g1 - 0.5_dp, gg2 - 0.5_dp, dp)
    ENDDO
  ENDDO
  x = CMPLX(hx, KIND=sp)
  scale = 1.0_sp / nnr
  n = (/ n3, n2, n1 /)
  istat = cufftPlanMany(pA, 3, n, n, 1, nnr, n, 1, nnr, CUFFT_C2C, nb)
  istat = cudaStreamCreate(strm)
  istat = cufftSetStream(pA, strm)
  nrr = (/ n1, n2, n3 /)
  CALL pfs_setup(nrr, nrr, 1, npw, (/ npw /), idx, nb, ier)
  IF (ier /= 0) THEN
    WRITE(*, *) 'pfft_setup ierr', ier
    STOP
  ENDIF
  CALL pfs_info(nst, nxa, pch)
  WRITE(*, '(a,3i5,a,i7,a,i6,a,f6.3,a,i4,a,i5)') 'grid', n1, n2, n3, ' npw', npw, ' sticks', nst, &
       ' (', REAL(nst) / (n1 * n2), ') x-columns', nxa, ' cols', nb
  istat = cudaEventCreate(e0); istat = cudaEventCreate(e1)
  nrep = 10
  DO rep = 0, nrep
    IF (rep == 1) istat = cudaEventRecord(e0, strm)
    CALL k_scat<<<dim3((nnr + 255) / 256, nb, 1), 256, 0, strm>>>(box, x, d_inv, nnr, npw)
    istat = cufftExecC2C(pA, box, box, CUFFT_INVERSE)
    CALL k_vmul<<<dim3((nnr + 255) / 256, nb, 1), 256, 0, strm>>>(box, v, nnr)
    istat = cufftExecC2C(pA, box, box, CUFFT_FORWARD)
    CALL k_gath<<<dim3((npw + 255) / 256, nb, 1), 256, 0, strm>>>(y, x, box, d_idx, nnr, npw, g2, et, scale)
  ENDDO
  istat = cudaEventRecord(e1, strm); istat = cudaEventSynchronize(e1); istat = cudaEventElapsedTime(ms, e0, e1)
  ta = 1.0e3_dp * ms / (nrep * nb); y1 = y
  y = (0.0_sp, 0.0_sp)
  DO rep = 0, nrep
    IF (rep == 1) istat = cudaEventRecord(e0, strm)
    ! one "k" with nslot = nb, npert = 1: column s has k = 1, band n = s (etk(n, 1) = et(s))
    CALL pfs_local(x, y, d_list, nb, nb, 1, npw, 1, nb, v, d_npwq, g2, et, scale, strm)
  ENDDO
  istat = cudaEventRecord(e1, strm); istat = cudaEventSynchronize(e1); istat = cudaEventElapsedTime(ms, e0, e1)
  istat = cudaGetLastError()
  IF (istat /= 0) WRITE(*, *) 'CUDA error: ', cudaGetErrorString(istat)
  tp = 1.0e3_dp * ms / (nrep * nb); y2 = y
  WRITE(*, '(a,f8.2,a,f8.2,a,f6.2,a,es9.2,a,es9.2)') '  FP32 us/column: cuFFT chain ', ta, '   pfft ', tp, &
       '   ratio ', ta / tp, '   max|dy|/max|y| ', MAXVAL(ABS(y1 - y2)) / MAXVAL(ABS(y1)), &
       '   rel L2 ', SQRT(SUM(ABS(y1 - y2)**2) / SUM(ABS(y1)**2))
END PROGRAM pftest_s
