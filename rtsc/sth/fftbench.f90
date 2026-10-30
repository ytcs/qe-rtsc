! fftbench: per-column cost of the engine's local-potential chain (scatter, FFT^-1, xV, FFT, gather)
! for cuFFT plan variants (groups/compute/KERNEL-LOG.md, Step 2 probe). Standalone:
!   nvfortran -O3 -cuda -gpu=cc86 -cudalib=cufft fftbench.f90 -o fftbench ; ./fftbench n1 n2 n3 [B]
! Variant A: current engine (full-box scatter, 3D Z2Z inverse, xV, 3D Z2Z forward, gather).
! Variant P: pruned z (scatter to the PW sticks, 1D z-FFT on sticks only, expand to planes, batched 2D
!            FFT per z-plane, xV, 2D forward, extract sticks, 1D z-FFT, gather).
! Both in FP64 (Z2Z) and FP32 (C2C). The sphere is |G|^2 <= (n/4)^2 in grid units (ecutrho = 4 ecutwfc).
MODULE fb
  USE cudafor
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp = KIND(1.d0), sp = KIND(1.0)
CONTAINS
  ATTRIBUTES(GLOBAL) SUBROUTINE k_scat(box, x, inv, nnr, npw)
    INTEGER, VALUE :: nnr, npw
    COMPLEX(dp) :: box(nnr, *), x(npw, *)
    INTEGER :: inv(nnr)
    INTEGER :: ir, j, ig
    j = blockIdx%y; ir = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ir <= nnr) THEN
      ig = inv(ir)
      IF (ig > 0) THEN
        box(ir, j) = x(ig, j)
      ELSE
        box(ir, j) = (0.0_dp, 0.0_dp)
      ENDIF
    ENDIF
  END SUBROUTINE k_scat
  ATTRIBUTES(GLOBAL) SUBROUTINE k_vmul(box, v, nnr)
    INTEGER, VALUE :: nnr
    COMPLEX(dp) :: box(nnr, *)
    REAL(dp) :: v(nnr)
    INTEGER :: ir, j
    j = blockIdx%y; ir = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ir <= nnr) box(ir, j) = box(ir, j) * v(ir)
  END SUBROUTINE k_vmul
  ATTRIBUTES(GLOBAL) SUBROUTINE k_gath(y, x, box, idx, nnr, npw)
    INTEGER, VALUE :: nnr, npw
    COMPLEX(dp) :: y(npw, *), x(npw, *), box(nnr, *)
    INTEGER :: idx(npw)
    INTEGER :: ig, j
    j = blockIdx%y; ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ig <= npw) y(ig, j) = 0.5_dp * x(ig, j) + box(idx(ig), j)
  END SUBROUTINE k_gath
  ! sticks array t(n3, nst, B): z contiguous
  ATTRIBUTES(GLOBAL) SUBROUTINE k_scat_st(t, x, sinv, nz, npw)
    INTEGER, VALUE :: nz, npw
    COMPLEX(dp) :: t(nz, *), x(npw, *)
    INTEGER :: sinv(nz)
    INTEGER :: p, j, ig
    j = blockIdx%y; p = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (p <= nz) THEN
      ig = sinv(p)
      IF (ig > 0) THEN
        t(p, j) = x(ig, j)
      ELSE
        t(p, j) = (0.0_dp, 0.0_dp)
      ENDIF
    ENDIF
  END SUBROUTINE k_scat_st
  ! expand sticks to the full box (zero elsewhere): one thread per box point; smap(i1,i2) = stick or 0
  ATTRIBUTES(GLOBAL) SUBROUTINE k_expand(box, t, smap, n1, n2, n3, nst)
    INTEGER, VALUE :: n1, n2, n3, nst
    COMPLEX(dp) :: box(n1 * n2 * n3, *), t(n3, nst, *)
    INTEGER :: smap(n1 * n2)
    INTEGER :: ir, j, ixy, i3, s
    j = blockIdx%y; ir = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ir <= n1 * n2 * n3) THEN
      ixy = MOD(ir - 1, n1 * n2) + 1; i3 = (ir - 1) / (n1 * n2) + 1
      s = smap(ixy)
      IF (s > 0) THEN
        box(ir, j) = t(i3, s, j)
      ELSE
        box(ir, j) = (0.0_dp, 0.0_dp)
      ENDIF
    ENDIF
  END SUBROUTINE k_expand
  ATTRIBUTES(GLOBAL) SUBROUTINE k_extract(t, box, sxy, n1, n2, n3, nst)
    INTEGER, VALUE :: n1, n2, n3, nst
    COMPLEX(dp) :: box(n1 * n2 * n3, *), t(n3, nst, *)
    INTEGER :: sxy(nst)
    INTEGER :: p, j, i3, s
    j = blockIdx%y; p = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (p <= n3 * nst) THEN
      i3 = MOD(p - 1, n3) + 1; s = (p - 1) / n3 + 1
      t(i3, s, j) = box(sxy(s) + (i3 - 1) * n1 * n2, j)
    ENDIF
  END SUBROUTINE k_extract
  ATTRIBUTES(GLOBAL) SUBROUTINE k_gath_st(y, x, t, sidx, nz, npw)
    INTEGER, VALUE :: nz, npw
    COMPLEX(dp) :: y(npw, *), x(npw, *), t(nz, *)
    INTEGER :: sidx(npw)
    INTEGER :: ig, j
    j = blockIdx%y; ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ig <= npw) y(ig, j) = 0.5_dp * x(ig, j) + t(sidx(ig), j)
  END SUBROUTINE k_gath_st
  ! FP32 copies
  ATTRIBUTES(GLOBAL) SUBROUTINE s_scat(box, x, inv, nnr, npw)
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
  END SUBROUTINE s_scat
  ATTRIBUTES(GLOBAL) SUBROUTINE s_vmul(box, v, nnr)
    INTEGER, VALUE :: nnr
    COMPLEX(sp) :: box(nnr, *)
    REAL(sp) :: v(nnr)
    INTEGER :: ir, j
    j = blockIdx%y; ir = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ir <= nnr) box(ir, j) = box(ir, j) * v(ir)
  END SUBROUTINE s_vmul
  ATTRIBUTES(GLOBAL) SUBROUTINE s_gath(y, x, box, idx, nnr, npw)
    INTEGER, VALUE :: nnr, npw
    COMPLEX(sp) :: y(npw, *), x(npw, *), box(nnr, *)
    INTEGER :: idx(npw)
    INTEGER :: ig, j
    j = blockIdx%y; ig = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (ig <= npw) y(ig, j) = 0.5_sp * x(ig, j) + box(idx(ig), j)
  END SUBROUTINE s_gath
END MODULE fb

PROGRAM fftbench
  USE cudafor
  USE cufft
  USE fb
  IMPLICIT NONE
  INTEGER :: n1, n2, n3, nb, nnr, npw, nst, i1, i2, i3, m1, m2, m3, ig, s, istat, rep, nrep, ixy
  INTEGER :: pA, pAi, pz, p2, sA, sz, s2
  CHARACTER(32) :: arg
  INTEGER, ALLOCATABLE :: inv(:), idx(:), smap(:), sxy(:), sinv(:), sidx(:)
  INTEGER, DEVICE, ALLOCATABLE :: d_inv(:), d_idx(:), d_smap(:), d_sxy(:), d_sinv(:), d_sidx(:)
  COMPLEX(dp), DEVICE, ALLOCATABLE :: box(:,:), x(:,:), y(:,:), t(:,:)
  COMPLEX(sp), DEVICE, ALLOCATABLE :: bs(:,:), xs(:,:), ys(:,:)
  REAL(dp), DEVICE, ALLOCATABLE :: v(:)
  REAL(sp), DEVICE, ALLOCATABLE :: vs(:)
  REAL(dp), ALLOCATABLE :: hv(:)
  COMPLEX(dp), ALLOCATABLE :: hx(:,:), y1(:,:), y2(:,:)
  REAL(dp) :: g1, g2, g3, r2, ta, tp, ts
  TYPE(cudaEvent) :: e0, e1
  REAL :: ms
  INTEGER :: n(3), n2d(2), nz1(1)
  !
  CALL get_command_argument(1, arg); READ(arg, *) n1
  CALL get_command_argument(2, arg); READ(arg, *) n2
  CALL get_command_argument(3, arg); READ(arg, *) n3
  nb = 256
  IF (command_argument_count() >= 4) THEN
    CALL get_command_argument(4, arg); READ(arg, *) nb
  ENDIF
  nnr = n1 * n2 * n3
  ALLOCATE(inv(nnr), smap(n1 * n2)); inv = 0; smap = 0
  ! sphere |G| <= n/4 in each direction (ellipsoid in grid units)
  npw = 0; nst = 0
  DO i3 = 1, n3
    DO i2 = 1, n2
      DO i1 = 1, n1
        m1 = i1 - 1; IF (m1 > n1 / 2) m1 = m1 - n1
        m2 = i2 - 1; IF (m2 > n2 / 2) m2 = m2 - n2
        m3 = i3 - 1; IF (m3 > n3 / 2) m3 = m3 - n3
        g1 = REAL(m1, dp) / (0.25_dp * n1); g2 = REAL(m2, dp) / (0.25_dp * n2); g3 = REAL(m3, dp) / (0.25_dp * n3)
        r2 = g1 * g1 + g2 * g2 + g3 * g3
        IF (r2 <= 1.0_dp) THEN
          npw = npw + 1
          inv(i1 + (i2 - 1) * n1 + (i3 - 1) * n1 * n2) = npw
          ixy = i1 + (i2 - 1) * n1
          IF (smap(ixy) == 0) THEN
            nst = nst + 1; smap(ixy) = nst
          ENDIF
        ENDIF
      ENDDO
    ENDDO
  ENDDO
  ALLOCATE(idx(npw), sxy(nst), sinv(n3 * nst), sidx(npw)); sinv = 0
  DO i3 = 1, n3
    DO ixy = 1, n1 * n2
      ig = inv(ixy + (i3 - 1) * n1 * n2)
      IF (ig > 0) THEN
        idx(ig) = ixy + (i3 - 1) * n1 * n2
        s = smap(ixy); sxy(s) = ixy
        sinv(i3 + (s - 1) * n3) = ig; sidx(ig) = i3 + (s - 1) * n3
      ENDIF
    ENDDO
  ENDDO
  WRITE(*, '(a,3i5,a,i7,a,i6,a,f6.3,a,f6.3,a,i5)') 'grid', n1, n2, n3, ' npw', npw, ' sticks', nst, &
     ' (', REAL(nst) / (n1 * n2), ' of columns)  sphere/box', REAL(npw) / nnr, ' batch', nb
  ALLOCATE(d_inv(nnr), d_idx(npw), d_smap(n1 * n2), d_sxy(nst), d_sinv(n3 * nst), d_sidx(npw))
  d_inv = inv; d_idx = idx; d_smap = smap; d_sxy = sxy; d_sinv = sinv; d_sidx = sidx
  ALLOCATE(box(nnr, nb), x(npw, nb), y(npw, nb), t(n3 * nst, nb), v(nnr), hv(nnr), hx(npw, nb), y1(npw, nb), y2(npw, nb))
  CALL RANDOM_NUMBER(hv); v = hv
  DO s = 1, nb
    DO ig = 1, npw
      CALL RANDOM_NUMBER(g1); CALL RANDOM_NUMBER(g2); hx(ig, s) = CMPLX(g1 - 0.5_dp, g2 - 0.5_dp, dp)
    ENDDO
  ENDDO
  x = hx
  n = (/ n3, n2, n1 /)
  istat = cufftPlanMany(pA, 3, n, n, 1, nnr, n, 1, nnr, CUFFT_Z2Z, nb)
  n2d = (/ n2, n1 /)
  istat = istat + cufftPlanMany(p2, 2, n2d, n2d, 1, n1 * n2, n2d, 1, n1 * n2, CUFFT_Z2Z, n3 * nb)
  nz1 = (/ n3 /)
  istat = istat + cufftPlanMany(pz, 1, nz1, nz1, 1, n3, nz1, 1, n3, CUFFT_Z2Z, nst * nb)
  IF (istat /= 0) STOP 'plan'
  istat = cudaEventCreate(e0); istat = cudaEventCreate(e1)
  nrep = 10
  ! ---- A: current chain
  DO rep = 0, nrep
    IF (rep == 1) istat = cudaEventRecord(e0, 0)
    CALL k_scat<<<dim3((nnr + 255) / 256, nb, 1), 256>>>(box, x, d_inv, nnr, npw)
    istat = cufftExecZ2Z(pA, box, box, CUFFT_INVERSE)
    CALL k_vmul<<<dim3((nnr + 255) / 256, nb, 1), 256>>>(box, v, nnr)
    istat = cufftExecZ2Z(pA, box, box, CUFFT_FORWARD)
    CALL k_gath<<<dim3((npw + 255) / 256, nb, 1), 256>>>(y, x, box, d_idx, nnr, npw)
  ENDDO
  istat = cudaEventRecord(e1, 0); istat = cudaEventSynchronize(e1); istat = cudaEventElapsedTime(ms, e0, e1)
  ta = 1.0e3_dp * ms / (nrep * nb); y1 = y
  ! ---- P: pruned z + batched 2D
  DO rep = 0, nrep
    IF (rep == 1) istat = cudaEventRecord(e0, 0)
    CALL k_scat_st<<<dim3((n3 * nst + 255) / 256, nb, 1), 256>>>(t, x, d_sinv, n3 * nst, npw)
    istat = cufftExecZ2Z(pz, t, t, CUFFT_INVERSE)
    CALL k_expand<<<dim3((nnr + 255) / 256, nb, 1), 256>>>(box, t, d_smap, n1, n2, n3, nst)
    istat = cufftExecZ2Z(p2, box, box, CUFFT_INVERSE)
    CALL k_vmul<<<dim3((nnr + 255) / 256, nb, 1), 256>>>(box, v, nnr)
    istat = cufftExecZ2Z(p2, box, box, CUFFT_FORWARD)
    CALL k_extract<<<dim3((n3 * nst + 255) / 256, nb, 1), 256>>>(t, box, d_sxy, n1, n2, n3, nst)
    istat = cufftExecZ2Z(pz, t, t, CUFFT_FORWARD)
    CALL k_gath_st<<<dim3((npw + 255) / 256, nb, 1), 256>>>(y, x, t, d_sidx, n3 * nst, npw)
  ENDDO
  istat = cudaEventRecord(e1, 0); istat = cudaEventSynchronize(e1); istat = cudaEventElapsedTime(ms, e0, e1)
  tp = 1.0e3_dp * ms / (nrep * nb); y2 = y
  WRITE(*, '(a,f8.2,a,f8.2,a,f6.2,a,es9.2)') '  FP64 us/column: A (current) ', ta, '   P (pruned z + 2D) ', tp, &
     '   A/P ', ta / tp, '   max|yA-yP|/max|yA| ', MAXVAL(ABS(y1 - y2)) / MAXVAL(ABS(y1))
  ! ---- FP32 A
  DEALLOCATE(box, t)
  ALLOCATE(bs(nnr, nb), xs(npw, nb), ys(npw, nb), vs(nnr))
  xs = CMPLX(hx, KIND=sp); vs = REAL(hv, sp)
  istat = cufftPlanMany(sA, 3, n, n, 1, nnr, n, 1, nnr, CUFFT_C2C, nb)
  DO rep = 0, nrep
    IF (rep == 1) istat = cudaEventRecord(e0, 0)
    CALL s_scat<<<dim3((nnr + 255) / 256, nb, 1), 256>>>(bs, xs, d_inv, nnr, npw)
    istat = cufftExecC2C(sA, bs, bs, CUFFT_INVERSE)
    CALL s_vmul<<<dim3((nnr + 255) / 256, nb, 1), 256>>>(bs, vs, nnr)
    istat = cufftExecC2C(sA, bs, bs, CUFFT_FORWARD)
    CALL s_gath<<<dim3((npw + 255) / 256, nb, 1), 256>>>(ys, xs, bs, d_idx, nnr, npw)
  ENDDO
  istat = cudaEventRecord(e1, 0); istat = cudaEventSynchronize(e1); istat = cudaEventElapsedTime(ms, e0, e1)
  ts = 1.0e3_dp * ms / (nrep * nb)
  WRITE(*, '(a,f8.2,a,f8.2,a)') '  FP32 us/column: A ', ts, '   (box traffic floor at 696 GB/s, 15 B FP64: ', &
     15.0_dp * 16 * nnr / 696.0e3_dp, ' us)'
END PROGRAM fftbench
