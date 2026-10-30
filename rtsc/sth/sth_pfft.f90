!
! rtsc: pruned, fused local-potential operator for the batched Sternheimer engine
! (groups/compute/KERNEL-LOG.md, lever 1). RTSC_STH_PFFT=1.
!
! For a column x (coefficients on the k+q sphere) the engine needs
!   y = g2kin x + (1/N) FFT_fwd[ V(r) * FFT_inv[x] ](G on the sphere) - e x
! (k_scatter_full + cuFFT 3D inverse + k_vmul + cuFFT 3D forward + k_gather_h: 9 launches and ~15 box
! sweeps of global memory per column). Here the same transform is done separably and pruned:
!   K1 (k_pz_inv)  per group of 32 z-sticks: load the sphere coefficients of the sticks (0 elsewhere),
!                  inverse FFT along z in shared memory, write the sticks S(stick, z) (~18 % of a box);
!   K2 (k_pxy)     per z-plane: place the plane's stick values in a shared (n1 x n2) plane, inverse FFT
!                  along y on the occupied x-columns only, inverse FFT along x on every row, multiply by
!                  V(r), forward FFT along x, forward FFT along y on the occupied x-columns, write the
!                  stick values back to S;
!   K3 (k_pz_fwd)  per group of sticks: forward FFT along z and the k_gather_h formula on the sphere.
! Global traffic per column is ~4 stick arrays (~0.7 box) instead of ~15 boxes; V (one plane per block)
! stays in L2. The 1D transforms are mixed-radix (2, 3, 4, 5) Stockham FFTs, one warp per line, in place
! in shared memory (each lane holds at most two butterflies in registers across the warp barrier).
! Same mathematics as cuFFT (exact DFT); different floating-point order (class E).
!
! Supported: nr1x = nr1, nr2x = nr2 (serial FFT box), every dimension <= 128 with factors 2, 3, 5 only,
! and the (n1 + pad) x n2 plane in shared memory. Otherwise pfft_ok() is false and the engine keeps cuFFT.
!
MODULE sth_pfft
  USE cudafor
  IMPLICIT NONE
  PRIVATE
  INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(14, 200)
  INTEGER, PARAMETER :: GS = 32          ! sticks per z-pass block
  INTEGER, PARAMETER :: NTH = 256        ! threads per block (8 warps)
  INTEGER, PARAMETER :: MAXF = 10
  !
  PUBLIC :: pfft_setup, pfft_free, pfft_local, pfft_ok, pfft_info
  !
  LOGICAL, SAVE :: ready = .FALSE.
  INTEGER, SAVE :: n1, n2, n3, ld, nst, nxa, nzst, ngrp, nkq, pchunk, shm_xy, shm_z
  INTEGER, SAVE :: h_nf(3), h_fac(MAXF, 3), lb(3)
  INTEGER, DEVICE, ALLOCATABLE :: d_fac(:,:), d_sx(:), d_sy(:), d_xa(:), d_zmap(:,:), d_tab(:,:,:)
  COMPLEX(dp), DEVICE, ALLOCATABLE :: d_tw1(:), d_tw2(:), d_tw3(:), d_tw1m(:), d_tw2m(:), d_S(:,:)
  !
CONTAINS
  !
  LOGICAL FUNCTION pfft_ok()
    pfft_ok = ready
  END FUNCTION pfft_ok
  !
  SUBROUTINE pfft_info(nst_, nxa_, pchunk_)
    INTEGER, INTENT(OUT) :: nst_, nxa_, pchunk_
    nst_ = nst; nxa_ = nxa; pchunk_ = pchunk
  END SUBROUTINE pfft_info
  !
  !-------------------------------------------------------------------------------------------------
  ! device FFT building blocks
  !-------------------------------------------------------------------------------------------------
  !
  ATTRIBUTES(DEVICE) SUBROUTINE bfly(P, base, es, R, m, j, ke, tw, sgn, a0, a1, a2, a3, a4)
    ! Stockham butterfly j of one stage (radix R, m = n/R): loads P(base + es*(j + r*m)), applies the
    ! twiddles tw(r*ke) = exp(sgn 2 pi i r (j mod ns) / (ns R)) (tw is the table of sign sgn) and the
    ! radix-R DFT with sign sgn. Destinations are precomputed by the caller (stage tables).
    COMPLEX(dp) :: P(0:*), tw(0:*)
    INTEGER, VALUE :: base, es, R, m, j, ke, sgn
    COMPLEX(dp) :: a0, a1, a2, a3, a4
    COMPLEX(dp) :: t1, t2, t3, t4, b1, b2, c1, c2
    REAL(dp), PARAMETER :: s3 = 0.866025403784438646763723170752936183_dp
    REAL(dp), PARAMETER :: c51 = 0.309016994374947424102293417182819059_dp, &
                           c52 = -0.809016994374947424102293417182819059_dp, &
                           s51 = 0.951056516295153572116439333379382143_dp, &
                           s52 = 0.587785252292473129168705954639072769_dp
    REAL(dp) :: sg
    sg = REAL(sgn, dp)
    a0 = P(base + es * j)
    a1 = P(base + es * (j + m))
    IF (ke > 0) a1 = a1 * tw(ke)
    IF (R >= 3) THEN
      a2 = P(base + es * (j + 2 * m))
      IF (ke > 0) a2 = a2 * tw(2 * ke)
    ENDIF
    IF (R >= 4) THEN
      a3 = P(base + es * (j + 3 * m))
      IF (ke > 0) a3 = a3 * tw(3 * ke)
    ENDIF
    IF (R == 5) THEN
      a4 = P(base + es * (j + 4 * m))
      IF (ke > 0) a4 = a4 * tw(4 * ke)
    ENDIF
    SELECT CASE (R)
    CASE (2)
      t1 = a0 + a1
      a1 = a0 - a1
      a0 = t1
    CASE (3)
      t1 = a1 + a2
      t2 = a0 - 0.5_dp * t1
      t3 = (a1 - a2) * (sg * s3)
      t3 = CMPLX(-AIMAG(t3), REAL(t3), dp)          ! i * sg * s3 * (a1 - a2)
      a0 = a0 + t1
      a1 = t2 + t3
      a2 = t2 - t3
    CASE (4)
      t1 = a0 + a2
      t2 = a0 - a2
      t3 = a1 + a3
      t4 = (a1 - a3) * sg
      t4 = CMPLX(-AIMAG(t4), REAL(t4), dp)          ! (sg i) (a1 - a3)
      a0 = t1 + t3
      a2 = t1 - t3
      a1 = t2 + t4
      a3 = t2 - t4
    CASE (5)
      t1 = a1 + a4
      t2 = a2 + a3
      t3 = a1 - a4
      t4 = a2 - a3
      b1 = a0 + c51 * t1 + c52 * t2
      b2 = a0 + c52 * t1 + c51 * t2
      c1 = sg * (s51 * t3 + s52 * t4)
      c1 = CMPLX(-AIMAG(c1), REAL(c1), dp)
      c2 = sg * (s52 * t3 - s51 * t4)
      c2 = CMPLX(-AIMAG(c2), REAL(c2), dp)
      a0 = a0 + t1 + t2
      a1 = b1 + c1
      a4 = b1 - c1
      a2 = b2 + c2
      a3 = b2 - c2
    END SELECT
  END SUBROUTINE bfly
  !
  ATTRIBUTES(DEVICE) SUBROUTINE bstore(P, base, es, R, ns, d, a0, a1, a2, a3, a4)
    COMPLEX(dp) :: P(0:*)
    INTEGER, VALUE :: base, es, R, ns, d
    COMPLEX(dp) :: a0, a1, a2, a3, a4
    P(base + es * d) = a0
    P(base + es * (d + ns)) = a1
    IF (R >= 3) P(base + es * (d + 2 * ns)) = a2
    IF (R >= 4) P(base + es * (d + 3 * ns)) = a3
    IF (R == 5) P(base + es * (d + 4 * ns)) = a4
  END SUBROUTINE bstore
  !
  ATTRIBUTES(DEVICE) SUBROUTINE wfft(P, b0, bs, xa, usexa, l0, nlb, es, n, nf, fac, tab, tw, sgn, lane)
    ! in-place FFTs (sign sgn, unnormalised) of the nlb lines l = l0..l0+nlb-1 by one warp; line l is
    ! P(base_l + es*i), i = 0..n-1, base_l = xa(l+1) if usexa, else b0 + bs*l. The (line, butterfly)
    ! items of a stage are spread over the 32 lanes, at most two per lane (nlb * n/R <= 64).
    ! tab(t+1, st) packs, for item t of stage st: line (6 bits), butterfly j (7), destination (7) and
    ! twiddle exponent (7); tw is the twiddle table of sign sgn.
    COMPLEX(dp) :: P(0:*), tw(0:*)
    INTEGER, VALUE :: b0, bs, usexa, l0, nlb, es, n, nf, sgn, lane
    INTEGER :: fac(*), xa(*), tab(64, *)
    INTEGER :: st, R, m, ns, dA, dB, tot, l, jA, jB, kA, kB, baseA, baseB, e
    LOGICAL :: okA, okB
    COMPLEX(dp) :: a0, a1, a2, a3, a4, b0v, b1, b2, b3, b4
    ns = 1
    DO st = 1, nf
      R = fac(st)
      m = n / R
      tot = nlb * m
      okA = lane < tot
      okB = lane + 32 < tot
      IF (okA) THEN
        e = tab(lane + 1, st)
        l = IAND(e, 63); jA = IAND(ISHFT(e, -6), 127); dA = IAND(ISHFT(e, -13), 127); kA = ISHFT(e, -20)
        IF (usexa /= 0) THEN
          baseA = xa(l0 + l + 1)
        ELSE
          baseA = b0 + bs * (l0 + l)
        ENDIF
        CALL bfly(P, baseA, es, R, m, jA, kA, tw, sgn, a0, a1, a2, a3, a4)
      ENDIF
      IF (okB) THEN
        e = tab(lane + 33, st)
        l = IAND(e, 63); jB = IAND(ISHFT(e, -6), 127); dB = IAND(ISHFT(e, -13), 127); kB = ISHFT(e, -20)
        IF (usexa /= 0) THEN
          baseB = xa(l0 + l + 1)
        ELSE
          baseB = b0 + bs * (l0 + l)
        ENDIF
        CALL bfly(P, baseB, es, R, m, jB, kB, tw, sgn, b0v, b1, b2, b3, b4)
      ENDIF
      CALL syncwarp(-1)
      IF (okA) CALL bstore(P, baseA, es, R, ns, dA, a0, a1, a2, a3, a4)
      IF (okB) CALL bstore(P, baseB, es, R, ns, dB, b0v, b1, b2, b3, b4)
      CALL syncwarp(-1)
      ns = ns * R
    ENDDO
  END SUBROUTINE wfft
  !
  !-------------------------------------------------------------------------------------------------
  ! kernels
  !-------------------------------------------------------------------------------------------------
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_pz_inv(S, x, list, s0, nslot, npert, npwx, nst, n3, zmap, nzst, &
                                         fac, nf3, tw3, lbz, tab)
    ! S(stick + nst*z, j) = inverse z-FFT of the column list(s0+j) on the sticks of group blockIdx%x
    INTEGER, VALUE :: s0, nslot, npert, npwx, nst, n3, nzst, nf3, lbz
    COMPLEX(dp) :: S(nzst, *), x(npwx, *), tw3(0:*)
    INTEGER :: list(*), zmap(nzst, *), fac(MAXF, 3), tab(64, MAXF, 3)
    COMPLEX(dp), SHARED :: sm(0:*)
    INTEGER :: j, col, k, g, tid, lane, w, nw, p, sl, i3, sg, ig, l, lz, tw0, nlz
    COMPLEX(dp) :: v
    j = blockIdx%y
    col = list(s0 + j)
    k = (col - 1) / (nslot * npert) + 1
    g = blockIdx%x - 1
    tid = threadIdx%x - 1; lane = IAND(tid, 31); w = tid / 32; nw = blockDim%x / 32
    lz = (GS + 1) * n3
    tw0 = lz
    DO p = tid, n3 - 1, blockDim%x
      sm(tw0 + p) = tw3(p)
    ENDDO
    DO p = tid, GS * n3 - 1, blockDim%x
      sl = MOD(p, GS); i3 = p / GS; sg = g * GS + sl
      v = (0.0_dp, 0.0_dp)
      IF (sg < nst) THEN
        ig = zmap(sg + 1 + nst * i3, k)
        IF (ig > 0) v = x(ig, col)
      ENDIF
      sm(sl + (GS + 1) * i3) = v
    ENDDO
    CALL syncthreads()
    nlz = MIN(GS, nst - g * GS)
    DO l = w * lbz, nlz - 1, nw * lbz
      CALL wfft(sm, 0, 1, fac, 0, l, MIN(lbz, nlz - l), GS + 1, n3, nf3, fac(1, 3), tab(1, 1, 3), sm(tw0), 1, lane)
    ENDDO
    CALL syncthreads()
    DO p = tid, GS * n3 - 1, blockDim%x
      sl = MOD(p, GS); i3 = p / GS; sg = g * GS + sl
      IF (sg < nst) S(sg + 1 + nst * i3, j) = sm(sl + (GS + 1) * i3)
    ENDDO
  END SUBROUTINE k_pz_inv
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_pxy(S, v, nst, n1, n2, n3, ld, nzst, sx, sy, xa, nxa, fac, nf1, nf2, tw1, tw2, &
                                      tw1m, tw2m, lbx, lby, tab)
    ! one z-plane (blockIdx%x) of column j: y-inverse on the occupied x-columns, x-inverse, *V,
    ! x-forward, y-forward on the occupied x-columns; reads and writes only the stick values
    INTEGER, VALUE :: nst, n1, n2, n3, ld, nzst, nxa, nf1, nf2, lbx, lby
    COMPLEX(dp) :: S(nzst, *), tw1(0:*), tw2(0:*), tw1m(0:*), tw2m(0:*)
    REAL(dp) :: v(*)
    INTEGER :: sx(*), sy(*), xa(*), fac(MAXF, 3), tab(64, MAXF, 3)
    COMPLEX(dp), SHARED :: sm(0:*)
    INTEGER :: j, i3, tid, lane, w, nw, p, l, i1, i2, np
    j = blockIdx%y
    i3 = blockIdx%x - 1
    tid = threadIdx%x - 1; lane = IAND(tid, 31); w = tid / 32; nw = blockDim%x / 32
    np = ld * n2
    DO p = tid, np - 1, blockDim%x
      sm(p) = (0.0_dp, 0.0_dp)
    ENDDO
    CALL syncthreads()
    DO p = tid, nst - 1, blockDim%x
      sm(sx(p + 1) + ld * sy(p + 1)) = S(p + 1 + nst * i3, j)
    ENDDO
    CALL syncthreads()
    DO l = w * lby, nxa - 1, nw * lby
      CALL wfft(sm, 0, 0, xa, 1, l, MIN(lby, nxa - l), ld, n2, nf2, fac(1, 2), tab(1, 1, 2), tw2, 1, lane)
    ENDDO
    CALL syncthreads()
    DO l = w * lbx, n2 - 1, nw * lbx
      CALL wfft(sm, 0, ld, xa, 0, l, MIN(lbx, n2 - l), 1, n1, nf1, fac(1, 1), tab(1, 1, 1), tw1, 1, lane)
    ENDDO
    CALL syncthreads()
    DO p = tid, n1 * n2 - 1, blockDim%x
      i1 = MOD(p, n1); i2 = p / n1
      sm(i1 + ld * i2) = sm(i1 + ld * i2) * v(p + 1 + n1 * n2 * i3)
    ENDDO
    CALL syncthreads()
    DO l = w * lbx, n2 - 1, nw * lbx
      CALL wfft(sm, 0, ld, xa, 0, l, MIN(lbx, n2 - l), 1, n1, nf1, fac(1, 1), tab(1, 1, 1), tw1m, -1, lane)
    ENDDO
    CALL syncthreads()
    DO l = w * lby, nxa - 1, nw * lby
      CALL wfft(sm, 0, 0, xa, 1, l, MIN(lby, nxa - l), ld, n2, nf2, fac(1, 2), tab(1, 1, 2), tw2m, -1, lane)
    ENDDO
    CALL syncthreads()
    DO p = tid, nst - 1, blockDim%x
      S(p + 1 + nst * i3, j) = sm(sx(p + 1) + ld * sy(p + 1))
    ENDDO
  END SUBROUTINE k_pxy
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_pz_fwd(y, x, S, list, s0, nslot, npert, npwx, nksq, nbnd, nst, n3, &
                                         zmap, nzst, fac, nf3, tw3, npwq, g2, etk, scale, lbz, tab)
    ! forward z-FFT of the sticks of group blockIdx%x, then (as k_gather_h, opvar 0)
    !   y(G) = g2kin(G) x(G) + scale * Vx(G) - e_n x(G) on the sphere; rows > npwq zeroed (group 0)
    INTEGER, VALUE :: s0, nslot, npert, npwx, nksq, nbnd, nst, n3, nzst, nf3, lbz
    REAL(dp), VALUE :: scale
    COMPLEX(dp) :: y(npwx, *), x(npwx, *), S(nzst, *), tw3(0:*)
    INTEGER :: list(*), zmap(nzst, *), fac(MAXF, 3), npwq(nksq), tab(64, MAXF, 3)
    REAL(dp) :: g2(npwx, nksq), etk(nbnd, nksq)
    COMPLEX(dp), SHARED :: sm(0:*)
    INTEGER :: j, col, k, n, g, tid, lane, w, nw, p, sl, i3, sg, ig, l, lz, tw0, nlz
    COMPLEX(dp) :: xv
    j = blockIdx%y
    col = list(s0 + j)
    k = (col - 1) / (nslot * npert) + 1
    n = MOD(col - 1, nslot) + 1
    g = blockIdx%x - 1
    tid = threadIdx%x - 1; lane = IAND(tid, 31); w = tid / 32; nw = blockDim%x / 32
    lz = (GS + 1) * n3
    tw0 = lz
    DO p = tid, n3 - 1, blockDim%x
      sm(tw0 + p) = CONJG(tw3(p))
    ENDDO
    DO p = tid, GS * n3 - 1, blockDim%x
      sl = MOD(p, GS); i3 = p / GS; sg = g * GS + sl
      IF (sg < nst) THEN
        sm(sl + (GS + 1) * i3) = S(sg + 1 + nst * i3, j)
      ELSE
        sm(sl + (GS + 1) * i3) = (0.0_dp, 0.0_dp)
      ENDIF
    ENDDO
    CALL syncthreads()
    nlz = MIN(GS, nst - g * GS)
    DO l = w * lbz, nlz - 1, nw * lbz
      CALL wfft(sm, 0, 1, fac, 0, l, MIN(lbz, nlz - l), GS + 1, n3, nf3, fac(1, 3), tab(1, 1, 3), sm(tw0), -1, lane)
    ENDDO
    CALL syncthreads()
    DO p = tid, GS * n3 - 1, blockDim%x
      sl = MOD(p, GS); i3 = p / GS; sg = g * GS + sl
      IF (sg < nst) THEN
        ig = zmap(sg + 1 + nst * i3, k)
        IF (ig > 0) THEN
          xv = x(ig, col)
          y(ig, col) = g2(ig, k) * xv + sm(sl + (GS + 1) * i3) * scale - etk(n, k) * xv
        ENDIF
      ENDIF
    ENDDO
    IF (g == 0) THEN
      DO ig = npwq(k) + 1 + tid, npwx, blockDim%x
        y(ig, col) = (0.0_dp, 0.0_dp)
      ENDDO
    ENDIF
  END SUBROUTINE k_pz_fwd
  !
  !-------------------------------------------------------------------------------------------------
  ! host side
  !-------------------------------------------------------------------------------------------------
  !
  SUBROUTINE factor(n, nf, f, okf)
    ! n = product of f(1:nf) with radices 4, 2, 3, 5 (4 first); okf false for other prime factors
    INTEGER, INTENT(IN) :: n
    INTEGER, INTENT(OUT) :: nf, f(MAXF)
    LOGICAL, INTENT(OUT) :: okf
    INTEGER :: r
    r = n; nf = 0; f = 1
    DO WHILE (MOD(r, 4) == 0 .AND. nf < MAXF)
      nf = nf + 1; f(nf) = 4; r = r / 4
    ENDDO
    DO WHILE (MOD(r, 2) == 0 .AND. nf < MAXF)
      nf = nf + 1; f(nf) = 2; r = r / 2
    ENDDO
    DO WHILE (MOD(r, 3) == 0 .AND. nf < MAXF)
      nf = nf + 1; f(nf) = 3; r = r / 3
    ENDDO
    DO WHILE (MOD(r, 5) == 0 .AND. nf < MAXF)
      nf = nf + 1; f(nf) = 5; r = r / 5
    ENDDO
    okf = (r == 1) .AND. n <= 128
  END SUBROUTINE factor
  !
  SUBROUTINE pfft_setup(nr, nrx, nksq, npwx, npwq, idxkq, chunk, ierr)
    !! Build the stick geometry for the k+q spheres of one q (union over k), the per-k map
    !! (stick, z) -> plane-wave index, twiddles and the stick work array for chunk columns.
    !! ierr /= 0: unsupported (the caller keeps cuFFT).
    INTEGER, INTENT(IN) :: nr(3), nrx(3), nksq, npwx, npwq(nksq), idxkq(npwx, nksq), chunk
    INTEGER, INTENT(OUT) :: ierr
    INTEGER, ALLOCATABLE :: smap(:), zm(:,:), sx(:), sy(:), xa(:)
    LOGICAL, ALLOCATABLE :: xon(:)
    COMPLEX(dp), ALLOCATABLE :: tw(:)
    LOGICAL :: o1, o2, o3
    INTEGER :: ik, ig, l, i1, i2, i3, s, istat, d, nmax
    REAL(dp), PARAMETER :: tpi = 6.283185307179586476925286766559005768_dp
    INTEGER(8) :: freeb, totb
    CALL pfft_free()
    ierr = 0
    n1 = nr(1); n2 = nr(2); n3 = nr(3)
    IF (nrx(1) /= n1 .OR. nrx(2) /= n2) THEN
      ierr = 1; RETURN
    ENDIF
    CALL factor(n1, h_nf(1), h_fac(:, 1), o1)
    CALL factor(n2, h_nf(2), h_fac(:, 2), o2)
    CALL factor(n3, h_nf(3), h_fac(:, 3), o3)
    IF (.NOT. (o1 .AND. o2 .AND. o3)) THEN
      ierr = 2; RETURN
    ENDIF
    ld = n1
    IF (MOD(n1, 2) == 0) ld = n1 + 1
    ! lines per warp batch: (lines x butterflies) of the widest stage (smallest radix) <= 64
    DO d = 1, 3
      lb(d) = MAX(1, 64 / (nr(d) / MINVAL(h_fac(1:h_nf(d), d))))
    ENDDO
    shm_xy = 16 * ld * n2
    shm_z = 16 * ((GS + 1) * n3 + n3)
    IF (shm_xy > 99 * 1024 .OR. shm_z > 48 * 1024) THEN
      ierr = 3; RETURN
    ENDIF
    ALLOCATE(smap(n1 * n2), xon(n1))
    smap = 0; xon = .FALSE.; nst = 0
    DO ik = 1, nksq
      DO ig = 1, npwq(ik)
        l = idxkq(ig, ik) - 1
        i1 = MOD(l, nrx(1)); i2 = MOD(l / nrx(1), nrx(2))
        IF (smap(i1 + 1 + n1 * i2) == 0) THEN
          nst = nst + 1; smap(i1 + 1 + n1 * i2) = nst
        ENDIF
      ENDDO
    ENDDO
    ALLOCATE(sx(nst), sy(nst))
    DO i2 = 0, n2 - 1
      DO i1 = 0, n1 - 1
        s = smap(i1 + 1 + n1 * i2)
        IF (s > 0) THEN
          sx(s) = i1; sy(s) = i2; xon(i1 + 1) = .TRUE.
        ENDIF
      ENDDO
    ENDDO
    nxa = COUNT(xon)
    ALLOCATE(xa(nxa))
    s = 0
    DO i1 = 1, n1
      IF (xon(i1)) THEN
        s = s + 1; xa(s) = i1 - 1
      ENDIF
    ENDDO
    nzst = nst * n3
    ngrp = (nst + GS - 1) / GS
    nkq = nksq
    ALLOCATE(zm(nzst, nksq))
    zm = 0
    DO ik = 1, nksq
      DO ig = 1, npwq(ik)
        l = idxkq(ig, ik) - 1
        i1 = MOD(l, nrx(1)); i2 = MOD(l / nrx(1), nrx(2)); i3 = l / (nrx(1) * nrx(2))
        zm(smap(i1 + 1 + n1 * i2) + nst * i3, ik) = ig
      ENDDO
    ENDDO
    nmax = n1 + n2 + n3
    ALLOCATE(tw(0:nmax - 1))
    DO d = 0, n1 - 1
      tw(d) = CMPLX(COS(tpi * d / n1), SIN(tpi * d / n1), dp)
    ENDDO
    DO d = 0, n2 - 1
      tw(n1 + d) = CMPLX(COS(tpi * d / n2), SIN(tpi * d / n2), dp)
    ENDDO
    DO d = 0, n3 - 1
      tw(n1 + n2 + d) = CMPLX(COS(tpi * d / n3), SIN(tpi * d / n3), dp)
    ENDDO
    ! work array: as many columns as fit in 1/8 of the free memory, at most chunk
    istat = cudaMemGetInfo(freeb, totb)
    pchunk = INT(MIN(INT(chunk, 8), MAX(1_8, (freeb / 8) / (16_8 * nzst))))
    ALLOCATE(d_fac(MAXF, 3), d_sx(nst), d_sy(nst), d_xa(nxa), d_zmap(nzst, nksq))
    ALLOCATE(d_tw1(0:n1 - 1), d_tw2(0:n2 - 1), d_tw3(0:n3 - 1), d_tw1m(0:n1 - 1), d_tw2m(0:n2 - 1))
    ALLOCATE(d_S(nzst, pchunk))
    d_fac = h_fac; d_sx = sx; d_sy = sy; d_xa = xa; d_zmap = zm
    ! stage tables: item t of stage st -> line, butterfly, destination, twiddle exponent (packed)
    BLOCK
      INTEGER :: htab(64, MAXF, 3), st, ns, R, m, step, t, ll, jj, kk, dd
      htab = 0
      DO d = 1, 3
        ns = 1
        DO st = 1, h_nf(d)
          R = h_fac(st, d); m = nr(d) / R; step = nr(d) / (ns * R)
          DO t = 0, 63
            ll = t / m; jj = MOD(t, m); kk = MOD(jj, ns); dd = (jj / ns) * ns * R + kk
            htab(t + 1, st, d) = IOR(IOR(IOR(MIN(ll, 63), ISHFT(jj, 6)), ISHFT(dd, 13)), ISHFT(kk * step, 20))
          ENDDO
          ns = ns * R
        ENDDO
      ENDDO
      ALLOCATE(d_tab(64, MAXF, 3))
      d_tab = htab
    END BLOCK
    d_tw1 = tw(0:n1 - 1); d_tw2 = tw(n1:n1 + n2 - 1); d_tw3 = tw(n1 + n2:nmax - 1)
    d_tw1m = CONJG(tw(0:n1 - 1)); d_tw2m = CONJG(tw(n1:n1 + n2 - 1))
    DEALLOCATE(smap, xon, sx, sy, xa, zm, tw)
    istat = cudaFuncSetAttribute(k_pxy, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_xy)
    IF (istat /= cudaSuccess) THEN
      CALL pfft_free()
      ierr = 4; RETURN
    ENDIF
    ready = .TRUE.
  END SUBROUTINE pfft_setup
  !
  SUBROUTINE pfft_free()
    IF (ALLOCATED(d_fac)) DEALLOCATE(d_fac, d_sx, d_sy, d_xa, d_zmap, d_tw1, d_tw2, d_tw3, d_tab, d_tw1m, d_tw2m)
    IF (ALLOCATED(d_S)) DEALLOCATE(d_S)
    ready = .FALSE.
  END SUBROUTINE pfft_free
  !
  SUBROUTINE pfft_local(x, y, list, nl, nslot, npert, npwx, nksq, nbnd, vrs, npwq, g2, etk, scale, strm)
    !! y(:, c) = g2kin x + scale * FFT_fwd[V FFT_inv[x]] - e x on the sphere, rows > npwq zeroed,
    !! for the nl columns c = list(1:nl) (the local part of apply_op)
    COMPLEX(dp), DEVICE :: x(npwx, *), y(npwx, *)
    INTEGER, DEVICE :: list(*), npwq(nksq)
    INTEGER, INTENT(IN) :: nl, nslot, npert, npwx, nksq, nbnd
    REAL(dp), DEVICE :: vrs(*), g2(npwx, nksq), etk(nbnd, nksq)
    REAL(dp), INTENT(IN) :: scale
    INTEGER(KIND=cuda_stream_kind), INTENT(IN) :: strm
    INTEGER :: s0, nb, istat
    s0 = 0
    DO WHILE (s0 < nl)
      nb = MIN(pchunk, nl - s0)
      CALL k_pz_inv<<<dim3(ngrp, nb, 1), NTH, shm_z, strm>>>(d_S, x, list, s0, nslot, npert, npwx, nst, n3, &
           d_zmap, nzst, d_fac, h_nf(3), d_tw3, lb(3), d_tab)
      CALL k_pxy<<<dim3(n3, nb, 1), NTH, shm_xy, strm>>>(d_S, vrs, nst, n1, n2, n3, ld, nzst, d_sx, d_sy, d_xa, &
           nxa, d_fac, h_nf(1), h_nf(2), d_tw1, d_tw2, d_tw1m, d_tw2m, lb(1), lb(2), d_tab)
      CALL k_pz_fwd<<<dim3(ngrp, nb, 1), NTH, shm_z, strm>>>(y, x, d_S, list, s0, nslot, npert, npwx, nksq, &
           nbnd, nst, n3, d_zmap, nzst, d_fac, h_nf(3), d_tw3, npwq, g2, etk, scale, lb(3), d_tab)
      istat = cudaGetLastError()
      IF (istat /= cudaSuccess) THEN
        WRITE(*, '(A,A)') 'sth_pfft: launch failed: ', TRIM(cudaGetErrorString(istat))
        STOP 1
      ENDIF
      s0 = s0 + nb
    ENDDO
  END SUBROUTINE pfft_local
  !
END MODULE sth_pfft
