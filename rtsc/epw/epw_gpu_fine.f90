!
! rtsc: fused fine-grid electron-phonon interpolation for ephwrite (Eliashberg) runs.
!
! Vanilla, per fine q: epmatwef(Re) = sum_Rp e^{iq.Rp} epmatwp(Re, Rp) (+ mp_sum over the pools'
! Rp slices), / ndegen_k(Re), then per in-window pair g = U(k+q) [sum_Re e^{ik.Re} epmatwef(Re)]
! U(k)^H, rotated to phonon modes, degeneracy-averaged and squared in write_ephmat.
! Here the two Fourier sums are reassociated "k first":
!   H(k; Rp)  = sum_Re e^{ik.Re} / ndegen_k(Re) epmatwp(Re, Rp)   once, for the pool's Fermi-shell k
!   X(k, q)   = sum_Rp e^{iq.Rp} H(k; Rp)                          one batched ZGEMM per q batch
! and the eigenpairs at k (q-independent) and on the full fine grid (for k+q) are computed once.
! The per-pair kernel does U(k+q) X U(k)^H on the bands ibndmin..ibndmax, the mode rotation, the
! two degeneracy averages of write_ephmat and writes g^2 in write_ephmat's order.
! Same algebra, different FP association (gate class E).
!
MODULE epw_gpu_fine
  USE cudafor
  USE cublas_v2
  IMPLICIT NONE
  PRIVATE
  INTEGER, PARAMETER :: dp = KIND(1.d0)
  REAL(dp), PARAMETER :: eps6 = 1.0E-6_dp
  !
  INTEGER :: nb = 0, nm = 0, nbf = 0, ibmin = 0, nrg = 0, nkw = 0, ngrid = 0, nkfs_ = 0
  COMPLEX(dp), DEVICE, ALLOCATABLE :: d_h(:), d_x(:), d_cq(:), d_cufk(:), d_cufg(:), d_uf(:)
  REAL(dp), DEVICE, ALLOCATABLE :: d_ek(:), d_eg(:), d_wf(:), d_out(:)
  INTEGER, DEVICE, ALLOCATABLE :: d_win(:), d_pair(:)
  INTEGER :: nqb_alloc = -1, npair_alloc = -1, nout_alloc = -1
  LOGICAL :: have_h = .FALSE.
  TYPE(cublasHandle) :: hb
  LOGICAL :: have_hb = .FALSE.
  !
  PUBLIC :: ef_shmem, ef_setup, ef_hpart, ef_set_h, ef_batch, ef_free
  !
CONTAINS
  !
  SUBROUTINE chk(istat, what)
    INTEGER, INTENT(IN) :: istat
    CHARACTER(*), INTENT(IN) :: what
    IF (istat /= 0) THEN
      WRITE(*, '(a, a, a, i0)') 'epw_gpu_fine: ', what, ' failed, status ', istat
      STOP 1
    ENDIF
  END SUBROUTINE chk
  !
  INTEGER FUNCTION ef_shmem(nb_, nm_, nbf_)
    !! bytes of dynamic shared memory of the pair kernel
    INTEGER, INTENT(IN) :: nb_, nm_, nbf_
    ef_shmem = 16 * (3 * nbf_ * nb_ + 2 * nbf_ * nbf_ * nm_) + 8 * nbf_ * nbf_ * nm_
  END FUNCTION ef_shmem
  !
  SUBROUTINE ef_free()
    IF (ALLOCATED(d_h)) DEALLOCATE(d_h)
    IF (ALLOCATED(d_x)) DEALLOCATE(d_x)
    IF (ALLOCATED(d_cq)) DEALLOCATE(d_cq)
    IF (ALLOCATED(d_cufk)) DEALLOCATE(d_cufk, d_ek)
    IF (ALLOCATED(d_cufg)) DEALLOCATE(d_cufg, d_eg)
    IF (ALLOCATED(d_uf)) DEALLOCATE(d_uf, d_wf)
    IF (ALLOCATED(d_out)) DEALLOCATE(d_out)
    IF (ALLOCATED(d_win)) DEALLOCATE(d_win)
    IF (ALLOCATED(d_pair)) DEALLOCATE(d_pair)
    nqb_alloc = -1; npair_alloc = -1; nout_alloc = -1
    have_h = .FALSE.
  END SUBROUTINE ef_free
  !
  SUBROUTINE ef_setup(nb_, nm_, nbf_, ibmin_, nrg_, nkw_, ngrid_, nkfs_in, cufk, ek, cufg, eg, win)
    !! Upload the eigenpairs at the pool's Fermi-shell k (cufk, ek: nkw_ points), on the full
    !! fine grid (cufg, eg: ngrid_ points) and the Fermi-window mask win(nbf, nkfs) of ekfs.
    INTEGER, INTENT(IN) :: nb_, nm_, nbf_, ibmin_, nrg_, nkw_, ngrid_, nkfs_in
    COMPLEX(dp), INTENT(IN) :: cufk(nb_ * nb_ * MAX(nkw_, 1)), cufg(nb_ * nb_ * ngrid_)
    REAL(dp), INTENT(IN) :: ek(nb_ * MAX(nkw_, 1)), eg(nb_ * ngrid_)
    INTEGER, INTENT(IN) :: win(nbf_ * nkfs_in)
    CALL ef_free()
    nb = nb_; nm = nm_; nbf = nbf_; ibmin = ibmin_; nrg = nrg_; nkw = nkw_; ngrid = ngrid_
    nkfs_ = nkfs_in
    ALLOCATE(d_cufk(nb * nb * MAX(nkw, 1)), d_ek(nb * MAX(nkw, 1)), d_cufg(nb * nb * ngrid), &
             d_eg(nb * ngrid), d_win(nbf * nkfs_in))
    d_cufk = cufk; d_ek = ek; d_cufg = cufg; d_eg = eg; d_win = win
    IF (.NOT. have_hb) THEN
      CALL chk(cublasCreate(hb), 'cublasCreate')
      have_hb = .TRUE.
    ENDIF
  END SUBROUTINE ef_setup
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_scale(nb2, nrk, nbat, use_ws, ndeg, ep, s)
    !! s(mn, Re, b) = ep(mn, Re, b) / ndegen_k(Re, m, n)  (0 where ndegen_k = 0), as
    !! wigner_divide_ndegen; ndeg(nrk, nb, nb) when use_ws, else ndeg(nrk) is used.
    INTEGER, VALUE :: nb2, nrk, nbat, use_ws
    INTEGER, DEVICE :: ndeg(nrk, *)
    COMPLEX(dp), DEVICE :: ep(nb2, nrk, nbat), s(nb2, nrk, nbat)
    INTEGER :: mn, ir, b, d
    mn = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    ir = blockIdx%y
    b = blockIdx%z
    IF (mn > nb2) RETURN
    IF (use_ws == 1) THEN
      d = ndeg(ir, mn)
    ELSE
      d = ndeg(ir, 1)
    ENDIF
    IF (d > 0) THEN
      s(mn, ir, b) = ep(mn, ir, b) / REAL(d, dp)
    ELSE
      s(mn, ir, b) = (0.0_dp, 0.0_dp)
    ENDIF
  END SUBROUTINE k_scale
  !
  SUBROUTINE ef_hpart(nb_, nrk, nm_, nirg, epmatwp, use_ws, ndegk, ndims, nkwg, cfk, hp)
    !! hp(mn, nu + nm (rpl - 1), kw) = sum_Re cfk(Re, kw) epmatwp(mn, Re, nu, rpl) / ndegen_k
    !! for this pool's Rp slice (nirg) and all nkwg Fermi-shell k of all pools.
    INTEGER, INTENT(IN) :: nb_, nrk, nm_, nirg, ndims, nkwg
    COMPLEX(dp), DEVICE, INTENT(IN) :: epmatwp(nb_ * nb_ * nrk * nm_ * MAX(nirg, 1))
    LOGICAL, INTENT(IN) :: use_ws
    INTEGER, INTENT(IN) :: ndegk(nrk * ndims * ndims)
    COMPLEX(dp), INTENT(IN) :: cfk(nrk * nkwg)
    COMPLEX(dp), INTENT(OUT) :: hp(nb_ * nb_ * nm_ * MAX(nirg, 1) * nkwg)
    COMPLEX(dp), DEVICE, ALLOCATABLE :: d_s(:), d_cfk(:), d_hp(:)
    INTEGER, DEVICE, ALLOCATABLE :: d_nd(:)
    INTEGER :: nb2, nbat, b0, nbc, iws
    INTEGER(8) :: chunk
    nb2 = nb_ * nb_
    nbat = nm_ * nirg
    IF (nbat <= 0 .OR. nkwg <= 0) THEN
      hp = (0.0_dp, 0.0_dp)
      RETURN
    ENDIF
    IF (.NOT. have_hb) THEN
      CALL chk(cublasCreate(hb), 'cublasCreate')
      have_hb = .TRUE.
    ENDIF
    ! chunks of b = (nu, rpl) of at most ~256 MB scaled copies
    chunk = MAX(1_8, MIN(INT(nbat, 8), 268435456_8 / (16_8 * nb2 * nrk)))
    ALLOCATE(d_s(nb2 * nrk * chunk), d_cfk(nrk * nkwg), d_hp(INT(nb2, 8) * nbat * nkwg), d_nd(nrk * ndims * ndims))
    d_cfk = cfk; d_nd = ndegk
    iws = 0
    IF (use_ws) iws = 1
    DO b0 = 1, nbat, INT(chunk)
      nbc = MIN(INT(chunk), nbat - b0 + 1)
      CALL k_scale<<<dim3((nb2 + 63) / 64, nrk, nbc), dim3(64, 1, 1)>>>(nb2, nrk, nbc, iws, d_nd, &
           epmatwp(INT(nb2, 8) * nrk * (b0 - 1) + 1:), d_s)
      CALL chk(cudaGetLastError(), 'k_scale')
      ! hp(:, b, kw) = s(:, :, b) . cfk(:, kw): batched over b, ldc = nb2 * nbat, strideC = nb2
      CALL chk(cublasZgemmStridedBatched(hb, CUBLAS_OP_N, CUBLAS_OP_N, nb2, nkwg, nrk, &
           (1.0_dp, 0.0_dp), d_s, nb2, INT(nb2, 8) * nrk, d_cfk, nrk, 0_8, (0.0_dp, 0.0_dp), &
           d_hp(INT(nb2, 8) * (b0 - 1) + 1:), nb2 * nbat, INT(nb2, 8), nbc), 'zgemm hpart')
    ENDDO
    hp = d_hp
    DEALLOCATE(d_s, d_cfk, d_hp, d_nd)
  END SUBROUTINE ef_hpart
  !
  SUBROUTINE ef_set_h(h)
    !! H(mn nu, Rp, kw) for the pool's Fermi-shell k, all Rp
    COMPLEX(dp), INTENT(IN) :: h(nb * nb * nm * nrg * MAX(nkw, 1))
    IF (ALLOCATED(d_h)) DEALLOCATE(d_h)
    ALLOCATE(d_h(INT(nb * nb * nm, 8) * nrg * MAX(nkw, 1)))
    d_h = h
    have_h = .TRUE.
  END SUBROUTINE ef_set_h
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_pair(nb, nm, nbf, ibmin, nqb, nkw, nkfs, x, cufk, ek, cufg, eg, uf, &
                                       wf, win, pair, out)
    !! One block per pair. pair(:, p) = (kw, j, gkq, ikfs, ikqfs, flag, offset).
    INTEGER, VALUE :: nb, nm, nbf, ibmin, nqb, nkw, nkfs
    COMPLEX(dp), DEVICE :: x(nb, nb, nm, nqb, nkw), cufk(nb, nb, *), cufg(nb, nb, *), uf(nm, nm, nqb)
    REAL(dp), DEVICE :: ek(nb, *), eg(nb, *), wf(nm, nqb), out(*)
    INTEGER, DEVICE :: win(nbf, nkfs), pair(7, *)
    COMPLEX(dp), SHARED :: sc(*)
    INTEGER :: p, kw, j, gkq, ikfs, ikqfs, flag, off, t, nt, ia, ib, ic, nu, mu, n, pb, nwi, nwj, ri, rj
    INTEGER :: oq, ok, ot, om, oe
    COMPLEX(dp) :: z
    REAL(dp) :: g, w1, w2
    !
    p = blockIdx%x
    t = threadIdx%x
    nt = blockDim%x
    kw = pair(1, p); j = pair(2, p); gkq = pair(3, p); ikfs = pair(4, p); ikqfs = pair(5, p)
    flag = pair(6, p); off = pair(7, p)
    ! shared layout (complex units): ckq(nbf, nb) | ck(nbf, nb) | T(nbf, nb) | M(nbf, nbf, nm) |
    ! E(nbf, nbf, nm) | then real A1(nbf, nbf, nm) stored in the real view after that
    oq = 0
    ok = nbf * nb
    ot = 2 * nbf * nb
    om = 3 * nbf * nb
    oe = om + nbf * nbf * nm
    IF (flag == 1) THEN
      DO ia = t, nbf * nb, nt
        ib = MOD(ia - 1, nbf) + 1           ! band row (ibmin - 1 + ib)
        ic = (ia - 1) / nbf + 1             ! Wannier column
        sc(oq + ia) = cufg(ibmin - 1 + ib, ic, gkq)
        sc(ok + ia) = cufk(ibmin - 1 + ib, ic, kw)
      ENDDO
      CALL syncthreads()
      DO nu = 1, nm
        ! T(jb, d) = sum_c cufkq(jb, c) X(c, d, nu)
        DO ia = t, nbf * nb, nt
          ib = MOD(ia - 1, nbf) + 1
          ic = (ia - 1) / nbf + 1
          z = (0.0_dp, 0.0_dp)
          DO n = 1, nb
            z = z + sc(oq + ib + nbf * (n - 1)) * x(n, ic, nu, j, kw)
          ENDDO
          sc(ot + ia) = z
        ENDDO
        CALL syncthreads()
        ! M(jb, ibb, nu) = sum_d T(jb, d) conj(cufkk(ibb, d))
        DO ia = t, nbf * nbf, nt
          ib = MOD(ia - 1, nbf) + 1
          ic = (ia - 1) / nbf + 1
          z = (0.0_dp, 0.0_dp)
          DO n = 1, nb
            z = z + sc(ot + ib + nbf * (n - 1)) * CONJG(sc(ok + ic + nbf * (n - 1)))
          ENDDO
          sc(om + ia + nbf * nbf * (nu - 1)) = z
        ENDDO
        CALL syncthreads()
      ENDDO
      ! E(jb, ibb, mu) = sum_nu M(jb, ibb, nu) uf(nu, mu)
      DO ia = t, nbf * nbf * nm, nt
        ib = MOD(ia - 1, nbf * nbf) + 1
        mu = (ia - 1) / (nbf * nbf) + 1
        z = (0.0_dp, 0.0_dp)
        DO nu = 1, nm
          z = z + sc(om + ib + nbf * nbf * (nu - 1)) * uf(nu, mu, j)
        ENDDO
        sc(oe + ia) = z
      ENDDO
    ELSE
      DO ia = t, nbf * nbf * nm, nt
        sc(oe + ia) = (0.0_dp, 0.0_dp)
      ENDDO
    ENDIF
    CALL syncthreads()
    ! average over degenerate k bands: A1(jb, ibb, mu) = sqrt(sum_p |E(jb, p, mu)|^2 / n),
    ! stored as complex (re = A1) in the M area
    DO ia = t, nbf * nbf * nm, nt
      ib = MOD(ia - 1, nbf) + 1                       ! jbnd (k+q)
      ic = MOD((ia - 1) / nbf, nbf) + 1               ! ibnd (k)
      mu = (ia - 1) / (nbf * nbf) + 1
      w1 = ek(ibmin - 1 + ic, kw)
      g = 0.0_dp
      n = 0
      DO pb = 1, nbf
        w2 = ek(ibmin - 1 + pb, kw)
        IF (ABS(w2 - w1) < eps6) THEN
          n = n + 1
          g = g + ABS(sc(oe + ib + nbf * (pb - 1) + nbf * nbf * (mu - 1)))**2
        ENDIF
      ENDDO
      sc(om + ia) = CMPLX(SQRT(g / REAL(n, 4)), 0.0_dp, KIND = dp)
    ENDDO
    CALL syncthreads()
    ! average over degenerate k+q bands and write g^2 (write_ephmat order: mu, ibnd, jbnd over the
    ! ekfs windows of k and k+q)
    nwi = 0
    nwj = 0
    DO pb = 1, nbf
      nwi = nwi + win(pb, ikfs)
      nwj = nwj + win(pb, ikqfs)
    ENDDO
    DO ia = t, nbf * nbf * nm, nt
      ib = MOD(ia - 1, nbf) + 1                       ! jbnd
      ic = MOD((ia - 1) / nbf, nbf) + 1               ! ibnd
      mu = (ia - 1) / (nbf * nbf) + 1
      IF (win(ic, ikfs) == 0 .OR. win(ib, ikqfs) == 0) CYCLE
      w1 = eg(ibmin - 1 + ib, gkq)
      g = 0.0_dp
      n = 0
      DO pb = 1, nbf
        w2 = eg(ibmin - 1 + pb, gkq)
        IF (ABS(w2 - w1) < eps6) THEN
          n = n + 1
          g = g + ABS(sc(om + pb + nbf * (ic - 1) + nbf * nbf * (mu - 1)))**2
        ENDIF
      ENDDO
      g = g / REAL(n, 4)
      ri = 0
      DO pb = 1, ic - 1
        ri = ri + win(pb, ikfs)
      ENDDO
      rj = 0
      DO pb = 1, ib - 1
        rj = rj + win(pb, ikqfs)
      ENDDO
      out(off + ((mu - 1) * nwi + ri) * nwj + rj + 1) = ABS(g) * (1.0_dp / (2.0_dp * wf(mu, j)))
    ENDDO
  END SUBROUTINE k_pair
  !
  SUBROUTINE ef_batch(nqb, cq, uf, wf, npair, pair, nout, out)
    !! X for the nqb q of the batch (cq(Rp, j) = e^{i q_j.Rp}), then the pair kernel.
    INTEGER, INTENT(IN) :: nqb, npair, nout
    COMPLEX(dp), INTENT(IN) :: cq(nrg * nqb), uf(nm * nm * nqb)
    REAL(dp), INTENT(IN) :: wf(nm * nqb)
    INTEGER, INTENT(IN) :: pair(7 * MAX(npair, 1))
    REAL(dp), INTENT(OUT) :: out(MAX(nout, 1))
    INTEGER :: nmn, shb
    IF (npair <= 0) RETURN
    nmn = nb * nb * nm
    IF (nqb > nqb_alloc) THEN
      IF (ALLOCATED(d_x)) DEALLOCATE(d_x, d_cq, d_uf, d_wf)
      ALLOCATE(d_x(INT(nmn, 8) * nqb * nkw), d_cq(nrg * nqb), d_uf(nm * nm * nqb), d_wf(nm * nqb))
      nqb_alloc = nqb
    ENDIF
    IF (npair > npair_alloc) THEN
      IF (ALLOCATED(d_pair)) DEALLOCATE(d_pair)
      ALLOCATE(d_pair(7 * npair))
      npair_alloc = npair
    ENDIF
    IF (nout > nout_alloc) THEN
      IF (ALLOCATED(d_out)) DEALLOCATE(d_out)
      ALLOCATE(d_out(MAX(nout, 1)))
      nout_alloc = nout
    ENDIF
    d_cq(1:nrg * nqb) = cq
    d_uf(1:nm * nm * nqb) = uf
    d_wf(1:nm * nqb) = wf
    d_pair(1:7 * npair) = pair(1:7 * npair)
    ! X(:, j, kw) = H(:, :, kw) . cq(:, j)
    CALL chk(cublasZgemmStridedBatched(hb, CUBLAS_OP_N, CUBLAS_OP_N, nmn, nqb, nrg, (1.0_dp, 0.0_dp), &
         d_h, nmn, INT(nmn, 8) * nrg, d_cq, nrg, 0_8, (0.0_dp, 0.0_dp), d_x, nmn, INT(nmn, 8) * nqb, nkw), &
         'zgemm X')
    shb = ef_shmem(nb, nm, nbf)
    CALL k_pair<<<npair, 64, shb>>>(nb, nm, nbf, ibmin, nqb, nkw, nkfs_, d_x, d_cufk, d_ek, d_cufg, d_eg, &
         d_uf, d_wf, d_win, d_pair, d_out)
    CALL chk(cudaGetLastError(), 'k_pair')
    IF (nout > 0) out(1:nout) = d_out(1:nout)
  END SUBROUTINE ef_batch
  !
END MODULE epw_gpu_fine
