!
! rtsc: GPU version of sum_eliashberg_aniso_iaxis_fsr_simple (anisotropic Migdal-Eliashberg,
! Fermi-surface restricted, uniform positive Matsubara sampling).
!
! For each (k, n) in the Fermi shell and each Matsubara frequency iw, vanilla accumulates over
! (q, m) pairs and iwp:
!   naz(iw) += w dw(iwp) [l(|n-n'|) - l(n+n'+1)]
!   az(iw)  += w dw(iwp) / sqrt(wp^2 + D^2) * wp * [l(|n-n'|) - l(n+n'+1)]
!   ad(iw)  += w dw(iwp) / sqrt(wp^2 + D^2) * D  * [l(|n-n'|) + l(n+n'+1) - 2 mu* s]
! where l(m) = lambda_kq(omega_m) is the akeri kernel. Here each thread block owns one (k, n)
! and a chunk of q; for every pair it builds l(1:2n) in shared memory from g2 and the phonon
! frequencies (the same formula as lambdar_aniso_ver1), so akeri is never stored. One thread
! per iw accumulates in vanilla's (q, m, iwp) order. Partial sums over q chunks are added in a
! fixed order (deterministic).
!
MODULE epw_gpu_eliash
  USE cudafor
  USE cublas_v2
  IMPLICIT NONE
  PRIVATE
  INTEGER, PARAMETER :: dp = KIND(1.d0)
  !
  REAL(dp), DEVICE, ALLOCATABLE :: d_g2(:), d_wf(:), d_ekfs(:), d_w0g(:), d_wqf(:)
  INTEGER, DEVICE, ALLOCATABLE :: d_ixkqf(:), d_ixqfs(:), d_nqfs(:)
  REAL(dp), DEVICE, ALLOCATABLE :: d_wsi(:), d_dwsi(:), d_adp(:), d_part(:)
  INTEGER, DEVICE, ALLOCATABLE :: d_wsn(:)
  ! F(m, nu, q) = wf / (wf^2 + omega_m^2) (0 for wf <= eps_acoustic), per temperature
  REAL(dp), DEVICE, ALLOCATABLE :: d_ftab(:)
  ! per iteration: ez = dw * wp / sqrt(wp^2 + D^2), ed = dw * D / sqrt(wp^2 + D^2)
  REAL(dp), DEVICE, ALLOCATABLE :: d_ez(:), d_ed(:)
  REAL(dp) :: ftab_pit = -1.0_dp
  INTEGER :: ftab_n2 = -1
  INTEGER(8) :: sz_static = -1
  INTEGER :: nsplit = 1
  INTEGER :: nsiw_alloc = -1, nsplit_alloc = -1, nadp_alloc = -1
  !
  ! Low-rank (GEMM) path. The kernel row of a pair, l(m) = 2 dosef sum_nu g2 f_nu(m) with
  ! f_nu(m) = w_nu / (w_nu^2 + omega_m^2), lies (to ~1e-14 relative) in the span of R orthonormal
  ! vectors u_r(m) (SVD of f over a log grid of w spanning the phonon frequencies). With
  ! c_r(nu, q) = 2 dosef sum_m u_r(m) f_nu(m), the sums of k_fsr become, per pool,
  !   Z(kn, r, c) = sum_s' A(kn, r, s') B(s', c)               (GEMM 1, c = iwp of ez | ed)
  !   az(iw, kn)  = sum_{r, iwp} [u_r(|iw-iwp|) - u_r(iw+iwp-1)] Z(kn, r, iwp)        (GEMM 2)
  !   ad(iw, kn)  = sum_{r, iwp} [u_r(|iw-iwp|) + u_r(iw+iwp-1)] Z(kn, r, N + iwp)
  !                 - mu*s sum_iwp Z(kn, 0, N + iwp)
  ! where A(kn, r, s') = sum over pairs (q, jbnd) of k with (jbnd, k+q) = s' of
  ! weight * sum_nu g2 c_r (r >= 1) or weight (r = 0), built once per temperature. naz does not
  ! depend on Delta and is computed once per temperature. Same sums, different FP order.
  INTEGER, PARAMETER :: lr_ngrid = 400, lr_rmax = 64
  REAL(dp), PARAMETER :: lr_svtol = 1.0E-15_dp
  REAL(dp), DEVICE, ALLOCATABLE :: d_lra(:), d_lrz(:), d_lrb(:), d_lrwm(:), d_lrwp(:), &
                                   d_lru(:), d_lrc(:), d_lrout(:), d_lrres(:)
  REAL(dp), ALLOCATABLE :: lr_naz(:)
  REAL(dp) :: lr_pit = -1.0_dp, lr_fail_pit = -1.0_dp
  INTEGER :: lr_nsiw = -1, lr_r = 0, lr_fail_nsiw = -1
  LOGICAL :: lr_have_handle = .FALSE.
  TYPE(cublasHandle) :: lr_h
  !
  PUBLIC :: ege_fsr, ege_reset, ege_pade
  !
CONTAINS
  !
  SUBROUTINE chk(istat, what)
    INTEGER, INTENT(IN) :: istat
    CHARACTER(*), INTENT(IN) :: what
    IF (istat /= 0) THEN
      WRITE(*, '(a, a, a, i0, 1x, a)') 'epw_gpu_eliash: ', what, ' failed, status ', istat, &
                                      TRIM(cudaGetErrorString(istat))
      STOP 1
    ENDIF
  END SUBROUTINE chk
  !
  SUBROUTINE ege_reset()
    !! Drop the device copies of the temperature-independent arrays.
    IF (ALLOCATED(d_g2)) DEALLOCATE(d_g2, d_wf, d_ekfs, d_w0g, d_wqf, d_ixkqf, d_ixqfs, d_nqfs)
    IF (ALLOCATED(d_wsi)) DEALLOCATE(d_wsi, d_dwsi, d_wsn, d_adp, d_part)
    IF (ALLOCATED(d_ftab)) DEALLOCATE(d_ftab)
    IF (ALLOCATED(d_ez)) DEALLOCATE(d_ez, d_ed)
    ftab_pit = -1.0_dp; ftab_n2 = -1
    sz_static = -1
    lr_fail_pit = -1.0_dp; lr_fail_nsiw = -1
    nsiw_alloc = -1; nsplit_alloc = -1; nadp_alloc = -1
    CALL lr_free()
  END SUBROUTINE ege_reset
  !
  SUBROUTINE lr_free()
    IF (ALLOCATED(d_lra)) DEALLOCATE(d_lra)
    IF (ALLOCATED(d_lrz)) DEALLOCATE(d_lrz)
    IF (ALLOCATED(d_lrb)) DEALLOCATE(d_lrb)
    IF (ALLOCATED(d_lrwm)) DEALLOCATE(d_lrwm, d_lrwp)
    IF (ALLOCATED(d_lru)) DEALLOCATE(d_lru)
    IF (ALLOCATED(d_lrc)) DEALLOCATE(d_lrc)
    IF (ALLOCATED(d_lrout)) DEALLOCATE(d_lrout)
    IF (ALLOCATED(d_lrres)) DEALLOCATE(d_lrres)
    IF (ALLOCATED(lr_naz)) DEALLOCATE(lr_naz)
    lr_pit = -1.0_dp; lr_nsiw = -1; lr_r = 0
  END SUBROUTINE lr_free
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_lrcoef(n2, nr, ncol, wf, eps_ac, pit, scal, u, c, res)
    !! c(r, col) = scal * sum_m u(m, r) f_col(m), f_col(m) = w / (w^2 + (2 (m-1) pit)^2) (0 if
    !! w <= eps_ac); res(col) = max_m |f - sum_r u c / scal| / f(1): the projection residual.
    INTEGER, VALUE :: n2, nr, ncol
    REAL(dp), VALUE :: eps_ac, pit, scal
    REAL(dp), DEVICE :: wf(ncol), u(n2, nr), c(nr, ncol), res(ncol)
    INTEGER :: col, m, r
    REAL(dp) :: w, om, f, acc(lr_rmax), rec, e
    col = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    IF (col > ncol) RETURN
    w = wf(col)
    IF (w <= eps_ac) THEN
      DO r = 1, nr
        c(r, col) = 0.0_dp
      ENDDO
      res(col) = 0.0_dp
      RETURN
    ENDIF
    DO r = 1, nr
      acc(r) = 0.0_dp
    ENDDO
    DO m = 1, n2
      om = DBLE(2 * (m - 1)) * pit
      f = w / (w * w + om * om)
      DO r = 1, nr
        acc(r) = acc(r) + u(m, r) * f
      ENDDO
    ENDDO
    e = 0.0_dp
    DO m = 1, n2
      om = DBLE(2 * (m - 1)) * pit
      f = w / (w * w + om * om)
      rec = 0.0_dp
      DO r = 1, nr
        rec = rec + u(m, r) * acc(r)
      ENDDO
      e = MAX(e, ABS(f - rec))
    ENDDO
    res(col) = e * w      ! relative to f(1) = 1 / w
    DO r = 1, nr
      c(r, col) = scal * acc(r)
    ENDDO
  END SUBROUTINE k_lrcoef
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_lrbuild(nkl, kl0, nb, mq, nm, nr1, nkn, nst, g2, c, ixkqf, ixqfs, &
                                          nqfs, ekfs, w0g, wqf, a, ef0, fsthick, inv_dos)
    !! A(kn, r, s') for kn = ibnd + nb (ikl - 1), r = 0..nr1-1, s' = jbnd + nb (kq - 1).
    !! One thread per (kn, r); pairs in vanilla's (q, jbnd) order; no atomics (deterministic).
    INTEGER, VALUE :: nkl, kl0, nb, mq, nm, nr1, nkn, nst
    REAL(dp), VALUE :: ef0, fsthick, inv_dos
    REAL(dp), DEVICE :: g2(nkl, mq, nb, nb, nm), c(nr1 - 1, nm, *), ekfs(nb, *), w0g(nb, *), wqf(*)
    INTEGER, DEVICE :: ixkqf(nkl, *), ixqfs(nkl, *), nqfs(*)
    REAL(dp), DEVICE :: a(nkn, 0:nr1 - 1, nst)
    INTEGER :: r, kn, ikl, ik, ibnd, iq, iq0, kq, jbnd, nu, s
    REAL(dp) :: weight, v
    r = threadIdx%x - 1
    kn = blockIdx%x
    IF (r >= nr1) RETURN
    ibnd = MOD(kn - 1, nb) + 1
    ikl = (kn - 1) / nb + 1
    ik = kl0 - 1 + ikl
    IF (ABS(ekfs(ibnd, ik) - ef0) >= fsthick) RETURN
    DO iq = 1, nqfs(ik)
      iq0 = ixqfs(ikl, iq)
      kq = ixkqf(ikl, iq0)
      DO jbnd = 1, nb
        IF (ABS(ekfs(jbnd, kq) - ef0) < fsthick) THEN
          weight = wqf(iq) * w0g(jbnd, kq) * inv_dos
          IF (r == 0) THEN
            v = weight
          ELSE
            v = 0.0_dp
            DO nu = 1, nm
              v = v + g2(ikl, iq, ibnd, jbnd, nu) * c(r, nu, iq0)
            ENDDO
            v = weight * v
          ENDIF
          s = jbnd + nb * (kq - 1)
          a(kn, r, s) = a(kn, r, s) + v
        ENDIF
      ENDDO
    ENDDO
  END SUBROUTINE k_lrbuild
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_lrnaz(nkn, nr1, nst, nsiw, a, dwsi, z)
    !! z(kn, r + nr1 (iwp - 1)) = dwsi(iwp) * sum_s' A(kn, r, s') (the naz input, r >= 1).
    INTEGER, VALUE :: nkn, nr1, nst, nsiw
    REAL(dp), DEVICE :: a(nkn, 0:nr1 - 1, nst), dwsi(nsiw), z(nkn, 0:nr1 - 1, nsiw)
    INTEGER :: kn, r, s, iwp
    REAL(dp) :: t
    kn = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    r = blockIdx%y - 1
    IF (kn > nkn) RETURN
    t = 0.0_dp
    IF (r > 0) THEN
      DO s = 1, nst
        t = t + a(kn, r, s)
      ENDDO
    ENDIF
    DO iwp = 1, nsiw
      z(kn, r, iwp) = t * dwsi(iwp)
    ENDDO
  END SUBROUTINE k_lrnaz
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_lrw(n2, nr1, nsiw, u, mucs, wm, wp)
    !! wm/wp(r + nr1 (iwp - 1), iw) = u_r(|iw - iwp|) -/+ u_r(iw + iwp - 1) (0-based kernel index);
    !! r = 0: wm = 0, wp = -mucs.
    INTEGER, VALUE :: n2, nr1, nsiw
    REAL(dp), VALUE :: mucs
    REAL(dp), DEVICE :: u(n2, nr1 - 1), wm(0:nr1 - 1, nsiw, nsiw), wp(0:nr1 - 1, nsiw, nsiw)
    INTEGER :: r, iwp, iw
    REAL(dp) :: lm, lp
    iwp = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    iw = blockIdx%y
    IF (iwp > nsiw) RETURN
    wm(0, iwp, iw) = 0.0_dp
    wp(0, iwp, iw) = -mucs
    DO r = 1, nr1 - 1
      lm = u(ABS(iw - iwp) + 1, r)
      lp = u(iw + iwp, r)
      wm(r, iwp, iw) = lm - lp
      wp(r, iwp, iw) = lm + lp
    ENDDO
  END SUBROUTINE k_lrw
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_lrb(nsiw, nst, wsi, dwsi, adp, b)
    !! b(iwp, s') = dw wp / sqrt(wp^2 + D^2), b(nsiw + iwp, s') = dw D / sqrt(wp^2 + D^2)
    INTEGER, VALUE :: nsiw, nst
    REAL(dp), DEVICE :: wsi(nsiw), dwsi(nsiw), adp(nsiw, nst), b(2 * nsiw, nst)
    INTEGER :: iw, s
    REAL(dp) :: d, e
    iw = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    s = blockIdx%y
    IF (iw > nsiw) RETURN
    d = adp(iw, s)
    e = dwsi(iw) / SQRT(wsi(iw) * wsi(iw) + d * d)
    b(iw, s) = e * wsi(iw)
    b(nsiw + iw, s) = e * d
  END SUBROUTINE k_lrb
  !
  SUBROUTINE lr_setup(nkl, kl0, nkfs, nb, mq, nm, nqtot, nsiw, n2, wf, ef0, fsthick, eps_ac, &
                      dosef, pit, inv_dos, mucs, ok)
    !! Per temperature: basis u_r, coefficients c_r(nu, q), A, the GEMM-2 kernels and naz.
    INTEGER, INTENT(IN) :: nkl, kl0, nkfs, nb, mq, nm, nqtot, nsiw, n2
    REAL(dp), INTENT(IN) :: wf(nm * nqtot), ef0, fsthick, eps_ac, dosef, pit, inv_dos, mucs
    LOGICAL, INTENT(OUT) :: ok
    INTEGER :: istat, i, m, ng, nr, nr1, nkn, nst, lwork, info, ncol, j
    REAL(dp) :: wmin, wmax, w, om, resmax
    REAL(dp), ALLOCATABLE :: f(:, :), s(:), u(:, :), vt(:, :), work(:), res(:)
    INTEGER(8) :: na
    !
    ok = .FALSE.
    ncol = nm * nqtot
    wmin = HUGE(1.0_dp); wmax = 0.0_dp
    DO i = 1, ncol
      IF (wf(i) > eps_ac) THEN
        wmin = MIN(wmin, wf(i)); wmax = MAX(wmax, wf(i))
      ENDIF
    ENDDO
    IF (wmax <= 0.0_dp) THEN
      IF (kl0 == 1) WRITE(*, '(a)') ' epw_gpu_eliash: no phonon above eps_acoustic'
      RETURN
    ENDIF
    ! SVD of f over a log grid of w (columns scaled to max 1)
    ng = lr_ngrid
    ALLOCATE(f(n2, ng), s(MIN(n2, ng)), u(n2, MIN(n2, ng)), vt(1, 1))
    DO j = 1, ng
      w = wmin * (wmax / wmin)**(DBLE(j - 1) / DBLE(ng - 1))
      IF (ng == 1 .OR. wmax <= wmin) w = wmin
      DO m = 1, n2
        om = DBLE(2 * (m - 1)) * pit
        f(m, j) = w * w / (w * w + om * om)
      ENDDO
    ENDDO
    ALLOCATE(work(1))
    CALL DGESVD('S', 'N', n2, ng, f, n2, s, u, n2, vt, 1, work, -1, info)
    lwork = INT(work(1)); DEALLOCATE(work); ALLOCATE(work(lwork))
    CALL DGESVD('S', 'N', n2, ng, f, n2, s, u, n2, vt, 1, work, lwork, info)
    IF (info /= 0) THEN
      IF (kl0 == 1) WRITE(*, '(a, i0)') ' epw_gpu_eliash: DGESVD info ', info
      RETURN
    ENDIF
    nr = 0
    DO i = 1, MIN(n2, ng)
      IF (s(i) > lr_svtol * s(1)) nr = i
    ENDDO
    nr = MIN(nr + 1, MIN(n2, ng), lr_rmax)
    nr1 = nr + 1
    nkn = nb * nkl
    nst = nb * nkfs
    na = INT(nkn, 8) * nr1 * nst
    IF (na * 8 > 6_8 * 1024**3) THEN     ! > 6 GB per pool: keep the direct kernel
      IF (kl0 == 1) WRITE(*, '(a, f10.1, a)') ' epw_gpu_eliash: A would need ', &
         DBLE(na) * 8.0_dp / 1024.0_dp**3, ' GB per pool (cap 6 GB)'
      RETURN
    ENDIF
    !
    CALL lr_free()
    ALLOCATE(d_lru(n2 * nr), d_lrc(nr * ncol), d_lrres(ncol), res(ncol))
    istat = cudaMemcpy(d_lru, u, n2 * nr); CALL chk(istat, 'H2D u')
    CALL k_lrcoef<<<(ncol + 127) / 128, 128>>>(n2, nr, ncol, d_wf, eps_ac, pit, 2.0_dp * dosef, &
         d_lru, d_lrc, d_lrres)
    istat = cudaGetLastError(); CALL chk(istat, 'k_lrcoef')
    res = d_lrres
    resmax = MAXVAL(res)
    IF (resmax > 1.0E-12_dp) THEN
      WRITE(*, '(a, es10.3, a)') ' epw_gpu_eliash: low-rank residual ', resmax, ', using the direct kernel'
      CALL lr_free()
      RETURN
    ENDIF
    !
    ALLOCATE(d_lra(na))
    istat = cudaMemset(d_lra, 0.0_dp, na); CALL chk(istat, 'memset A')
    CALL k_lrbuild<<<nkn, 32 * ((nr1 + 31) / 32)>>>(nkl, kl0, nb, mq, nm, nr1, nkn, nst, d_g2, &
         d_lrc, d_ixkqf, d_ixqfs, d_nqfs, d_ekfs, d_w0g, d_wqf, d_lra, ef0, fsthick, inv_dos)
    istat = cudaGetLastError(); CALL chk(istat, 'k_lrbuild')
    !
    ALLOCATE(d_lrwm(nr1 * nsiw * nsiw), d_lrwp(nr1 * nsiw * nsiw))
    CALL k_lrw<<<dim3((nsiw + 127) / 128, nsiw, 1), dim3(128, 1, 1)>>>(n2, nr1, nsiw, d_lru, mucs, &
         d_lrwm, d_lrwp)
    istat = cudaGetLastError(); CALL chk(istat, 'k_lrw')
    ALLOCATE(d_lrz(INT(nkn, 8) * nr1 * 2 * nsiw), d_lrb(2 * nsiw * nst), d_lrout(2 * nsiw * nkn))
    ! naz (Delta independent)
    CALL k_lrnaz<<<dim3((nkn + 127) / 128, nr1, 1), dim3(128, 1, 1)>>>(nkn, nr1, nst, nsiw, d_lra, &
         d_dwsi, d_lrz)
    istat = cudaGetLastError(); CALL chk(istat, 'k_lrnaz')
    CALL lr_handle()
    istat = cublasDgemm(lr_h, CUBLAS_OP_T, CUBLAS_OP_T, nsiw, nkn, nr1 * nsiw, 1.0_dp, d_lrwm, &
                        nr1 * nsiw, d_lrz, nkn, 0.0_dp, d_lrout, nsiw)
    CALL chk(istat, 'dgemm naz')
    ALLOCATE(lr_naz(nsiw * nkn))
    istat = cudaMemcpy(lr_naz, d_lrout, nsiw * nkn); CALL chk(istat, 'D2H naz')
    lr_pit = pit; lr_nsiw = nsiw; lr_r = nr
    ok = .TRUE.
    IF (kl0 == 1) WRITE(*, '(a, i4, a, i4, a, es10.3, a, f8.1, a)') ' epw_gpu_eliash: low rank R = ', &
       nr, ' for 2N = ', n2, ', max residual ', resmax, ', A ', DBLE(na) * 8.0_dp / 1024.0_dp**2, ' MB'
    DEALLOCATE(f, s, u, vt, work, res)
  END SUBROUTINE lr_setup
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_pade(n, nst, nsw, z, a, b, ws, pa, pb)
    !! pade_eval (three-term recursion) for coefficient sets a and b at w = ws(iw), one thread
    !! per (iw, state).
    INTEGER, VALUE :: n, nst, nsw
    COMPLEX(dp), DEVICE :: z(n), a(n, nst), b(n, nst), pa(nsw, nst), pb(nsw, nst)
    REAL(dp), DEVICE :: ws(nsw)
    INTEGER :: iw, is, i
    COMPLEX(dp) :: w, t, a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1, d2
    iw = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    is = blockIdx%y
    IF (iw > nsw) RETURN
    w = CMPLX(ws(iw), 0.0_dp, KIND = dp)
    a0 = (0.0_dp, 0.0_dp); a1 = a(1, is); b0 = (1.0_dp, 0.0_dp); b1 = (1.0_dp, 0.0_dp)
    c0 = (0.0_dp, 0.0_dp); c1 = b(1, is); d0 = (1.0_dp, 0.0_dp); d1 = (1.0_dp, 0.0_dp)
    DO i = 2, n
      t = w - z(i - 1)
      a2 = a1 + t * a(i, is) * a0
      b2 = b1 + t * a(i, is) * b0
      c2 = c1 + t * b(i, is) * c0
      d2 = d1 + t * b(i, is) * d0
      a0 = a1; a1 = a2; b0 = b1; b1 = b2
      c0 = c1; c1 = c2; d0 = d1; d1 = d2
    ENDDO
    pa(iw, is) = a1 / b1
    pb(iw, is) = c1 / d1
  END SUBROUTINE k_pade
  !
  SUBROUTINE ege_pade(n, nst, nsw, z, a, b, ws, pa, pb)
    !! Pade approximants with coefficients a(:, is), b(:, is) on the points z, evaluated at the
    !! real frequencies ws: pa(iw, is), pb(iw, is) (as pade_eval).
    INTEGER, INTENT(IN) :: n, nst, nsw
    COMPLEX(dp), INTENT(IN) :: z(n), a(n, nst), b(n, nst)
    REAL(dp), INTENT(IN) :: ws(nsw)
    COMPLEX(dp), INTENT(OUT) :: pa(nsw, nst), pb(nsw, nst)
    COMPLEX(dp), DEVICE, ALLOCATABLE :: dz(:), da(:, :), db(:, :), dpa(:, :), dpb(:, :)
    REAL(dp), DEVICE, ALLOCATABLE :: dws(:)
    INTEGER :: istat
    IF (nst <= 0) RETURN
    ALLOCATE(dz(n), da(n, nst), db(n, nst), dpa(nsw, nst), dpb(nsw, nst), dws(nsw))
    dz = z; da = a; db = b; dws = ws
    CALL k_pade<<<dim3((nsw + 127) / 128, nst, 1), dim3(128, 1, 1)>>>(n, nst, nsw, dz, da, db, &
         dws, dpa, dpb)
    istat = cudaGetLastError(); CALL chk(istat, 'k_pade')
    pa = dpa; pb = dpb
    DEALLOCATE(dz, da, db, dpa, dpb, dws)
  END SUBROUTINE ege_pade
  !
  SUBROUTINE lr_handle()
    INTEGER :: istat
    IF (lr_have_handle) RETURN
    istat = cublasCreate(lr_h); CALL chk(istat, 'cublasCreate')
    lr_have_handle = .TRUE.
  END SUBROUTINE lr_handle
  !
  SUBROUTINE lr_iter(nkl, nkfs, nb, nsiw, naz, az, ad)
    !! One application of the summed kernel to the current Delta (d_adp).
    INTEGER, INTENT(IN) :: nkl, nkfs, nb, nsiw
    REAL(dp), INTENT(OUT) :: naz(nsiw * nb * nkl), az(nsiw * nb * nkl), ad(nsiw * nb * nkl)
    INTEGER :: istat, nkn, nst, nr1, mm, jj
    nkn = nb * nkl; nst = nb * nkfs; nr1 = lr_r + 1
    mm = nkn * nr1; jj = nr1 * nsiw
    CALL k_lrb<<<dim3((nsiw + 127) / 128, nst, 1), dim3(128, 1, 1)>>>(nsiw, nst, d_wsi, d_dwsi, &
         d_adp, d_lrb)
    istat = cudaGetLastError(); CALL chk(istat, 'k_lrb')
    ! Z(kn + nkn r, c) = sum_s' A(kn + nkn r, s') b(c, s')
    istat = cublasDgemm(lr_h, CUBLAS_OP_N, CUBLAS_OP_T, mm, 2 * nsiw, nst, 1.0_dp, d_lra, mm, &
                        d_lrb, 2 * nsiw, 0.0_dp, d_lrz, mm)
    CALL chk(istat, 'dgemm Z')
    istat = cublasDgemm(lr_h, CUBLAS_OP_T, CUBLAS_OP_T, nsiw, nkn, jj, 1.0_dp, d_lrwm, jj, &
                        d_lrz, nkn, 0.0_dp, d_lrout, nsiw)
    CALL chk(istat, 'dgemm az')
    istat = cublasDgemm(lr_h, CUBLAS_OP_T, CUBLAS_OP_T, nsiw, nkn, jj, 1.0_dp, d_lrwp, jj, &
                        d_lrz(INT(nkn, 8) * jj + 1:), nkn, 0.0_dp, d_lrout(nsiw * nkn + 1:), nsiw)
    CALL chk(istat, 'dgemm ad')
    istat = cudaMemcpy(az, d_lrout, nsiw * nkn); CALL chk(istat, 'D2H az')
    istat = cudaMemcpy(ad, d_lrout(nsiw * nkn + 1:), nsiw * nkn); CALL chk(istat, 'D2H ad')
    naz = lr_naz
  END SUBROUTINE lr_iter
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_ftab(n2, nm, nqtot, wf, eps_ac, pit, ftab)
    !! ftab(m, nu, q) = wf(nu,q) / (wf^2 + omega_m^2), omega_m = 2 (m-1) pi T (lambdar_aniso_ver1)
    INTEGER, VALUE :: n2, nm, nqtot
    REAL(dp), VALUE :: eps_ac, pit
    REAL(dp), DEVICE :: wf(nm, nqtot), ftab(n2, nm, nqtot)
    INTEGER :: m, nu, iq
    REAL(dp) :: w, om
    m = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    nu = blockIdx%y
    iq = blockIdx%z
    IF (m > n2) RETURN
    w = wf(nu, iq)
    om = DBLE(2 * (m - 1)) * pit
    IF (w > eps_ac) THEN
      ftab(m, nu, iq) = w / (w * w + om * om)
    ELSE
      ftab(m, nu, iq) = 0.0_dp
    ENDIF
  END SUBROUTINE k_ftab
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_etab(nsiw, ntot, wsi, dwsi, adp, ez, ed)
    INTEGER, VALUE :: nsiw, ntot
    REAL(dp), DEVICE :: wsi(nsiw), dwsi(nsiw), adp(nsiw, *), ez(nsiw, *), ed(nsiw, *)
    INTEGER :: iw, j
    REAL(dp) :: d, r
    iw = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    j = blockIdx%y
    IF (iw > nsiw .OR. j > ntot) RETURN
    d = adp(iw, j)
    r = dwsi(iw) / SQRT(wsi(iw) * wsi(iw) + d * d)
    ez(iw, j) = r * wsi(iw)
    ed(iw, j) = r * d
  END SUBROUTINE k_etab
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_fsr5(nkl, kl0, nb, mq, nm, nsiw, n2, n2p, nchunk, ntile, &
                                       g2, ftab, ixkqf, ixqfs, nqfs, ekfs, w0g, wqf, &
                                       dwsi, ez, ed, part, ef0, fsthick, dosef, inv_dos, mucs)
    !! As k_fsr4 (uniform grid, 4 consecutive iw per lane, sliding register windows), but a
    !! block is 4 warps working on different q of the same (k, n, 128-wide iw tile): each warp
    !! builds its own kernel row (warp sync only), the 1/sqrt(wp^2+D^2) factors come from the
    !! per-iteration tables ez/ed, and the pair weight multiplies the iwp sum once. The four
    !! warps' partial sums are added in fixed order at the end.
    INTEGER, VALUE :: nkl, kl0, nb, mq, nm, nsiw, n2, n2p, nchunk, ntile
    REAL(dp), VALUE :: ef0, fsthick, dosef, inv_dos, mucs
    REAL(dp), DEVICE :: g2(nkl, mq, nb, nb, nm), ftab(n2, nm, *), ekfs(nb, *), w0g(nb, *), wqf(*)
    INTEGER, DEVICE :: ixkqf(nkl, *), ixqfs(nkl, *), nqfs(*)
    REAL(dp), DEVICE :: dwsi(nsiw), ez(nsiw, nb, *), ed(nsiw, nb, *), part(nsiw, 3, nb, nkl, *)
    REAL(dp), SHARED :: sh(*)
    INTEGER :: ikl, ik, ibnd, ic, it, iq, iq0, iq1, iq2, kq, jbnd, m, nu, iwp, lane, w, i0, ko, ro
    REAL(dp) :: weight, lam, km, w1, w2, w3
    REAL(dp) :: m0, m1, m2, m3, p0, p1, p2, p3
    REAL(dp) :: sn0, sn1, sn2, sn3, sz0, sz1, sz2, sz3, sd0, sd1, sd2, sd3
    REAL(dp) :: an0, an1, an2, an3, az0, az1, az2, az3, ad0, ad1, ad2, ad3
    !
    ikl = blockIdx%x
    it = (blockIdx%y - 1) / nb
    ibnd = MOD(blockIdx%y - 1, nb) + 1
    ic = blockIdx%z
    ik = kl0 - 1 + ikl
    lane = MOD(threadIdx%x - 1, 32)
    w = (threadIdx%x - 1) / 32
    i0 = 128 * it + 4 * lane + 1
    ko = w * n2p              ! this warp's kernel row: sh(ko+1 : ko+n2p)
    an0 = 0.0_dp; an1 = 0.0_dp; an2 = 0.0_dp; an3 = 0.0_dp
    az0 = 0.0_dp; az1 = 0.0_dp; az2 = 0.0_dp; az3 = 0.0_dp
    ad0 = 0.0_dp; ad1 = 0.0_dp; ad2 = 0.0_dp; ad3 = 0.0_dp
    IF (ABS(ekfs(ibnd, ik) - ef0) < fsthick) THEN
      iq1 = (ic - 1) * nchunk + 1
      iq2 = MIN(nqfs(ik), ic * nchunk)
      DO iq = iq1 + w, iq2, 4
        iq0 = ixqfs(ikl, iq)
        kq = ixkqf(ikl, iq0)
        DO jbnd = 1, nb
          IF (ABS(ekfs(jbnd, kq) - ef0) < fsthick) THEN
            weight = wqf(iq) * w0g(jbnd, kq) * inv_dos
            CALL syncwarp(-1)
            DO m = lane + 1, n2, 32
              lam = 0.0_dp
              DO nu = 1, nm
                lam = lam + g2(ikl, iq, ibnd, jbnd, nu) * ftab(m, nu, iq0)
              ENDDO
              sh(ko + m) = 2.0_dp * lam * dosef
            ENDDO
            CALL syncwarp(-1)
            IF (i0 <= nsiw) THEN
              sn0 = 0.0_dp; sn1 = 0.0_dp; sn2 = 0.0_dp; sn3 = 0.0_dp
              sz0 = 0.0_dp; sz1 = 0.0_dp; sz2 = 0.0_dp; sz3 = 0.0_dp
              sd0 = 0.0_dp; sd1 = 0.0_dp; sd2 = 0.0_dp; sd3 = 0.0_dp
              m0 = sh(ko + i0);     m1 = sh(ko + i0 + 1); m2 = sh(ko + i0 + 2); m3 = sh(ko + i0 + 3)
              p0 = sh(ko + i0 + 1); p1 = sh(ko + i0 + 2); p2 = sh(ko + i0 + 3); p3 = sh(ko + i0 + 4)
              DO iwp = 1, nsiw
                w1 = dwsi(iwp); w2 = ez(iwp, jbnd, kq); w3 = ed(iwp, jbnd, kq)
                km = m0 - p0; sn0 = sn0 + w1 * km; sz0 = sz0 + w2 * km; sd0 = sd0 + w3 * (m0 + p0 - mucs)
                km = m1 - p1; sn1 = sn1 + w1 * km; sz1 = sz1 + w2 * km; sd1 = sd1 + w3 * (m1 + p1 - mucs)
                km = m2 - p2; sn2 = sn2 + w1 * km; sz2 = sz2 + w2 * km; sd2 = sd2 + w3 * (m2 + p2 - mucs)
                km = m3 - p3; sn3 = sn3 + w1 * km; sz3 = sz3 + w2 * km; sd3 = sd3 + w3 * (m3 + p3 - mucs)
                m3 = m2; m2 = m1; m1 = m0; m0 = sh(ko + ABS(i0 - iwp - 1) + 1)
                p0 = p1; p1 = p2; p2 = p3; p3 = sh(ko + i0 + iwp + 4)
              ENDDO
              an0 = an0 + weight * sn0; an1 = an1 + weight * sn1; an2 = an2 + weight * sn2; an3 = an3 + weight * sn3
              az0 = az0 + weight * sz0; az1 = az1 + weight * sz1; az2 = az2 + weight * sz2; az3 = az3 + weight * sz3
              ad0 = ad0 + weight * sd0; ad1 = ad1 + weight * sd1; ad2 = ad2 + weight * sd2; ad3 = ad3 + weight * sd3
            ENDIF
          ENDIF
        ENDDO
      ENDDO
    ENDIF
    ! reduce the 4 warps in fixed order: sh(ro + 12*(t-1) + v), t = thread, v = value
    CALL syncthreads()
    ro = 4 * n2p + 12 * (threadIdx%x - 1)
    sh(ro + 1) = an0; sh(ro + 2) = an1; sh(ro + 3) = an2; sh(ro + 4) = an3
    sh(ro + 5) = az0; sh(ro + 6) = az1; sh(ro + 7) = az2; sh(ro + 8) = az3
    sh(ro + 9) = ad0; sh(ro + 10) = ad1; sh(ro + 11) = ad2; sh(ro + 12) = ad3
    CALL syncthreads()
    IF (w == 0) THEN
      DO m = 2, 4
        ro = 4 * n2p + 12 * ((m - 1) * 32 + lane)
        an0 = an0 + sh(ro + 1); an1 = an1 + sh(ro + 2); an2 = an2 + sh(ro + 3); an3 = an3 + sh(ro + 4)
        az0 = az0 + sh(ro + 5); az1 = az1 + sh(ro + 6); az2 = az2 + sh(ro + 7); az3 = az3 + sh(ro + 8)
        ad0 = ad0 + sh(ro + 9); ad1 = ad1 + sh(ro + 10); ad2 = ad2 + sh(ro + 11); ad3 = ad3 + sh(ro + 12)
      ENDDO
      IF (i0 <= nsiw) THEN
        part(i0, 1, ibnd, ikl, ic) = an0; part(i0, 2, ibnd, ikl, ic) = az0; part(i0, 3, ibnd, ikl, ic) = ad0
      ENDIF
      IF (i0 + 1 <= nsiw) THEN
        part(i0 + 1, 1, ibnd, ikl, ic) = an1; part(i0 + 1, 2, ibnd, ikl, ic) = az1
        part(i0 + 1, 3, ibnd, ikl, ic) = ad1
      ENDIF
      IF (i0 + 2 <= nsiw) THEN
        part(i0 + 2, 1, ibnd, ikl, ic) = an2; part(i0 + 2, 2, ibnd, ikl, ic) = az2
        part(i0 + 2, 3, ibnd, ikl, ic) = ad2
      ENDIF
      IF (i0 + 3 <= nsiw) THEN
        part(i0 + 3, 1, ibnd, ikl, ic) = an3; part(i0 + 3, 2, ibnd, ikl, ic) = az3
        part(i0 + 3, 3, ibnd, ikl, ic) = ad3
      ENDIF
    ENDIF
  END SUBROUTINE k_fsr5
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_fsr4(nkl, kl0, nb, mq, nm, nsiw, n2, n2p, nchunk, &
                                       g2, ftab, ixkqf, ixqfs, nqfs, ekfs, w0g, wqf, &
                                       wsi, dwsi, adp, part, ef0, fsthick, dosef, inv_dos, mucs)
    !! Uniform positive Matsubara grid (wsn(iw) = iw - 1): lambda(|n-n'|) = l(|iw-iwp|+1),
    !! lambda(n+n'+1) = l(iw+iwp). Each thread owns 4 consecutive iw and keeps sliding register
    !! windows over both kernel rows, so each iwp step costs 2 shared loads (plus 3 broadcasts)
    !! for 4 outputs. The kernel row is sum_nu g2 * ftab: no divisions per pair.
    INTEGER, VALUE :: nkl, kl0, nb, mq, nm, nsiw, n2, n2p, nchunk
    REAL(dp), VALUE :: ef0, fsthick, dosef, inv_dos, mucs
    REAL(dp), DEVICE :: g2(nkl, mq, nb, nb, nm), ftab(n2, nm, *), ekfs(nb, *), w0g(nb, *), wqf(*)
    INTEGER, DEVICE :: ixkqf(nkl, *), ixqfs(nkl, *), nqfs(*)
    REAL(dp), DEVICE :: wsi(nsiw), dwsi(nsiw), adp(nsiw, nb, *), part(nsiw, 3, nb, nkl, *)
    REAL(dp), SHARED :: sh(*)
    INTEGER :: ikl, ik, ibnd, ic, iq, iq0, iq1, iq2, kq, jbnd, m, nu, iwp, tid, nthr, i0, j
    REAL(dp) :: weight, lam, d, e, km, w1, w2, w3, gv
    REAL(dp) :: m0, m1, m2, m3, p0, p1, p2, p3
    REAL(dp) :: an0, an1, an2, an3, az0, az1, az2, az3, ad0, ad1, ad2, ad3
    !
    ikl = blockIdx%x
    ibnd = blockIdx%y
    ic = blockIdx%z
    ik = kl0 - 1 + ikl
    tid = threadIdx%x
    nthr = blockDim%x
    i0 = 4 * (tid - 1) + 1
    an0 = 0.0_dp; an1 = 0.0_dp; an2 = 0.0_dp; an3 = 0.0_dp
    az0 = 0.0_dp; az1 = 0.0_dp; az2 = 0.0_dp; az3 = 0.0_dp
    ad0 = 0.0_dp; ad1 = 0.0_dp; ad2 = 0.0_dp; ad3 = 0.0_dp
    IF (ABS(ekfs(ibnd, ik) - ef0) < fsthick) THEN
      iq1 = (ic - 1) * nchunk + 1
      iq2 = MIN(nqfs(ik), ic * nchunk)
      DO iq = iq1, iq2
        iq0 = ixqfs(ikl, iq)
        kq = ixkqf(ikl, iq0)
        DO jbnd = 1, nb
          IF (ABS(ekfs(jbnd, kq) - ef0) < fsthick) THEN
            weight = wqf(iq) * w0g(jbnd, kq) * inv_dos
            CALL syncthreads()
            DO m = tid, n2, nthr
              lam = 0.0_dp
              DO nu = 1, nm
                lam = lam + g2(ikl, iq, ibnd, jbnd, nu) * ftab(m, nu, iq0)
              ENDDO
              sh(m) = 2.0_dp * lam * dosef
            ENDDO
            DO m = tid, nsiw, nthr
              d = adp(m, jbnd, kq)
              e = weight * dwsi(m) / SQRT(wsi(m) * wsi(m) + d * d)
              sh(n2p + m) = weight * dwsi(m)
              sh(n2p + nsiw + m) = e * wsi(m)
              sh(n2p + 2 * nsiw + m) = e * d
            ENDDO
            CALL syncthreads()
            IF (i0 <= nsiw) THEN
              m0 = sh(i0);     m1 = sh(i0 + 1); m2 = sh(i0 + 2); m3 = sh(i0 + 3)
              p0 = sh(i0 + 1); p1 = sh(i0 + 2); p2 = sh(i0 + 3); p3 = sh(i0 + 4)
              DO iwp = 1, nsiw
                w1 = sh(n2p + iwp); w2 = sh(n2p + nsiw + iwp); w3 = sh(n2p + 2 * nsiw + iwp)
                km = m0 - p0; an0 = an0 + w1 * km; az0 = az0 + w2 * km; ad0 = ad0 + w3 * (m0 + p0 - mucs)
                km = m1 - p1; an1 = an1 + w1 * km; az1 = az1 + w2 * km; ad1 = ad1 + w3 * (m1 + p1 - mucs)
                km = m2 - p2; an2 = an2 + w1 * km; az2 = az2 + w2 * km; ad2 = ad2 + w3 * (m2 + p2 - mucs)
                km = m3 - p3; an3 = an3 + w1 * km; az3 = az3 + w2 * km; ad3 = ad3 + w3 * (m3 + p3 - mucs)
                m3 = m2; m2 = m1; m1 = m0; m0 = sh(ABS(i0 - iwp - 1) + 1)
                p0 = p1; p1 = p2; p2 = p3; p3 = sh(i0 + iwp + 4)
              ENDDO
            ENDIF
          ENDIF
        ENDDO
      ENDDO
    ENDIF
    IF (i0 <= nsiw) THEN
      part(i0, 1, ibnd, ikl, ic) = an0; part(i0, 2, ibnd, ikl, ic) = az0; part(i0, 3, ibnd, ikl, ic) = ad0
    ENDIF
    IF (i0 + 1 <= nsiw) THEN
      part(i0 + 1, 1, ibnd, ikl, ic) = an1; part(i0 + 1, 2, ibnd, ikl, ic) = az1
      part(i0 + 1, 3, ibnd, ikl, ic) = ad1
    ENDIF
    IF (i0 + 2 <= nsiw) THEN
      part(i0 + 2, 1, ibnd, ikl, ic) = an2; part(i0 + 2, 2, ibnd, ikl, ic) = az2
      part(i0 + 2, 3, ibnd, ikl, ic) = ad2
    ENDIF
    IF (i0 + 3 <= nsiw) THEN
      part(i0 + 3, 1, ibnd, ikl, ic) = an3; part(i0 + 3, 2, ibnd, ikl, ic) = az3
      part(i0 + 3, 3, ibnd, ikl, ic) = ad3
    ENDIF
  END SUBROUTINE k_fsr4
  !
  ATTRIBUTES(GLOBAL) SUBROUTINE k_fsr(nkl, kl0, nb, mq, nm, nsiw, n2, nchunk, &
                                      g2, wf, ixkqf, ixqfs, nqfs, ekfs, w0g, wqf, &
                                      wsi, wsn, dwsi, adp, part, ef0, fsthick, eps_ac, dosef, &
                                      pit, inv_dos, mucs)
    INTEGER, VALUE :: nkl, kl0, nb, mq, nm, nsiw, n2, nchunk
    REAL(dp), VALUE :: ef0, fsthick, eps_ac, dosef, pit, inv_dos, mucs
    REAL(dp), DEVICE :: g2(nkl, mq, nb, nb, nm), wf(nm, *), ekfs(nb, *), w0g(nb, *), wqf(*)
    INTEGER, DEVICE :: ixkqf(nkl, *), ixqfs(nkl, *), nqfs(*), wsn(nsiw)
    REAL(dp), DEVICE :: wsi(nsiw), dwsi(nsiw), adp(nsiw, nb, *), part(nsiw, 3, nb, nkl, *)
    REAL(dp), SHARED :: sh(*)
    INTEGER :: ikl, ik, ibnd, ic, iq, iq0, iq1, iq2, kq, jbnd, m, nu, iw, iwp, nw, tid, nthr
    REAL(dp) :: an, az, ad, weight, om, lam, w, d, e, lm, lp, km
    !
    ikl = blockIdx%x
    ibnd = blockIdx%y
    ic = blockIdx%z
    ik = kl0 - 1 + ikl
    tid = threadIdx%x
    nthr = blockDim%x
    iw = tid
    an = 0.0_dp; az = 0.0_dp; ad = 0.0_dp
    IF (ABS(ekfs(ibnd, ik) - ef0) < fsthick) THEN
      iq1 = (ic - 1) * nchunk + 1
      iq2 = MIN(nqfs(ik), ic * nchunk)
      nw = 0
      IF (iw <= nsiw) nw = wsn(iw)
      DO iq = iq1, iq2
        iq0 = ixqfs(ikl, iq)
        kq = ixkqf(ikl, iq0)
        DO jbnd = 1, nb
          IF (ABS(ekfs(jbnd, kq) - ef0) < fsthick) THEN
            weight = wqf(iq) * w0g(jbnd, kq) * inv_dos
            CALL syncthreads()
            ! l(m) = 2 dosef sum_nu g2 wf / (wf^2 + omega_m^2), omega_m = 2 (m-1) pi T
            DO m = tid, n2, nthr
              om = DBLE(2 * (m - 1)) * pit
              lam = 0.0_dp
              DO nu = 1, nm
                w = wf(nu, iq0)
                IF (w > eps_ac) lam = lam + g2(ikl, iq, ibnd, jbnd, nu) * w / (w * w + om * om)
              ENDDO
              sh(m) = 2.0_dp * lam * dosef
            ENDDO
            DO m = tid, nsiw, nthr
              d = adp(m, jbnd, kq)
              e = weight * dwsi(m) / SQRT(wsi(m) * wsi(m) + d * d)
              sh(n2 + m) = weight * dwsi(m)
              sh(n2 + nsiw + m) = e * wsi(m)
              sh(n2 + 2 * nsiw + m) = e * d
            ENDDO
            CALL syncthreads()
            IF (iw <= nsiw) THEN
              DO iwp = 1, nsiw
                lm = sh(ABS(nw - wsn(iwp)) + 1)
                lp = sh(ABS(nw + wsn(iwp) + 1) + 1)
                km = lm - lp
                an = an + sh(n2 + iwp) * km
                az = az + sh(n2 + nsiw + iwp) * km
                ad = ad + sh(n2 + 2 * nsiw + iwp) * (lm + lp - mucs)
              ENDDO
            ENDIF
          ENDIF
        ENDDO
      ENDDO
    ENDIF
    IF (iw <= nsiw) THEN
      part(iw, 1, ibnd, ikl, ic) = an
      part(iw, 2, ibnd, ikl, ic) = az
      part(iw, 3, ibnd, ikl, ic) = ad
    ENDIF
  END SUBROUTINE k_fsr
  !
  SUBROUTINE ege_fsr(nkl, kl0, nkfs, nb, mq, nm, nqtot, nsiw, n2, g2, wf, ixkqf, ixqfs, nqfs, &
                     ekfs, w0g, wqf, wsi, wsn, dwsi, adp, naz, az, ad, ef0, fsthick, eps_ac, &
                     dosef, pit, inv_dos, mucs)
    !! naz/az/ad(nsiw, nb, nkl): the sums of sum_eliashberg_aniso_iaxis_fsr_simple for the
    !! pool's k range kl0 .. kl0+nkl-1.
    INTEGER, INTENT(IN) :: nkl, kl0, nkfs, nb, mq, nm, nqtot, nsiw, n2
    REAL(dp), INTENT(IN) :: g2(nkl * mq * nb * nb * nm), wf(nm * nqtot), ekfs(nb * nkfs), &
                            w0g(nb * nkfs), wqf(nqtot)
    INTEGER, INTENT(IN) :: ixkqf(nkl * nqtot), ixqfs(nkl * mq), nqfs(nkfs), wsn(nsiw)
    REAL(dp), INTENT(IN) :: wsi(nsiw), dwsi(nsiw), adp(nsiw * nb * nkfs)
    REAL(dp), INTENT(OUT) :: naz(nsiw, nb, nkl), az(nsiw, nb, nkl), ad(nsiw, nb, nkl)
    REAL(dp), INTENT(IN) :: ef0, fsthick, eps_ac, dosef, pit, inv_dos, mucs
    !
    INTEGER(8) :: sz
    INTEGER :: istat, nthr, nchunk, ic, ib, ik, iw, maxq, n2p, ntile
    INTEGER :: shbytes
    LOGICAL :: uniform, lrok, lrchk
    REAL(dp), ALLOCATABLE :: chk_n(:, :, :), chk_z(:, :, :), chk_d(:, :, :)
    CHARACTER(LEN = 16) :: envval
    REAL(dp), ALLOCATABLE :: part(:, :, :, :, :)
    TYPE(dim3) :: grd, blk
    !
    lrchk = .FALSE.
    IF (nkl <= 0) RETURN
    IF (nsiw > 1024) CALL chk(1, 'nsiw > 1024 not supported')
    ! temperature-independent arrays: upload once (g2 is fixed for the whole run)
    sz = INT(nkl, 8) * mq * nb * nb * nm + INT(nm, 8) * nqtot + 7 * nkfs + kl0
    IF (sz /= sz_static) THEN
      CALL ege_reset()
      ALLOCATE(d_g2(nkl * mq * nb * nb * nm), d_wf(nm * nqtot), d_ekfs(nb * nkfs), d_w0g(nb * nkfs), &
               d_wqf(nqtot), d_ixkqf(nkl * nqtot), d_ixqfs(nkl * mq), d_nqfs(nkfs))
      d_g2 = g2; d_wf = wf; d_ekfs = ekfs; d_w0g = w0g; d_wqf = wqf
      d_ixkqf = ixkqf; d_ixqfs = ixqfs; d_nqfs = nqfs
      sz_static = sz
    ENDIF
    ! split q over blocks so that the grid fills the GPU
    maxq = MAX(1, MAXVAL(nqfs(kl0:kl0 + nkl - 1)))
    nsplit = MAX(1, MIN(maxq, (8 * 108 + nkl * nb - 1) / (nkl * nb)))
    nchunk = (maxq + nsplit - 1) / nsplit
    nsplit = (maxq + nchunk - 1) / nchunk
    ! per-temperature buffers: reallocate only when the sizes change (cudaFree synchronises)
    IF (nsiw /= nsiw_alloc .OR. nsplit /= nsplit_alloc .OR. nsiw * nb * nkfs /= nadp_alloc) THEN
      IF (ALLOCATED(d_wsi)) DEALLOCATE(d_wsi, d_dwsi, d_wsn, d_adp, d_part)
      ALLOCATE(d_wsi(nsiw), d_dwsi(nsiw), d_wsn(nsiw), d_adp(nsiw * nb * nkfs), &
               d_part(nsiw * 3 * nb * nkl * nsplit))
      nsiw_alloc = nsiw; nsplit_alloc = nsplit; nadp_alloc = nsiw * nb * nkfs
    ENDIF
    d_wsi = wsi; d_dwsi = dwsi; d_wsn = wsn; d_adp = adp
    !
    istat = cudaGetLastError()   ! clear an error left pending by earlier (non-rtsc) code
    uniform = .TRUE.
    DO iw = 1, nsiw
      IF (wsn(iw) /= iw - 1) uniform = .FALSE.
    ENDDO
    uniform = uniform .AND. (n2 == 2 * nsiw)
    ! Default: low-rank GEMM path (uniform grid). RTSC_EPW_ELIASH_KERNEL=1 selects the direct
    ! kernel k_fsr; 4 or 5 the register-window variants k_fsr4/k_fsr5.
    CALL GET_ENVIRONMENT_VARIABLE('RTSC_EPW_ELIASH_KERNEL', envval)
    IF (uniform .AND. TRIM(envval) /= '1' .AND. TRIM(envval) /= '4' .AND. TRIM(envval) /= '5') THEN
      lrok = (pit == lr_pit .AND. nsiw == lr_nsiw)
      IF (.NOT. lrok .AND. .NOT. (pit == lr_fail_pit .AND. nsiw == lr_fail_nsiw)) THEN
        CALL lr_setup(nkl, kl0, nkfs, nb, mq, nm, nqtot, nsiw, n2, wf, ef0, fsthick, eps_ac, &
                      dosef, pit, inv_dos, mucs, lrok)
        IF (.NOT. lrok) THEN
          ! remember the failure for this temperature: no new setup attempt every iteration
          lr_fail_pit = pit; lr_fail_nsiw = nsiw
          IF (kl0 == 1) WRITE(*, '(a)') ' epw_gpu_eliash: low-rank setup not possible, using the direct kernel k_fsr'
        ENDIF
      ENDIF
      IF (lrok) THEN
        CALL lr_iter(nkl, nkfs, nb, nsiw, naz, az, ad)
        CALL GET_ENVIRONMENT_VARIABLE('RTSC_EPW_ELIASH_CHECK', envval)
        IF (TRIM(envval) /= '1') RETURN
        ! check mode: also run k_fsr on the same input and report the max relative differences
        ALLOCATE(chk_n(nsiw, nb, nkl), chk_z(nsiw, nb, nkl), chk_d(nsiw, nb, nkl))
        chk_n = naz; chk_z = az; chk_d = ad
        lrchk = .TRUE.
      ENDIF
    ENDIF
    IF (TRIM(envval) /= '4' .AND. TRIM(envval) /= '5') uniform = .FALSE.
    IF (lrchk) uniform = .FALSE.
    IF (uniform) THEN
      ! kernel factor table for this temperature
      IF (pit /= ftab_pit .OR. n2 /= ftab_n2) THEN
        IF (ALLOCATED(d_ftab)) DEALLOCATE(d_ftab)
        ALLOCATE(d_ftab(INT(n2, 8) * nm * nqtot))
        CALL k_ftab<<<dim3((n2 + 127) / 128, nm, nqtot), dim3(128, 1, 1)>>>(n2, nm, nqtot, d_wf, &
             eps_ac, pit, d_ftab)
        istat = cudaGetLastError(); CALL chk(istat, 'k_ftab launch')
        ftab_pit = pit; ftab_n2 = n2
      ENDIF
      n2p = n2 + 8   ! padding: the register windows prefetch up to 4 entries past n2
      IF (nsiw <= 512 .AND. TRIM(envval) == '5') THEN
        IF (ALLOCATED(d_ez)) THEN
          IF (SIZE(d_ez) /= nsiw * nb * nkfs) DEALLOCATE(d_ez, d_ed)
        ENDIF
        IF (.NOT. ALLOCATED(d_ez)) ALLOCATE(d_ez(nsiw * nb * nkfs), d_ed(nsiw * nb * nkfs))
        CALL k_etab<<<dim3((nsiw + 127) / 128, nb * nkfs, 1), dim3(128, 1, 1)>>>(nsiw, nb * nkfs, &
             d_wsi, d_dwsi, d_adp, d_ez, d_ed)
        istat = cudaGetLastError(); CALL chk(istat, 'k_etab launch')
        ntile = (nsiw + 127) / 128
        shbytes = 8 * (4 * n2p + 12 * 128)
        CALL k_fsr5<<<dim3(nkl, nb * ntile, nsplit), dim3(128, 1, 1), shbytes>>>(nkl, kl0, nb, mq, &
             nm, nsiw, n2, n2p, nchunk, ntile, d_g2, d_ftab, d_ixkqf, d_ixqfs, d_nqfs, d_ekfs, d_w0g, &
             d_wqf, d_dwsi, d_ez, d_ed, d_part, ef0, fsthick, dosef, inv_dos, mucs)
        istat = cudaGetLastError(); CALL chk(istat, 'k_fsr5 launch')
      ELSE
        nthr = MIN(1024, 32 * (((nsiw + 3) / 4 + 31) / 32))
        shbytes = 8 * (n2p + 3 * nsiw)
        blk = dim3(nthr, 1, 1)
        grd = dim3(nkl, nb, nsplit)
        CALL k_fsr4<<<grd, blk, shbytes>>>(nkl, kl0, nb, mq, nm, nsiw, n2, n2p, nchunk, &
             d_g2, d_ftab, d_ixkqf, d_ixqfs, d_nqfs, d_ekfs, d_w0g, d_wqf, d_wsi, d_dwsi, d_adp, &
             d_part, ef0, fsthick, dosef, inv_dos, mucs)
        istat = cudaGetLastError(); CALL chk(istat, 'k_fsr4 launch')
      ENDIF
    ELSE
      nthr = MIN(1024, 32 * ((nsiw + 31) / 32))
      shbytes = 8 * (n2 + 3 * nsiw)
      blk = dim3(nthr, 1, 1)
      grd = dim3(nkl, nb, nsplit)
      CALL k_fsr<<<grd, blk, shbytes>>>(nkl, kl0, nb, mq, nm, nsiw, n2, nchunk, &
           d_g2, d_wf, d_ixkqf, d_ixqfs, d_nqfs, d_ekfs, d_w0g, d_wqf, d_wsi, d_wsn, d_dwsi, d_adp, &
           d_part, ef0, fsthick, eps_ac, dosef, pit, inv_dos, mucs)
      istat = cudaGetLastError(); CALL chk(istat, 'k_fsr launch')
    ENDIF
    ALLOCATE(part(nsiw, 3, nb, nkl, nsplit))
    istat = cudaDeviceSynchronize(); CALL chk(istat, 'k_fsr')
    istat = cudaMemcpy(part, d_part, nsiw * 3 * nb * nkl * nsplit); CALL chk(istat, 'D2H part')
    naz = 0.0_dp; az = 0.0_dp; ad = 0.0_dp
    DO ic = 1, nsplit
      DO ik = 1, nkl
        DO ib = 1, nb
          DO iw = 1, nsiw
            naz(iw, ib, ik) = naz(iw, ib, ik) + part(iw, 1, ib, ik, ic)
            az(iw, ib, ik) = az(iw, ib, ik) + part(iw, 2, ib, ik, ic)
            ad(iw, ib, ik) = ad(iw, ib, ik) + part(iw, 3, ib, ik, ic)
          ENDDO
        ENDDO
      ENDDO
    ENDDO
    DEALLOCATE(part)
    IF (lrchk) THEN
      WRITE(*, '(a, i5, 3es11.3)') ' epw_gpu_eliash check: kl0, max rel diff naz/az/ad ', kl0, &
        MAXVAL(ABS(chk_n - naz)) / MAX(MAXVAL(ABS(naz)), TINY(1.0_dp)), &
        MAXVAL(ABS(chk_z - az)) / MAX(MAXVAL(ABS(az)), TINY(1.0_dp)), &
        MAXVAL(ABS(chk_d - ad)) / MAX(MAXVAL(ABS(ad)), TINY(1.0_dp))
      naz = chk_n; az = chk_z; ad = chk_d
      DEALLOCATE(chk_n, chk_z, chk_d)
    ENDIF
  END SUBROUTINE ege_fsr
  !
END MODULE epw_gpu_eliash
