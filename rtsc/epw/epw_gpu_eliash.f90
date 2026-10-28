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
  !
  PUBLIC :: ege_fsr, ege_reset
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
  END SUBROUTINE ege_reset
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
    LOGICAL :: uniform
    CHARACTER(LEN = 16) :: envval
    REAL(dp), ALLOCATABLE :: part(:, :, :, :, :)
    TYPE(dim3) :: grd, blk
    !
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
    IF (ALLOCATED(d_wsi)) DEALLOCATE(d_wsi, d_dwsi, d_wsn, d_adp, d_part)
    ALLOCATE(d_wsi(nsiw), d_dwsi(nsiw), d_wsn(nsiw), d_adp(nsiw * nb * nkfs), &
             d_part(nsiw * 3 * nb * nkl * nsplit))
    d_wsi = wsi; d_dwsi = dwsi; d_wsn = wsn; d_adp = adp
    !
    istat = cudaGetLastError()   ! clear an error left pending by earlier (non-rtsc) code
    uniform = .TRUE.
    DO iw = 1, nsiw
      IF (wsn(iw) /= iw - 1) uniform = .FALSE.
    ENDDO
    uniform = uniform .AND. (n2 == 2 * nsiw)
    CALL GET_ENVIRONMENT_VARIABLE('RTSC_EPW_ELIASH_K4', envval)
    IF (TRIM(envval) == '0') uniform = .FALSE.   ! force the reference kernel k_fsr
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
      CALL GET_ENVIRONMENT_VARIABLE('RTSC_EPW_ELIASH_K5', envval)
      IF (nsiw <= 512 .AND. TRIM(envval) /= '0') THEN
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
  END SUBROUTINE ege_fsr
  !
END MODULE epw_gpu_eliash
