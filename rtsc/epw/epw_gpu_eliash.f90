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
    sz_static = -1
  END SUBROUTINE ege_reset
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
    INTEGER :: istat, nthr, nchunk, ic, ib, ik, iw, maxq
    INTEGER :: shbytes
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
    nthr = MIN(1024, 32 * ((nsiw + 31) / 32))
    shbytes = 8 * (n2 + 3 * nsiw)
    blk = dim3(nthr, 1, 1)
    grd = dim3(nkl, nb, nsplit)
    CALL k_fsr<<<grd, blk, shbytes>>>(nkl, kl0, nb, mq, nm, nsiw, n2, nchunk, &
         d_g2, d_wf, d_ixkqf, d_ixqfs, d_nqfs, d_ekfs, d_w0g, d_wqf, d_wsi, d_wsn, d_dwsi, d_adp, &
         d_part, ef0, fsthick, eps_ac, dosef, pit, inv_dos, mucs)
    istat = cudaGetLastError(); CALL chk(istat, 'k_fsr launch')
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
