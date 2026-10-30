!
! rtsc: fused per-k projections for phq_init and dynmat_us (RTSC_PHQ_FUSE=1; off by default).
! groups/compute/NSCFROT-LOG.md.
!
! phq_init computes, per k, becp1 = <beta|psi> and alphap_i = <beta|d_i psi> (4 small device GEMMs, each with
! its own launch, sync and copy); dynmat_us then reads every psi_k again, rebuilds vkb on the host and runs 6 host
! GEMMs gamma_ij = <beta| -(k+G)_i (k+G)_j psi> before its band loop. With the switch on, phq_init builds the 10
! blocks [psi, d_x psi, d_y psi, d_z psi, d_xx psi, d_yx psi, d_yy psi, d_zx psi, d_zy psi, d_zz psi] on the GPU
! and runs ONE device GEMM vkb^H [..] per k (one copy back), fills becp1 / alphap from it, and accumulates the
! nonlocal second-derivative term of dynmat_us on the host as
!   dyn(na i, na j) += sum_n w_n sum_ih [ conj(G_ij) (D B) + conj(B) (D G_ij) + conj(A_i) (D A_j) + conj(A_j) (D A_i) ]_ih,n
! (D = deeq of atom na; the same sum as dynmat_us's ih, jh loop, reordered). dynmat_us then skips its k loop and
! adds this term. Norm-conserving, collinear, no DFT+U, no elph_mat; otherwise the vanilla path runs.
!
MODULE rtsc_phq_fuse
  USE kinds, ONLY : DP
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: phq_fuse_on, phq_fuse_begin, phq_fuse_k, phq_fuse_end, dyn_nl, dyn_nl_valid, phq_vkb_on, phq_fuse_all, &
            vkb_batch
  !
  COMPLEX(DP), ALLOCATABLE, SAVE :: dyn_nl(:,:)
  LOGICAL, SAVE :: dyn_nl_valid = .FALSE.
  COMPLEX(DP), ALLOCATABLE, SAVE :: auxa(:,:), bb(:,:)
  REAL(DP), SAVE :: t_gpu = 0, t_host = 0
  INTEGER, SAVE :: nkf = 0
  !
CONTAINS
  !
  LOGICAL FUNCTION phq_fuse_on()
    USE noncollin_module, ONLY : noncolin
    USE uspp,             ONLY : okvan, nkb
    USE ldaU,             ONLY : lda_plus_u
    USE el_phon,          ONLY : elph_mat
    USE lsda_mod,         ONLY : lsda
    USE mp_bands,         ONLY : nproc_bgrp
    CHARACTER(LEN=16) :: s
    INTEGER :: ios
    phq_fuse_on = .FALSE.
#if defined(__CUDA)
    CALL get_environment_variable('RTSC_PHQ_FUSE', s, STATUS=ios)
    IF (.NOT. (ios == 0 .AND. TRIM(s) == '1')) RETURN
    IF (noncolin .OR. okvan .OR. lda_plus_u .OR. elph_mat .OR. lsda .OR. nproc_bgrp /= 1 .OR. nkb == 0) RETURN
    phq_fuse_on = .TRUE.
#endif
  END FUNCTION phq_fuse_on
  !
  SUBROUTINE phq_fuse_begin()
    USE wvfct,     ONLY : npwx, nbnd
    USE uspp,      ONLY : nkb
    USE ions_base, ONLY : nat
    IF (ALLOCATED(dyn_nl)) DEALLOCATE(dyn_nl)
    ALLOCATE(dyn_nl(3 * nat, 3 * nat))
    dyn_nl = (0.d0, 0.d0)
    dyn_nl_valid = .FALSE.
    ALLOCATE(auxa(npwx, 10 * nbnd), bb(nkb, 10 * nbnd))
    !$acc enter data create(auxa, bb)
  END SUBROUTINE phq_fuse_begin
  !
  SUBROUTINE phq_fuse_end()
    USE io_global, ONLY : stdout
    !$acc exit data delete(auxa, bb)
    DEALLOCATE(auxa, bb)
    dyn_nl_valid = .TRUE.
    WRITE(stdout, '(5x,"rtsc phq_fuse: ",I0," k (cumulative): GPU blocks+GEMM+copy ",F8.3," s, host becp/dyn ",F8.3," s")') &
         nkf, t_gpu, t_host
  END SUBROUTINE phq_fuse_end
  !
  !------------------------------------------------------------------------------------------
  SUBROUTINE phq_fuse_k(ik, ikk, npw)
    !------------------------------------------------------------------------------------------
    !! evc (device) and vkb (device) of k-point ikk are set; fills becp1(ik), alphap(:, ik), adds to dyn_nl
    USE wvfct,         ONLY : npwx, nbnd, wg
    USE uspp,          ONLY : nkb, vkb, ofsbeta, deeq
    USE uspp_param,    ONLY : nh
    USE ions_base,     ONLY : nat, ityp
    USE wavefunctions, ONLY : evc
    USE klist,         ONLY : xk, igk_k
    USE gvect,         ONLY : g
    USE cell_base,     ONLY : tpiba, tpiba2
    USE lrus,          ONLY : becp1
    USE phus,          ONLY : alphap
    USE control_lr,    ONLY : nbnd_occ
    USE lsda_mod,      ONLY : current_spin
    INTEGER, INTENT(IN) :: ik, ikk, npw
    INTEGER :: ib, ig, itmp, ipol, nocc, na, nt, o, n, ic, jc, ijb, ih
    INTEGER, PARAMETER :: pi(6) = (/1, 2, 2, 3, 3, 3/), pj(6) = (/1, 1, 2, 1, 2, 3/)
    REAL(DP) :: k1, k2, k3, q1, q2, q3, w, tp, tp2
    INTEGER :: nb, nx
    COMPLEX(DP) :: e, s
    COMPLEX(DP), ALLOCATABLE :: x(:,:,:), y(:,:,:), dm(:,:)
    !
    INTEGER(8) :: c0, c1, c2, cr
    CALL start_clock('phq_fuse')
    CALL SYSTEM_CLOCK(c0, cr)
    k1 = xk(1, ikk); k2 = xk(2, ikk); k3 = xk(3, ikk)
    tp = tpiba; tp2 = tpiba2; nb = nbnd; nx = npwx
    !$acc parallel loop collapse(2) present(auxa, evc, igk_k, g)
    DO ib = 1, nb
      DO ig = 1, nx
        IF (ig <= npw) THEN
          itmp = igk_k(ig, ikk)
          q1 = k1 + g(1, itmp); q2 = k2 + g(2, itmp); q3 = k3 + g(3, itmp)
          e = evc(ig, ib)
          auxa(ig, ib) = e
          auxa(ig, ib + nb) = e * tp * (0.d0, 1.d0) * q1
          auxa(ig, ib + 2 * nb) = e * tp * (0.d0, 1.d0) * q2
          auxa(ig, ib + 3 * nb) = e * tp * (0.d0, 1.d0) * q3
          auxa(ig, ib + 4 * nb) = - e * tp2 * q1 * q1
          auxa(ig, ib + 5 * nb) = - e * tp2 * q2 * q1
          auxa(ig, ib + 6 * nb) = - e * tp2 * q2 * q2
          auxa(ig, ib + 7 * nb) = - e * tp2 * q3 * q1
          auxa(ig, ib + 8 * nb) = - e * tp2 * q3 * q2
          auxa(ig, ib + 9 * nb) = - e * tp2 * q3 * q3
        ELSE
          auxa(ig, ib) = 0; auxa(ig, ib + nb) = 0; auxa(ig, ib + 2 * nb) = 0; auxa(ig, ib + 3 * nb) = 0
          auxa(ig, ib + 4 * nb) = 0; auxa(ig, ib + 5 * nb) = 0; auxa(ig, ib + 6 * nb) = 0
          auxa(ig, ib + 7 * nb) = 0; auxa(ig, ib + 8 * nb) = 0; auxa(ig, ib + 9 * nb) = 0
        ENDIF
      ENDDO
    ENDDO
    !$acc host_data use_device(vkb, auxa, bb)
    CALL MYZGEMM('C', 'N', nkb, 10 * nbnd, npw, (1.0_DP, 0.0_DP), vkb, npwx, auxa, npwx, (0.0_DP, 0.0_DP), bb, nkb)
    !$acc end host_data
    !$acc update host(bb)
    CALL SYSTEM_CLOCK(c1)
    !
    CALL fuse_host(ik, ikk, bb)
    CALL SYSTEM_CLOCK(c2)
    t_gpu = t_gpu + REAL(c1 - c0, DP) / REAL(cr, DP)
    t_host = t_host + REAL(c2 - c1, DP) / REAL(cr, DP)
    nkf = nkf + 1
    CALL stop_clock('phq_fuse')
  CONTAINS
    INTEGER FUNCTION gidx(a, b)
      INTEGER, INTENT(IN) :: a, b
      INTEGER :: l
      DO l = 1, 6
        IF ((pi(l) == a .AND. pj(l) == b) .OR. (pi(l) == b .AND. pj(l) == a)) THEN
          gidx = l
          RETURN
        ENDIF
      ENDDO
      gidx = 1
    END FUNCTION gidx
  END SUBROUTINE phq_fuse_k
  !
  !
  LOGICAL FUNCTION phq_vkb_on()
    !! RTSC_PHQ_VKB=1 (with RTSC_PHQ_FUSE=1): phq_init's k loop in chunks, vkb of a whole chunk in one set of kernels
    CHARACTER(LEN=16) :: s
    INTEGER :: ios
    CALL get_environment_variable('RTSC_PHQ_VKB', s, STATUS=ios)
    phq_vkb_on = (ios == 0 .AND. TRIM(s) == '1')
  END FUNCTION phq_vkb_on
  !
  !------------------------------------------------------------------------------------------
  SUBROUTINE vkb_batch(n, ikl, vkbb)
    !------------------------------------------------------------------------------------------
    !! vkbb(:, :, j) (device) = init_us_2's vkb of k-point ikl(j), j = 1..n, with the same formulas and the same
    !! device routines (ylmr2, interp_beta), element by element, so the result is bitwise init_us_2's; the kernels
    !! run once for the n k-points instead of once per k.
    USE wvfct,      ONLY : npwx
    USE klist,      ONLY : xk, ngk, igk_k
    USE gvect,      ONLY : g, mill, eigts1, eigts2, eigts3
    USE cell_base,  ONLY : tpiba
    USE ions_base,  ONLY : nat, ityp, tau
    USE uspp,       ONLY : nkb, nhtol, nhtolm, indv
    USE uspp_param, ONLY : lmaxkb, nbetam, nhm, nh, nsp
    USE beta_mod,   ONLY : interp_beta
    USE constants,  ONLY : tpi
    INTEGER, INTENT(IN) :: n, ikl(n)
    COMPLEX(DP), INTENT(INOUT) :: vkbb(:,:,:)
    INTEGER :: nx, ntot, j, ig, f, iv, nt, na, ih, jkb, nhnt, nb, lm, ik, lmax2
    INTEGER, ALLOCATABLE :: igkb(:,:), npwb(:)
    REAL(DP), ALLOCATABLE :: xkb(:,:), argb(:,:), gk(:,:), qg(:), ylm(:,:), vq(:,:), vkb1(:,:)
    COMPLEX(DP), ALLOCATABLE :: sk(:)
    COMPLEX(DP) :: pref
    !
    IF (lmaxkb < 0) RETURN
    CALL start_clock('init_us_2')
    nx = npwx; ntot = nx * n; lmax2 = (lmaxkb + 1)**2
    ALLOCATE(igkb(nx, n), npwb(n), xkb(3, n), argb(nat, n))
    DO j = 1, n
      ik = ikl(j)
      npwb(j) = ngk(ik)
      igkb(:, j) = 1
      igkb(1:ngk(ik), j) = igk_k(1:ngk(ik), ik)
      xkb(:, j) = xk(:, ik)
      DO na = 1, nat
        argb(na, j) = (xk(1, ik) * tau(1, na) + xk(2, ik) * tau(2, na) + xk(3, ik) * tau(3, na)) * tpi
      ENDDO
    ENDDO
    ALLOCATE(gk(3, ntot), qg(ntot), ylm(ntot, lmax2), vq(ntot, nbetam), vkb1(ntot, nhm), sk(ntot))
    !$acc data copyin(igkb, npwb, xkb, argb, nhtol, nhtolm, indv) create(gk, qg, ylm, vq, vkb1, sk) &
    !$acc      present(g, mill, eigts1, eigts2, eigts3, vkbb)
    !$acc parallel loop collapse(2)
    DO j = 1, n
      DO ig = 1, nx
        f = (j - 1) * nx + ig
        IF (ig <= npwb(j)) THEN
          iv = igkb(ig, j)
          gk(1, f) = xkb(1, j) + g(1, iv)
          gk(2, f) = xkb(2, j) + g(2, iv)
          gk(3, f) = xkb(3, j) + g(3, iv)
          qg(f) = gk(1, f) * gk(1, f) + gk(2, f) * gk(2, f) + gk(3, f) * gk(3, f)
        ELSE
          gk(1, f) = 0.0_DP; gk(2, f) = 0.0_DP; gk(3, f) = 0.0_DP; qg(f) = 0.0_DP
        ENDIF
      ENDDO
    ENDDO
    CALL ylmr2(lmax2, ntot, gk, qg, ylm)
    !$acc parallel loop
    DO f = 1, ntot
      qg(f) = SQRT(qg(f)) * tpiba
    ENDDO
    !$acc parallel loop collapse(3)
    DO j = 1, n
      DO ih = 1, nkb
        DO ig = 1, nx
          vkbb(ig, ih, j) = (0.0_DP, 0.0_DP)
        ENDDO
      ENDDO
    ENDDO
    jkb = 0
    DO nt = 1, nsp
      CALL interp_beta(nt, ntot, qg, vq)
      nhnt = nh(nt)
      !$acc parallel loop collapse(2)
      DO ih = 1, nhnt
        DO f = 1, ntot
          nb = indv(ih, nt)
          lm = nhtolm(ih, nt)
          vkb1(f, ih) = ylm(f, lm) * vq(f, nb)
        ENDDO
      ENDDO
      DO na = 1, nat
        IF (ityp(na) /= nt) CYCLE
        !$acc parallel loop collapse(2)
        DO j = 1, n
          DO ig = 1, nx
            f = (j - 1) * nx + ig
            IF (ig <= npwb(j)) THEN
              iv = igkb(ig, j)
              sk(f) = eigts1(mill(1, iv), na) * eigts2(mill(2, iv), na) * eigts3(mill(3, iv), na) * &
                      CMPLX(COS(argb(na, j)), -SIN(argb(na, j)), KIND=DP)
            ENDIF
          ENDDO
        ENDDO
        !$acc parallel loop collapse(3)
        DO j = 1, n
          DO ih = 1, nhnt
            DO ig = 1, nx
              IF (ig <= npwb(j)) THEN
                f = (j - 1) * nx + ig
                pref = (0.d0, -1.d0)**nhtol(ih, nt)
                vkbb(ig, jkb + ih, j) = vkb1(f, ih) * sk(f) * pref
              ENDIF
            ENDDO
          ENDDO
        ENDDO
        jkb = jkb + nhnt
      ENDDO
    ENDDO
    !$acc end data
    DEALLOCATE(igkb, npwb, xkb, argb, gk, qg, ylm, vq, vkb1, sk)
    CALL stop_clock('init_us_2')
  END SUBROUTINE vkb_batch
  !
  !------------------------------------------------------------------------------------------
  SUBROUTINE phq_fuse_all()
    !------------------------------------------------------------------------------------------
    !! RTSC_PHQ_FUSE=1 + RTSC_PHQ_VKB=1: phq_init's whole k loop, in chunks of k-points: wavefunctions copied to the
    !! GPU once per chunk, vkb by vkb_batch, the 10-block build for the chunk in one kernel, one GEMM per k without
    !! a host sync in between, one copy back per chunk; then becp1 / alphap / dynmat_us's term on the host.
    USE wvfct,         ONLY : npwx, nbnd
    USE uspp,          ONLY : nkb, vkb
    USE uspp_param,    ONLY : lmaxkb, nbetam, nhm
    USE wavefunctions, ONLY : evc
    USE klist,         ONLY : xk, ngk, igk_k
    USE gvect,         ONLY : g
    USE cell_base,     ONLY : tpiba, tpiba2
    USE qpoint,        ONLY : xq, nksq, ikks, ikqs
    USE control_lr,    ONLY : lgamma
    USE eqv,           ONLY : evq
    USE units_lr,      ONLY : lrwfc, iuwfc
    USE buffers,       ONLY : get_buffer
    USE constants,     ONLY : eps8
    USE uspp_init,     ONLY : init_us_2
    INTEGER :: nkc, k0, n, j, ik, ikk, ikq, ib, ig, itmp, nb, nx, npw
    INTEGER(8) :: memk
    REAL(DP) :: q1, q2, q3, tp, tp2
    COMPLEX(DP) :: e
    COMPLEX(DP), ALLOCATABLE :: vkbb(:,:,:), evcb(:,:,:), auxb(:,:,:), bbb(:,:,:)
    INTEGER, ALLOCATABLE :: npwc(:), ikkc(:)
    INTEGER(8) :: c0, c1, c2, cr
    !
    CALL start_clock('phq_fuse')
    nb = nbnd; nx = npwx; tp = tpiba; tp2 = tpiba2
    ! bytes per k: vkbb, evcb, auxb, bbb and vkb_batch's work arrays
    memk = 16_8 * nx * (nkb + 11 * nbnd) + 16_8 * nkb * 10 * nbnd + &
           8_8 * nx * (5 + (lmaxkb + 1)**2 + nbetam + nhm) + 16_8 * nx
    nkc = INT(MAX(1_8, MIN(INT(nksq, 8), 1500000000_8 / memk)))
    ALLOCATE(vkbb(nx, nkb, nkc), evcb(nx, nb, nkc), auxb(nx, 10 * nb, nkc), bbb(nkb, 10 * nb, nkc))
    ALLOCATE(npwc(nkc), ikkc(nkc))
    !$acc enter data create(vkbb, evcb, auxb, bbb)
    DO k0 = 1, nksq, nkc
      CALL SYSTEM_CLOCK(c0, cr)
      n = MIN(nkc, nksq - k0 + 1)
      DO j = 1, n
        ik = k0 + j - 1
        ikk = ikks(ik); ikq = ikqs(ik)
        IF (.NOT. lgamma) THEN
          IF (ANY(ABS(xq(:) - (xk(:, ikq) - xk(:, ikk))) > eps8)) CALL errore('phq_init', 'wrong order of k points', 1)
        ENDIF
        CALL get_buffer(evc, lrwfc, iuwfc, ikk)
        evcb(:, :, j) = evc(1:nx, 1:nb)
        npwc(j) = ngk(ikk); ikkc(j) = ikk
      ENDDO
      !$acc update device(evcb(:, :, 1:n))
      CALL vkb_batch(n, ikkc(1:n), vkbb)
      !$acc parallel loop collapse(3) present(auxb, evcb, igk_k, g, xk) copyin(npwc, ikkc)
      DO j = 1, n
        DO ib = 1, nb
          DO ig = 1, nx
            IF (ig <= npwc(j)) THEN
              itmp = igk_k(ig, ikkc(j))
              q1 = xk(1, ikkc(j)) + g(1, itmp); q2 = xk(2, ikkc(j)) + g(2, itmp); q3 = xk(3, ikkc(j)) + g(3, itmp)
              e = evcb(ig, ib, j)
              auxb(ig, ib, j) = e
              auxb(ig, ib + nb, j) = e * tp * (0.d0, 1.d0) * q1
              auxb(ig, ib + 2 * nb, j) = e * tp * (0.d0, 1.d0) * q2
              auxb(ig, ib + 3 * nb, j) = e * tp * (0.d0, 1.d0) * q3
              auxb(ig, ib + 4 * nb, j) = - e * tp2 * q1 * q1
              auxb(ig, ib + 5 * nb, j) = - e * tp2 * q2 * q1
              auxb(ig, ib + 6 * nb, j) = - e * tp2 * q2 * q2
              auxb(ig, ib + 7 * nb, j) = - e * tp2 * q3 * q1
              auxb(ig, ib + 8 * nb, j) = - e * tp2 * q3 * q2
              auxb(ig, ib + 9 * nb, j) = - e * tp2 * q3 * q3
            ELSE
              auxb(ig, ib, j) = 0; auxb(ig, ib + nb, j) = 0; auxb(ig, ib + 2 * nb, j) = 0
              auxb(ig, ib + 3 * nb, j) = 0; auxb(ig, ib + 4 * nb, j) = 0; auxb(ig, ib + 5 * nb, j) = 0
              auxb(ig, ib + 6 * nb, j) = 0; auxb(ig, ib + 7 * nb, j) = 0; auxb(ig, ib + 8 * nb, j) = 0
              auxb(ig, ib + 9 * nb, j) = 0
            ENDIF
          ENDDO
        ENDDO
      ENDDO
      !$acc host_data use_device(vkbb, auxb, bbb)
      DO j = 1, n
        CALL MYZGEMM('C', 'N', nkb, 10 * nb, npwc(j), (1.0_DP, 0.0_DP), vkbb(1, 1, j), nx, auxb(1, 1, j), nx, &
                     (0.0_DP, 0.0_DP), bbb(1, 1, j), nkb)
      ENDDO
      !$acc end host_data
      !$acc update host(bbb(:, :, 1:n))
      CALL SYSTEM_CLOCK(c1)
      DO j = 1, n
        CALL fuse_host(k0 + j - 1, ikkc(j), bbb(:, :, j))
      ENDDO
      CALL SYSTEM_CLOCK(c2)
      t_gpu = t_gpu + REAL(c1 - c0, DP) / REAL(cr, DP)
      t_host = t_host + REAL(c2 - c1, DP) / REAL(cr, DP)
      nkf = nkf + n
    ENDDO
    !$acc exit data delete(vkbb, evcb, auxb, bbb)
    DEALLOCATE(vkbb, evcb, auxb, bbb, npwc, ikkc)
    ! leave the state the per-k loop leaves: evc, vkb of the last k; evq of the last k+q
    ikk = ikks(nksq)
    npw = ngk(ikk)
    CALL init_us_2(npw, igk_k(1, ikk), xk(1, ikk), vkb, .true.)
    !$acc update device(evc)
    IF (.NOT. lgamma) CALL get_buffer(evq, lrwfc, iuwfc, ikqs(nksq))
    CALL stop_clock('phq_fuse')
  END SUBROUTINE phq_fuse_all
  !
  !------------------------------------------------------------------------------------------
  SUBROUTINE fuse_host(ik, ikk, bk)
    !------------------------------------------------------------------------------------------
    !! from bk = vkb^H [psi, d_i psi, d_ij psi] of k-point ikk: becp1(ik), alphap(:, ik), dynmat_us's term
    USE wvfct,         ONLY : nbnd, wg
    USE uspp,          ONLY : nkb, ofsbeta, deeq
    USE uspp_param,    ONLY : nh
    USE ions_base,     ONLY : nat, ityp
    USE lrus,          ONLY : becp1
    USE phus,          ONLY : alphap
    USE control_lr,    ONLY : nbnd_occ
    USE lsda_mod,      ONLY : current_spin
    INTEGER, INTENT(IN) :: ik, ikk
    COMPLEX(DP), INTENT(IN) :: bk(nkb, 10 * nbnd)
    INTEGER :: ib, ipol, nocc, na, nt, o, n, ic, jc, ijb, ih
    INTEGER, PARAMETER :: pi(6) = (/1, 2, 2, 3, 3, 3/), pj(6) = (/1, 1, 2, 1, 2, 3/)
    REAL(DP) :: w
    COMPLEX(DP) :: s
    COMPLEX(DP), ALLOCATABLE :: x(:,:,:), y(:,:,:), dm(:,:)
    becp1(ik)%k(1:nkb, 1:nbnd) = bk(1:nkb, 1:nbnd)
    DO ipol = 1, 3
      alphap(ipol, ik)%k(1:nkb, 1:nbnd) = bk(1:nkb, ipol * nbnd + 1:(ipol + 1) * nbnd)
    ENDDO
    !
    ! dynmat_us nonlocal term for this k (bands 1..nbnd_occ, weights wg)
    nocc = nbnd_occ(ikk)
    IF (nocc > 0) THEN
      DO na = 1, nat
        nt = ityp(na)
        n = nh(nt)
        IF (n == 0) CYCLE
        o = ofsbeta(na)
        ALLOCATE(x(n, nocc, 10), y(n, nocc, 10), dm(n, n))
        dm(:, :) = CMPLX(deeq(1:n, 1:n, na, current_spin), 0.d0, KIND=DP)
        DO ijb = 1, 10
          x(:, :, ijb) = bk(o + 1:o + n, (ijb - 1) * nbnd + 1:(ijb - 1) * nbnd + nocc)
          y(:, :, ijb) = MATMUL(dm, x(:, :, ijb))
        ENDDO
        DO ic = 1, 3
          DO jc = 1, 3
            ! gamma block of the symmetric pair (ic, jc)
            ijb = 4 + gidx(ic, jc)
            s = (0.d0, 0.d0)
            DO ib = 1, nocc
              w = wg(ib, ikk)
              DO ih = 1, n
                s = s + w * (CONJG(x(ih, ib, ijb)) * y(ih, ib, 1) + CONJG(x(ih, ib, 1)) * y(ih, ib, ijb) + &
                             CONJG(x(ih, ib, 1 + ic)) * y(ih, ib, 1 + jc) + CONJG(x(ih, ib, 1 + jc)) * y(ih, ib, 1 + ic))
              ENDDO
            ENDDO
            dyn_nl(3 * (na - 1) + ic, 3 * (na - 1) + jc) = dyn_nl(3 * (na - 1) + ic, 3 * (na - 1) + jc) + s
          ENDDO
        ENDDO
        DEALLOCATE(x, y, dm)
      ENDDO
    ENDIF
  CONTAINS
    INTEGER FUNCTION gidx(a, b)
      INTEGER, INTENT(IN) :: a, b
      INTEGER :: l
      DO l = 1, 6
        IF ((pi(l) == a .AND. pj(l) == b) .OR. (pi(l) == b .AND. pj(l) == a)) THEN
          gidx = l
          RETURN
        ENDIF
      ENDDO
      gidx = 1
    END FUNCTION gidx
  END SUBROUTINE fuse_host
  !
END MODULE rtsc_phq_fuse
