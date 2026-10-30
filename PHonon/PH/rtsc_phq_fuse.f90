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
  PUBLIC :: phq_fuse_on, phq_fuse_begin, phq_fuse_k, phq_fuse_end, dyn_nl, dyn_nl_valid
  !
  COMPLEX(DP), ALLOCATABLE, SAVE :: dyn_nl(:,:)
  LOGICAL, SAVE :: dyn_nl_valid = .FALSE.
  COMPLEX(DP), ALLOCATABLE, SAVE :: auxa(:,:), bb(:,:)
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
    !$acc exit data delete(auxa, bb)
    DEALLOCATE(auxa, bb)
    dyn_nl_valid = .TRUE.
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
    CALL start_clock('phq_fuse')
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
    !
    becp1(ik)%k(1:nkb, 1:nbnd) = bb(1:nkb, 1:nbnd)
    DO ipol = 1, 3
      alphap(ipol, ik)%k(1:nkb, 1:nbnd) = bb(1:nkb, ipol * nbnd + 1:(ipol + 1) * nbnd)
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
          x(:, :, ijb) = bb(o + 1:o + n, (ijb - 1) * nbnd + 1:(ijb - 1) * nbnd + nocc)
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
END MODULE rtsc_phq_fuse
