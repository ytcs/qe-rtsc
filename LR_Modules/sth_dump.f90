!
! rtsc: offline dump of one sternheimer_kernel call (inputs and outputs), for developing and
! testing the batched Sternheimer engine without ph.x in the loop.
!
! Enabled by environment variables (off by default; no effect on normal runs):
!   RTSC_STH_DUMP=<dir>        directory to write into (must exist)
!   RTSC_STH_DUMP_CALL=<n>     which sternheimer_kernel call to dump (default 2: not first_iter)
! Only the serial, collinear, norm-conserving case with one pool is dumped; anything else is skipped
! with a message. Layout: <dir>/meta.txt (key value), and raw stream-unformatted arrays:
!   global: vrs.bin dmat.bin dvscf.bin drho_in.bin drho_out.bin
!   per k (kNNNN_): igk_k.bin igk_kq.bin evc.bin evq.bin et_k.bin et_kq.bin g2kin.bin vkb.bin
!                   hdiag.bin, and per pert (kNNNN_pPP_): dvbare.bin dpsi_in.bin dpsi_out.bin
! igk_* hold FFT linear indices dffts%nl(igk_k(:,ik)), 1-based, length npw / npwq.
!
MODULE sth_dump
  USE kinds, ONLY : DP
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: sth_dump_begin, sth_dump_end
  INTEGER, SAVE :: ncall = 0
  LOGICAL, SAVE :: active = .FALSE.
  CHARACTER(LEN=256), SAVE :: dir = ''
CONTAINS
  !
  SUBROUTINE wbin(fname, n, a)
    CHARACTER(LEN=*), INTENT(IN) :: fname
    INTEGER, INTENT(IN) :: n
    COMPLEX(DP), INTENT(IN) :: a(n)
    INTEGER :: u
    OPEN(NEWUNIT=u, FILE=TRIM(dir)//'/'//fname, ACCESS='stream', FORM='unformatted', STATUS='replace')
    WRITE(u) a
    CLOSE(u)
  END SUBROUTINE wbin
  !
  SUBROUTINE wbinr(fname, n, a)
    CHARACTER(LEN=*), INTENT(IN) :: fname
    INTEGER, INTENT(IN) :: n
    REAL(DP), INTENT(IN) :: a(n)
    INTEGER :: u
    OPEN(NEWUNIT=u, FILE=TRIM(dir)//'/'//fname, ACCESS='stream', FORM='unformatted', STATUS='replace')
    WRITE(u) a
    CLOSE(u)
  END SUBROUTINE wbinr
  !
  SUBROUTINE wbini(fname, n, a)
    CHARACTER(LEN=*), INTENT(IN) :: fname
    INTEGER, INTENT(IN) :: n
    INTEGER, INTENT(IN) :: a(n)
    INTEGER :: u
    OPEN(NEWUNIT=u, FILE=TRIM(dir)//'/'//fname, ACCESS='stream', FORM='unformatted', STATUS='replace')
    WRITE(u) a
    CLOSE(u)
  END SUBROUTINE wbini
  !
  CHARACTER(LEN=16) FUNCTION ktag(ik, ipert)
    INTEGER, INTENT(IN) :: ik, ipert
    IF (ipert > 0) THEN
      WRITE(ktag, '("k",I4.4,"_p",I2.2,"_")') ik, ipert
    ELSE
      WRITE(ktag, '("k",I4.4,"_")') ik
    ENDIF
  END FUNCTION ktag
  !
  !--------------------------------------------------------------------------
  SUBROUTINE sth_dump_begin(first_iter, npert, lrdvpsi, iudvpsi, thresh, dvscfins, drhoout)
    !--------------------------------------------------------------------------
    USE io_global,        ONLY : stdout
    USE fft_base,         ONLY : dffts
    USE ions_base,        ONLY : nat, ityp
    USE klist,            ONLY : xk, wk, ngk, igk_k, lgauss, degauss, ngauss
    USE ener,             ONLY : ef
    USE cell_base,        ONLY : omega, tpiba2
    USE wvfct,            ONLY : nbnd, npwx, et, g2kin
    USE wavefunctions,    ONLY : evc
    USE noncollin_module, ONLY : noncolin, nspin_mag
    USE lsda_mod,         ONLY : lsda
    USE uspp,             ONLY : vkb, nkb, deeq, ofsbeta, okvan
    USE uspp_param,       ONLY : nh
    USE uspp_init,        ONLY : init_us_2
    USE scf,              ONLY : vrs
    USE buffers,          ONLY : get_buffer
    USE units_lr,         ONLY : iuwfc, lrwfc, lrdwf, iudwf
    USE control_lr,       ONLY : nbnd_occ, lgamma, alpha_pv
    USE qpoint,           ONLY : nksq, ikks, ikqs
    USE eqv,              ONLY : dpsi, dvpsi, evq
    USE mp_pools,         ONLY : npool
    USE mp_bands,         ONLY : nproc_bgrp
    LOGICAL, INTENT(IN) :: first_iter
    INTEGER, INTENT(IN) :: npert, lrdvpsi, iudvpsi
    REAL(DP), INTENT(IN) :: thresh
    COMPLEX(DP), INTENT(IN) :: dvscfins(dffts%nnr, nspin_mag, npert)
    COMPLEX(DP), INTENT(IN) :: drhoout(dffts%nnr, nspin_mag, npert)
    !
    INTEGER :: want, ios, ik, ikk, ikq, npw, npwq, ipert, nrec, na, ih, jh, u
    INTEGER, ALLOCATABLE :: idx(:)
    REAL(DP), ALLOCATABLE :: h_diag(:,:)
    COMPLEX(DP), ALLOCATABLE :: dmat(:,:)
    CHARACTER(LEN=32) :: s
    !
    ncall = ncall + 1
    active = .FALSE.
    CALL get_environment_variable('RTSC_STH_DUMP', dir, STATUS=ios)
    IF (ios /= 0 .OR. LEN_TRIM(dir) == 0) RETURN
    want = 2
    CALL get_environment_variable('RTSC_STH_DUMP_CALL', s, STATUS=ios)
    IF (ios == 0) READ(s, *) want
    IF (ncall /= want) RETURN
    IF (noncolin .OR. lsda .OR. okvan .OR. npool > 1 .OR. nproc_bgrp > 1 .OR. nspin_mag /= 1) THEN
      WRITE(stdout, '(5x,"rtsc sth_dump: unsupported case (needs serial, collinear, NC); skipped")')
      RETURN
    ENDIF
    active = .TRUE.
    WRITE(stdout, '(5x,"rtsc sth_dump: dumping call ",I4," to ",A)') ncall, TRIM(dir)
    !
    OPEN(NEWUNIT=u, FILE=TRIM(dir)//'/meta.txt', STATUS='replace')
    WRITE(u,*) 'nksq ', nksq
    WRITE(u,*) 'npwx ', npwx
    WRITE(u,*) 'nbnd ', nbnd
    WRITE(u,*) 'nkb ', nkb
    WRITE(u,*) 'npert ', npert
    WRITE(u,*) 'nnr ', dffts%nnr
    WRITE(u,*) 'nr1 ', dffts%nr1, ' nr2 ', dffts%nr2, ' nr3 ', dffts%nr3
    WRITE(u,*) 'nr1x ', dffts%nr1x, ' nr2x ', dffts%nr2x, ' nr3x ', dffts%nr3x
    WRITE(u,*) 'first_iter ', first_iter
    WRITE(u,'(A,ES24.16)') ' thresh ', thresh
    WRITE(u,'(A,ES24.16)') ' alpha_pv ', alpha_pv
    WRITE(u,'(A,ES24.16)') ' ef ', ef
    WRITE(u,'(A,ES24.16)') ' degauss ', degauss
    WRITE(u,*) 'ngauss ', ngauss
    WRITE(u,*) 'lgauss ', lgauss
    WRITE(u,*) 'lgamma ', lgamma
    WRITE(u,'(A,ES24.16)') ' omega ', omega
    WRITE(u,'(A,ES24.16)') ' tpiba2 ', tpiba2
    DO ik = 1, nksq
      ikk = ikks(ik); ikq = ikqs(ik)
      WRITE(u,'(A,I6,A,I8,A,I8,A,I4,A,I4,A,ES24.16)') ' k ', ik, ' npw ', ngk(ikk), ' npwq ', ngk(ikq), &
        ' nocc_k ', nbnd_occ(ikk), ' nocc_kq ', nbnd_occ(ikq), ' wk ', wk(ikk)
    ENDDO
    CLOSE(u)
    !
    ! dense nkb x nkb D matrix (block diagonal per atom), as used by add_vuspsi
    ALLOCATE(dmat(nkb, nkb))
    dmat = (0.d0, 0.d0)
    DO na = 1, nat
      DO ih = 1, nh(ityp(na))
        DO jh = 1, nh(ityp(na))
          dmat(ofsbeta(na)+ih, ofsbeta(na)+jh) = CMPLX(deeq(ih, jh, na, 1), 0.d0, KIND=DP)
        ENDDO
      ENDDO
    ENDDO
    CALL wbin('dmat.bin', nkb*nkb, dmat)
    DEALLOCATE(dmat)
    CALL wbinr('vrs.bin', dffts%nnr, vrs(:,1))
    CALL wbin('dvscf.bin', dffts%nnr*npert, dvscfins)
    CALL wbin('drho_in.bin', dffts%nnr*npert, drhoout)
    !
    ALLOCATE(idx(npwx), h_diag(npwx, nbnd))
    DO ik = 1, nksq
      ikk = ikks(ik); ikq = ikqs(ik)
      npw = ngk(ikk); npwq = ngk(ikq)
      idx(1:npw) = dffts%nl(igk_k(1:npw, ikk))
      CALL wbini(TRIM(ktag(ik,0))//'igk_k.bin', npw, idx)
      idx(1:npwq) = dffts%nl(igk_k(1:npwq, ikq))
      CALL wbini(TRIM(ktag(ik,0))//'igk_kq.bin', npwq, idx)
      CALL get_buffer(evc, lrwfc, iuwfc, ikk)
      CALL wbin(TRIM(ktag(ik,0))//'evc.bin', npwx*nbnd, evc)
      IF (.NOT. lgamma) CALL get_buffer(evq, lrwfc, iuwfc, ikq)
      CALL wbin(TRIM(ktag(ik,0))//'evq.bin', npwx*nbnd, evq)
      CALL wbinr(TRIM(ktag(ik,0))//'et_k.bin', nbnd, et(:, ikk))
      CALL wbinr(TRIM(ktag(ik,0))//'et_kq.bin', nbnd, et(:, ikq))
      CALL g2_kin(ikq)
      CALL wbinr(TRIM(ktag(ik,0))//'g2kin.bin', npwx, g2kin)
      CALL init_us_2(npwq, igk_k(1, ikq), xk(1, ikq), vkb, .true.)
      !$acc update host(vkb)
      CALL wbin(TRIM(ktag(ik,0))//'vkb.bin', npwx*nkb, vkb)
      CALL h_prec(ik, evq, h_diag)
      CALL wbinr(TRIM(ktag(ik,0))//'hdiag.bin', npwx*nbnd, h_diag)
      DO ipert = 1, npert
        nrec = (ipert - 1) * nksq + ik
        CALL get_buffer(dvpsi, lrdvpsi, iudvpsi, nrec)
        CALL wbin(TRIM(ktag(ik,ipert))//'dvbare.bin', npwx*nbnd, dvpsi)
        IF (.NOT. first_iter) THEN
          CALL get_buffer(dpsi, lrdwf, iudwf, nrec)
          CALL wbin(TRIM(ktag(ik,ipert))//'dpsi_in.bin', npwx*nbnd, dpsi)
        ENDIF
      ENDDO
    ENDDO
    DEALLOCATE(idx, h_diag)
  END SUBROUTINE sth_dump_begin
  !
  !--------------------------------------------------------------------------
  SUBROUTINE sth_dump_end(npert, drhoout, tot_num_iter, tot_cg_calls)
    !--------------------------------------------------------------------------
    USE fft_base,         ONLY : dffts
    USE noncollin_module, ONLY : nspin_mag
    USE buffers,          ONLY : get_buffer
    USE units_lr,         ONLY : lrdwf, iudwf
    USE qpoint,           ONLY : nksq
    USE eqv,              ONLY : dpsi
    USE wvfct,            ONLY : nbnd, npwx
    INTEGER, INTENT(IN) :: npert, tot_num_iter, tot_cg_calls
    COMPLEX(DP), INTENT(IN) :: drhoout(dffts%nnr, nspin_mag, npert)
    INTEGER :: ik, ipert, nrec, u
    IF (.NOT. active) RETURN
    CALL wbin('drho_out.bin', dffts%nnr*npert, drhoout)
    DO ik = 1, nksq
      DO ipert = 1, npert
        nrec = (ipert - 1) * nksq + ik
        CALL get_buffer(dpsi, lrdwf, iudwf, nrec)
        CALL wbin(TRIM(ktag(ik,ipert))//'dpsi_out.bin', npwx*nbnd, dpsi)
      ENDDO
    ENDDO
    OPEN(NEWUNIT=u, FILE=TRIM(dir)//'/meta.txt', POSITION='append')
    WRITE(u,*) 'tot_num_iter ', tot_num_iter
    WRITE(u,*) 'tot_cg_calls ', tot_cg_calls
    CLOSE(u)
    active = .FALSE.
  END SUBROUTINE sth_dump_end
END MODULE sth_dump
