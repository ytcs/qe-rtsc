!
! rtsc: starting wavefunctions for the ph.x nscf (c_bands_nscf_ph) by symmetry rotation of the scf
! wavefunctions in the restart directory, instead of random vectors.
!
! Enabled by RTSC_NSCF_ROT=1 (off by default). Davidson then runs exactly as in vanilla, to the
! same threshold (ethr_nscf = 1e-9/nelec), from a much better start. The start only changes how
! many Davidson iterations are needed; it cannot change what Davidson converges to beyond ethr
! (vanilla's own result already depends on its random start at that level). If no symmetry image
! is found for a k-point, or the rotated vector does not cover the k+G sphere, that k-point uses
! vanilla's init_wfc.
!
! For a target k' (crystal coordinates of bg) the code looks for a stored k, a crystal symmetry
! S (s(:,:,isym), isym <= nsym, with zero fractional translation) and tr = +1 or -1 (time reversal)
! with k' = tr S k + G0, G0 integer (the convention of kpoint_grid: k'_i = sum_j s(i,j) k_j).
! Then c_k'(tr S m - G0) = c_k(m) (tr = +1) or conj(c_k(m)) (tr = -1), m the Miller indices.
!
MODULE sth_nscf_start
  USE kinds, ONLY : DP
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: nscf_start_prepare, nscf_start_get, nscf_start_free
  !
  LOGICAL, SAVE :: active = .FALSE.
  INTEGER, SAVE :: nks0 = 0, nbnd0 = 0, ngwx0 = 0
  INTEGER, SAVE :: nfound = 0, nfallback = 0
  REAL(DP), ALLOCATABLE, SAVE :: xk0(:,:)          ! stored k, crystal coordinates
  INTEGER, ALLOCATABLE, SAVE :: ngw0(:), mill0(:,:,:)
  COMPLEX(DP), ALLOCATABLE, SAVE :: wfc0(:,:,:)
  !
CONTAINS
  !
  !------------------------------------------------------------------------------------------
  SUBROUTINE nscf_start_prepare()
    !------------------------------------------------------------------------------------------
    USE io_global,        ONLY : stdout
    USE io_files,         ONLY : restart_dir
    USE cell_base,        ONLY : at
    USE wvfct,            ONLY : nbnd
    USE noncollin_module, ONLY : noncolin
    USE control_flags,    ONLY : gamma_only
    USE mp_bands,         ONLY : nproc_bgrp
    USE lsda_mod,         ONLY : lsda
    CHARACTER(LEN=320) :: fname
    CHARACTER(LEN=16) :: s
    INTEGER :: ios, u, i, ik_, ispin, npol, ngw, igwx, nb, ib, pass
    LOGICAL :: gam, exst
    REAL(DP) :: xk(3), scalef, b(3, 3)
    COMPLEX(DP), ALLOCATABLE :: w(:)
    !
    CALL nscf_start_free()
    CALL get_environment_variable('RTSC_NSCF_ROT', s, STATUS=ios)
    IF (ios /= 0 .OR. TRIM(s) /= '1') RETURN
    IF (noncolin .OR. gamma_only .OR. lsda .OR. nproc_bgrp /= 1) RETURN
    CALL start_clock('nscf_rot')
    !
    ! pass 1: count files and sizes; pass 2: read
    DO pass = 1, 2
      i = 0
      DO
        WRITE(fname, '(A,"wfc",I0,".dat")') TRIM(restart_dir()), i + 1
        INQUIRE(FILE=TRIM(fname), EXIST=exst)
        IF (.NOT. exst) EXIT
        OPEN(NEWUNIT=u, FILE=TRIM(fname), FORM='unformatted', STATUS='old', ACTION='read', IOSTAT=ios)
        IF (ios /= 0) EXIT
        READ(u, IOSTAT=ios) ik_, xk, ispin, gam, scalef
        IF (ios == 0) READ(u, IOSTAT=ios) ngw, igwx, npol, nb
        IF (ios /= 0 .OR. npol /= 1 .OR. gam) THEN
          CLOSE(u)
          EXIT
        ENDIF
        i = i + 1
        IF (pass == 1) THEN
          ngwx0 = MAX(ngwx0, igwx)
          nbnd0 = MAX(nbnd0, nb)
        ELSE
          READ(u) b
          READ(u) mill0(1:3, 1:igwx, i)
          ALLOCATE(w(igwx))
          DO ib = 1, MIN(nb, nbnd)
            READ(u) w
            wfc0(1:igwx, ib, i) = w
          ENDDO
          DEALLOCATE(w)
          ngw0(i) = igwx
          ! cartesian (2 pi / alat) -> crystal coordinates of bg
          xk0(:, i) = MATMUL(TRANSPOSE(at), xk)
        ENDIF
        CLOSE(u)
      ENDDO
      IF (pass == 1) THEN
        nks0 = i
        IF (nks0 == 0 .OR. nbnd0 < nbnd) THEN
          nks0 = 0
          CALL stop_clock('nscf_rot')
          RETURN
        ENDIF
        ALLOCATE(xk0(3, nks0), ngw0(nks0), mill0(3, ngwx0, nks0), wfc0(ngwx0, nbnd, nks0))
        wfc0 = (0.d0, 0.d0)
      ENDIF
    ENDDO
    active = .TRUE.
    nfound = 0; nfallback = 0
    WRITE(stdout, '(5x,"rtsc nscf: rotated starting wavefunctions from ",I0," stored k-points")') nks0
    CALL stop_clock('nscf_rot')
  END SUBROUTINE nscf_start_prepare
  !
  !------------------------------------------------------------------------------------------
  LOGICAL FUNCTION nscf_start_get(ik, evc)
    !------------------------------------------------------------------------------------------
    !! fills evc(npwx, nbnd) with the rotated start for k-point ik of the current list
    USE cell_base,        ONLY : at
    USE klist,            ONLY : xk, ngk, igk_k
    USE wvfct,            ONLY : nbnd, npwx
    USE fft_base,         ONLY : dfftp
    USE symm_base,        ONLY : s, ft, nsym
    INTEGER, INTENT(IN) :: ik
    COMPLEX(DP), INTENT(OUT) :: evc(npwx, nbnd)
    REAL(DP) :: xkc(3), kr(3), d(3)
    INTEGER :: i0, isym, itr, tr, g0(3), m(3), mr(3), ig, n1, n2, n3, ir, nmap, j, ibest, isbest, trbest
    INTEGER :: g0best(3)
    INTEGER, ALLOCATABLE :: pos(:)
    LOGICAL :: found
    !
    nscf_start_get = .FALSE.
    IF (.NOT. active) RETURN
    CALL start_clock('nscf_rot')
    xkc = MATMUL(TRANSPOSE(at), xk(:, ik))
    found = .FALSE.
    search: DO i0 = 1, nks0
      DO isym = 1, nsym
        IF (ANY(ABS(ft(:, isym)) > 1.d-8)) CYCLE
        DO itr = 1, 2
          tr = 3 - 2 * itr
          DO j = 1, 3
            kr(j) = tr * (s(j, 1, isym) * xk0(1, i0) + s(j, 2, isym) * xk0(2, i0) + s(j, 3, isym) * xk0(3, i0))
          ENDDO
          d = xkc - kr
          IF (ALL(ABS(d - NINT(d)) < 1.d-5)) THEN
            found = .TRUE.
            ibest = i0; isbest = isym; trbest = tr; g0best = NINT(d)
            EXIT search
          ENDIF
        ENDDO
      ENDDO
    ENDDO search
    IF (.NOT. found) THEN
      nfallback = nfallback + 1
      CALL stop_clock('nscf_rot')
      RETURN
    ENDIF
    !
    ! position of each k'+G of the target list on the (dense) FFT grid
    ALLOCATE(pos(dfftp%nr1 * dfftp%nr2 * dfftp%nr3))
    pos = 0
    DO ig = 1, ngk(ik)
      ! Miller indices of target G from the gvect list
      CALL mill_of(igk_k(ig, ik), m)
      pos(lin(m)) = ig
    ENDDO
    evc = (0.d0, 0.d0)
    nmap = 0
    DO ig = 1, ngw0(ibest)
      m = mill0(:, ig, ibest)
      DO j = 1, 3
        mr(j) = trbest * (s(j, 1, isbest) * m(1) + s(j, 2, isbest) * m(2) + s(j, 3, isbest) * m(3)) - g0best(j)
      ENDDO
      ir = pos(lin(mr))
      IF (ir == 0) CYCLE
      nmap = nmap + 1
      IF (trbest == 1) THEN
        evc(ir, 1:nbnd) = wfc0(ig, 1:nbnd, ibest)
      ELSE
        evc(ir, 1:nbnd) = CONJG(wfc0(ig, 1:nbnd, ibest))
      ENDIF
    ENDDO
    DEALLOCATE(pos)
    ! the k+G sphere maps onto itself; accept only (near-)complete coverage
    IF (nmap >= ngk(ik) - MAX(2, ngk(ik) / 1000) .AND. nmap <= ngk(ik)) THEN
      nscf_start_get = .TRUE.
      nfound = nfound + 1
    ELSE
      nfallback = nfallback + 1
    ENDIF
    CALL stop_clock('nscf_rot')
  CONTAINS
    INTEGER FUNCTION lin(mm)
      INTEGER, INTENT(IN) :: mm(3)
      n1 = MODULO(mm(1), dfftp%nr1); n2 = MODULO(mm(2), dfftp%nr2); n3 = MODULO(mm(3), dfftp%nr3)
      lin = 1 + n1 + n2 * dfftp%nr1 + n3 * dfftp%nr1 * dfftp%nr2
    END FUNCTION lin
  END FUNCTION nscf_start_get
  !
  SUBROUTINE mill_of(ig, m)
    USE gvect, ONLY : mill
    INTEGER, INTENT(IN) :: ig
    INTEGER, INTENT(OUT) :: m(3)
    m = mill(:, ig)
  END SUBROUTINE mill_of
  !
  SUBROUTINE nscf_start_free()
    USE io_global, ONLY : stdout
    IF (active) WRITE(stdout, '(5x,"rtsc nscf: rotated starts ",I0,", vanilla starts ",I0)') nfound, nfallback
    active = .FALSE.
    nks0 = 0; nbnd0 = 0; ngwx0 = 0
    IF (ALLOCATED(xk0)) DEALLOCATE(xk0, ngw0, mill0, wfc0)
  END SUBROUTINE nscf_start_free
  !
END MODULE sth_nscf_start
