!
! rtsc: exact symmetry rotation instead of the per-q nscf eigensolve (RTSC_NSCF_SYMROT=1; off by default).
! groups/compute/NSCFROT-LOG.md.
!
! ph.x runs an nscf for every q on the k list of the small group of q plus the k+q. On a commensurate grid
! every k of that list is an image, under the full crystal group (including fractional translations) and
! time reversal, of a k-point whose eigenpairs are already known. This module keeps a cache of converged
! eigenpairs (one representative per full-group orbit, filled lazily, kept for the whole ph.x process) and
! obtains every other k of the list by an exact rotation in G space:
!
!   k' = tr S k + G0 (crystal coordinates of bg, kpoint_grid's convention k'_i = sum_j s(i,j) k_j),
!   c_k'(tr S m - G0) = c_k(m) exp(-2 pi i (k + m).ft)              (tr = +1),
!   c_k'(tr S m - G0) = conj(c_k(m)) exp(+2 pi i (k + m).ft)         (tr = -1, time reversal),
!
! with m the Miller index and ft the fractional translation of operation S (crystal axis; QE's op is
! x -> s^T x - ft, and k -> s k is its inverse, whence the phase uses the source k + m).
! The eigenvalues are copied (et(:, k') = e(k)); the occupations follow from them in the usual way.
! Degenerate subspaces: the rotated vectors are an orthonormal basis of the same eigenspace; DFPT does not
! depend on that gauge. Band count: all nbnd bands of the representative are rotated.
!
! Representatives not yet in the cache are diagonalized by the existing nscf path (batched Davidson or the
! per-k loop), at ethr * RTSC_NSCF_SYMROT_TIGHT (default 0.01), then stored. A k-point is rotated only if the
! k+G sphere maps onto itself exactly (same count, every G found); otherwise it is diagonalized itself.
!
! RTSC_NSCF_SYMROT_CHECK=1 (diagnostic): residuals |H psi - e psi| of every rotated and every Davidson k are
! computed with h_psi and their maxima printed (exactness check of the rotation, phases and time reversal).
!
MODULE sth_nscf_symrot
  USE kinds, ONLY : DP
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: symrot_on, symrot_plan, symrot_after_solve, symrot_report, symrot_tight, symrot_check_on, symrot_resid
  ! rtsc epw3 (RTSC_PW_SYMROT=1; groups/compute/EPW3-LOG.md): the same rotation in pw.x's nscf (c_bands_nscf), the
  ! Wannier k grid read by EPW. The module lives in PW/src so that both pw.x and ph.x use it.
  PUBLIC :: symrot_pw_on, symrot_pw_tight, symrot_pw_check_on, symrot_note_notconv, symrot_rep_guard
  INTEGER, SAVE :: last_notconv = 0, nbadrep = 0
  !
  ! cache of representatives (persists across q)
  INTEGER, SAVE :: nc = 0, ncmax = 0, npwc = 0, nbc = 0
  REAL(DP), ALLOCATABLE, SAVE :: cxk(:,:), ce(:,:)
  INTEGER, ALLOCATABLE, SAVE :: cngw(:), cmill(:,:,:)
  COMPLEX(DP), ALLOCATABLE, SAVE :: cwfc(:,:,:)
  ! plan for the current k list
  INTEGER, ALLOCATABLE, SAVE :: kind_(:), ksrc(:), ksym(:), ktr(:), kg0(:,:)
  ! kind_: 0 = Davidson (not a rep: fallback), 1 = rotated from cache, 2 = rep (Davidson, then cached),
  !        3 = rotated from a rep of this list after its solve
  INTEGER, SAVE :: nrot_c = 0, nrot_l = 0, nrep = 0, nfb = 0
  INTEGER, SAVE :: tot_rot = 0, tot_dav = 0
  REAL(DP), SAVE :: rmax_rot = 0, rmax_dav = 0
  INTEGER, ALLOCATABLE, SAVE :: pos(:)
  INTEGER, SAVE :: npos = 0
  ! hash tables of orbit images (k mod G on a 720720-grid): tc for the cache, tl for this list's reps
  INTEGER(8), PARAMETER :: MGRID = 720720_8
  TYPE htab
    INTEGER(8), ALLOCATABLE :: key(:)
    INTEGER, ALLOCATABLE :: val(:,:)    ! (id, isym, tr)
    INTEGER :: n = 0, cap = 0
  END TYPE htab
  TYPE(htab), SAVE :: tc, tl
  REAL(DP), SAVE :: t_plan = 0, t_after = 0
  LOGICAL, SAVE :: prefer_ft = .FALSE., rev_ops = .FALSE.
  ! the full crystal group, captured once: ph.x reorders symm_base's s/ft at every q (copy_sym puts the small group
  ! of q first), so op indices stored in the cache must refer to this fixed copy
  INTEGER, SAVE :: sr(3, 3, 48), nsr = 0
  REAL(DP), SAVE :: ftr(3, 48)
  INTEGER, SAVE :: nft = 0     ! rotations that used an op with a fractional translation (cumulative)
  !
CONTAINS
  !
  LOGICAL FUNCTION symrot_on()
    USE noncollin_module, ONLY : noncolin
    USE control_flags,    ONLY : gamma_only
    USE lsda_mod,         ONLY : lsda
    USE mp_bands,         ONLY : nproc_bgrp
    USE mp_pools,         ONLY : npool
    USE ldaU,             ONLY : lda_plus_u
    USE uspp,             ONLY : okvan
    CHARACTER(LEN=16) :: s
    INTEGER :: ios
    symrot_on = .FALSE.
    CALL get_environment_variable('RTSC_NSCF_SYMROT', s, STATUS=ios)
    IF (.NOT. (ios == 0 .AND. TRIM(s) == '1')) RETURN
    ! single pool / band group: the cache must see every k of the list
    IF (noncolin .OR. gamma_only .OR. lsda .OR. nproc_bgrp /= 1 .OR. npool /= 1 .OR. lda_plus_u) RETURN
    IF (okvan) RETURN   ! NC only (the overlap S = 1 is assumed in the residual check; USPP untested)
    symrot_on = .TRUE.
  END FUNCTION symrot_on
  !
  SUBROUTINE symrot_note_notconv(n)
    !! diag_bands reports its count of unconverged eigenvalues (stored only; no effect on the eigensolve)
    INTEGER, INTENT(IN) :: n
    last_notconv = n
  END SUBROUTINE symrot_note_notconv
  !
  LOGICAL FUNCTION symrot_rep_guard(ik)
    !! RTSC_PW_SYMROT: a representative whose eigensolve left unconverged eigenvalues is not a rotation source;
    !! the k-points waiting for it are diagonalized themselves (as the switch-off build does for every k).
    !! .TRUE. = ik was such a representative (the caller re-diagonalizes it at pw.x's own ethr)
    USE io_global, ONLY : stdout
    INTEGER, INTENT(IN) :: ik
    symrot_rep_guard = .FALSE.
    IF (last_notconv > 0 .AND. ALLOCATED(kind_)) THEN
      IF (kind_(ik) == 2) THEN
        kind_(ik) = 0
        nbadrep = nbadrep + 1
        symrot_rep_guard = .TRUE.
        WRITE(stdout, '(5x,"rtsc nscf symrot: representative ",I0," has ",I0, &
             &" unconverged eigenvalues: its orbit is diagonalized instead (",I0," such reps)")') ik, last_notconv, nbadrep
      ENDIF
    ENDIF
    last_notconv = 0
  END FUNCTION symrot_rep_guard
  !
  LOGICAL FUNCTION symrot_pw_on()
    !! RTSC_PW_SYMROT=1, with the same declines as RTSC_NSCF_SYMROT
    CHARACTER(LEN=16) :: s
    INTEGER :: ios
    symrot_pw_on = .FALSE.
    CALL get_environment_variable('RTSC_PW_SYMROT', s, STATUS=ios)
    IF (.NOT. (ios == 0 .AND. TRIM(s) == '1')) RETURN
    symrot_pw_on = symrot_ok()
  END FUNCTION symrot_pw_on
  !
  LOGICAL FUNCTION symrot_ok()
    USE noncollin_module, ONLY : noncolin
    USE control_flags,    ONLY : gamma_only
    USE lsda_mod,         ONLY : lsda
    USE mp_bands,         ONLY : nproc_bgrp
    USE mp_pools,         ONLY : npool
    USE ldaU,             ONLY : lda_plus_u
    USE uspp,             ONLY : okvan
    symrot_ok = .NOT. (noncolin .OR. gamma_only .OR. lsda .OR. nproc_bgrp /= 1 .OR. npool /= 1 .OR. lda_plus_u &
                       .OR. okvan)
  END FUNCTION symrot_ok
  !
  REAL(DP) FUNCTION symrot_pw_tight()
    !! RTSC_PW_SYMROT_TIGHT: factor on pw.x's nscf ethr for the representatives (default 1: pw.x's own ethr)
    CHARACTER(LEN=32) :: s
    INTEGER :: ios
    symrot_pw_tight = 1.0_DP
    CALL get_environment_variable('RTSC_PW_SYMROT_TIGHT', s, STATUS=ios)
    IF (ios == 0 .AND. LEN_TRIM(s) > 0) READ(s, *, IOSTAT=ios) symrot_pw_tight
    IF (ios /= 0 .OR. symrot_pw_tight <= 0) symrot_pw_tight = 1.0_DP
  END FUNCTION symrot_pw_tight
  !
  LOGICAL FUNCTION symrot_pw_check_on()
    CHARACTER(LEN=16) :: s
    INTEGER :: ios
    CALL get_environment_variable('RTSC_PW_SYMROT_CHECK', s, STATUS=ios)
    symrot_pw_check_on = (ios == 0 .AND. TRIM(s) == '1')
  END FUNCTION symrot_pw_check_on
  !
  REAL(DP) FUNCTION symrot_tight()
    CHARACTER(LEN=32) :: s
    INTEGER :: ios
    symrot_tight = 0.01_DP
    CALL get_environment_variable('RTSC_NSCF_SYMROT_TIGHT', s, STATUS=ios)
    IF (ios == 0 .AND. LEN_TRIM(s) > 0) READ(s, *, IOSTAT=ios) symrot_tight
    IF (ios /= 0 .OR. symrot_tight <= 0) symrot_tight = 0.01_DP
  END FUNCTION symrot_tight
  !
  LOGICAL FUNCTION symrot_check_on()
    CHARACTER(LEN=16) :: s
    INTEGER :: ios
    CALL get_environment_variable('RTSC_NSCF_SYMROT_CHECK', s, STATUS=ios)
    symrot_check_on = (ios == 0 .AND. TRIM(s) == '1')
  END FUNCTION symrot_check_on
  !
  !------------------------------------------------------------------------------------------
  SUBROUTINE kcryst(ik, c)
    USE cell_base, ONLY : at
    USE klist,     ONLY : xk
    INTEGER, INTENT(IN) :: ik
    REAL(DP), INTENT(OUT) :: c(3)
    c = MATMUL(TRANSPOSE(at), xk(:, ik))
  END SUBROUTINE kcryst
  !
  LOGICAL FUNCTION find_op(xs, xt, isym, tr, g0)
    !! xt = tr S xs + g0 for some op of the full crystal group (and time reversal)?
    USE symm_base, ONLY : time_reversal
    REAL(DP), INTENT(IN) :: xs(3), xt(3)
    INTEGER, INTENT(OUT) :: isym, tr, g0(3)
    INTEGER :: is, itr, j
    REAL(DP) :: kr(3), d(3)
    find_op = .FALSE.
    DO itr = 1, 2
      tr = 3 - 2 * itr
      IF (tr == -1 .AND. .NOT. time_reversal) EXIT
      DO is = 1, nsr
        DO j = 1, 3
          kr(j) = tr * (sr(j, 1, is) * xs(1) + sr(j, 2, is) * xs(2) + sr(j, 3, is) * xs(3))
        ENDDO
        d = xt - kr
        IF (ALL(ABS(d - NINT(d)) < 1.d-6)) THEN
          isym = is; g0 = NINT(d); find_op = .TRUE.
          RETURN
        ENDIF
      ENDDO
    ENDDO
  END FUNCTION find_op
  !
  INTEGER FUNCTION lin(mm)
    USE fft_base, ONLY : dfftp
    INTEGER, INTENT(IN) :: mm(3)
    INTEGER :: n1, n2, n3
    n1 = MODULO(mm(1), dfftp%nr1); n2 = MODULO(mm(2), dfftp%nr2); n3 = MODULO(mm(3), dfftp%nr3)
    lin = 1 + n1 + n2 * dfftp%nr1 + n3 * dfftp%nr1 * dfftp%nr2
  END FUNCTION lin
  !
  SUBROUTINE pos_set(ik, val)
    !! pos(lin(G)) = ig for the k+G list of ik (val = 1), or reset to 0 (val = 0)
    USE klist, ONLY : ngk, igk_k
    USE gvect, ONLY : mill
    INTEGER, INTENT(IN) :: ik, val
    INTEGER :: ig
    DO ig = 1, ngk(ik)
      pos(lin(mill(:, igk_k(ig, ik)))) = ig * val
    ENDDO
  END SUBROUTINE pos_set
  !
  LOGICAL FUNCTION map_ok(ngs, ms, ik, isym, tr, g0)
    !! does the source sphere (ngs Miller indices ms) map exactly onto the k+G list of ik? (pos set for ik)
    USE klist,     ONLY : ngk
    INTEGER, INTENT(IN) :: ngs, ms(3, ngs), ik, isym, tr, g0(3)
    INTEGER :: ig, j, mr(3), ir
    map_ok = .FALSE.
    IF (ngs /= ngk(ik)) RETURN
    DO ig = 1, ngs
      DO j = 1, 3
        mr(j) = tr * (sr(j, 1, isym) * ms(1, ig) + sr(j, 2, isym) * ms(2, ig) + sr(j, 3, isym) * ms(3, ig)) - g0(j)
      ENDDO
      ir = pos(lin(mr))
      IF (ir == 0) RETURN
    ENDDO
    map_ok = .TRUE.
  END FUNCTION map_ok
  !
  SUBROUTINE rotate(ngs, ms, xs, ws, ik, isym, tr, g0, evc)
    !! evc(:, 1:nbnd) of ik from the source (Miller ms, crystal k xs, coefficients ws); pos set for ik
    USE wvfct,     ONLY : nbnd, npwx
    USE constants, ONLY : tpi
    INTEGER, INTENT(IN) :: ngs, ms(3, ngs), ik, isym, tr, g0(3)
    REAL(DP), INTENT(IN) :: xs(3)
    COMPLEX(DP), INTENT(IN) :: ws(:,:)
    COMPLEX(DP), INTENT(OUT) :: evc(npwx, nbnd)
    INTEGER :: ig, j, mr(3), ib
    REAL(DP) :: arg
    LOGICAL :: hasft
    INTEGER, ALLOCATABLE :: imap(:)
    COMPLEX(DP), ALLOCATABLE :: ph(:)
    hasft = ANY(ABS(ftr(:, isym)) > 1.d-12)
    IF (hasft) nft = nft + 1
    ALLOCATE(imap(ngs))
    IF (hasft) ALLOCATE(ph(ngs))
    DO ig = 1, ngs
      DO j = 1, 3
        mr(j) = tr * (sr(j, 1, isym) * ms(1, ig) + sr(j, 2, isym) * ms(2, ig) + sr(j, 3, isym) * ms(3, ig)) - g0(j)
      ENDDO
      imap(ig) = pos(lin(mr))
      IF (hasft) THEN
        arg = tpi * ((xs(1) + ms(1, ig)) * ftr(1, isym) + (xs(2) + ms(2, ig)) * ftr(2, isym) + &
                     (xs(3) + ms(3, ig)) * ftr(3, isym))
        ph(ig) = CMPLX(COS(arg), -SIN(arg), KIND=DP)
      ENDIF
    ENDDO
    ! band-outer copy (contiguous reads; the sphere maps onto itself, so every row 1..ngs is written)
    DO ib = 1, nbnd
      evc(ngs + 1:npwx, ib) = (0.d0, 0.d0)
      IF (hasft) THEN
        IF (tr == 1) THEN
          DO ig = 1, ngs
            evc(imap(ig), ib) = ws(ig, ib) * ph(ig)
          ENDDO
        ELSE
          DO ig = 1, ngs
            evc(imap(ig), ib) = CONJG(ws(ig, ib) * ph(ig))
          ENDDO
        ENDIF
      ELSE
        IF (tr == 1) THEN
          DO ig = 1, ngs
            evc(imap(ig), ib) = ws(ig, ib)
          ENDDO
        ELSE
          DO ig = 1, ngs
            evc(imap(ig), ib) = CONJG(ws(ig, ib))
          ENDDO
        ENDIF
      ENDIF
    ENDDO
    DEALLOCATE(imap)
    IF (hasft) DEALLOCATE(ph)
  END SUBROUTINE rotate
  !
  LOGICAL FUNCTION hkey(x, key)
    REAL(DP), INTENT(IN) :: x(3)
    INTEGER(8), INTENT(OUT) :: key
    INTEGER(8) :: i(3)
    INTEGER :: j
    REAL(DP) :: y
    hkey = .FALSE.
    DO j = 1, 3
      y = x(j) * REAL(MGRID, DP)
      IF (ABS(y - ANINT(y)) > 1.d-3) RETURN
      i(j) = MODULO(NINT(y, KIND=8), MGRID)
    ENDDO
    key = (i(1) * MGRID + i(2)) * MGRID + i(3)
    hkey = .TRUE.
  END FUNCTION hkey
  !
  SUBROUTINE h_reset(t)
    TYPE(htab), INTENT(INOUT) :: t
    IF (ALLOCATED(t%key)) DEALLOCATE(t%key, t%val)
    t%cap = 1024; t%n = 0
    ALLOCATE(t%key(t%cap), t%val(3, t%cap))
    t%key = -1
  END SUBROUTINE h_reset
  !
  INTEGER FUNCTION h_slot(t, key)
    TYPE(htab), INTENT(IN) :: t
    INTEGER(8), INTENT(IN) :: key
    INTEGER(8) :: h
    h = IEOR(key, ISHFT(key, -29)) * 2654435761_8
    h_slot = INT(MODULO(IEOR(h, ISHFT(h, -32)), INT(t%cap, 8))) + 1
    DO WHILE (t%key(h_slot) /= -1 .AND. t%key(h_slot) /= key)
      h_slot = MODULO(h_slot, t%cap) + 1
    ENDDO
  END FUNCTION h_slot
  !
  RECURSIVE SUBROUTINE h_put(t, key, id, isym, tr)
    TYPE(htab), INTENT(INOUT) :: t
    INTEGER(8), INTENT(IN) :: key
    INTEGER, INTENT(IN) :: id, isym, tr
    INTEGER(8), ALLOCATABLE :: k0(:)
    INTEGER, ALLOCATABLE :: v0(:,:)
    INTEGER :: i, c0
    IF (t%cap == 0) CALL h_reset(t)
    IF (2 * (t%n + 1) > t%cap) THEN
      CALL MOVE_ALLOC(t%key, k0); CALL MOVE_ALLOC(t%val, v0)
      c0 = t%cap; t%cap = 2 * c0; t%n = 0
      ALLOCATE(t%key(t%cap), t%val(3, t%cap))
      t%key = -1
      DO i = 1, c0
        IF (k0(i) /= -1) CALL h_put(t, k0(i), v0(1, i), v0(2, i), v0(3, i))
      ENDDO
      DEALLOCATE(k0, v0)
    ENDIF
    i = h_slot(t, key)
    IF (t%key(i) == key) RETURN     ! keep the first source of this image
    t%key(i) = key; t%val(:, i) = (/id, isym, tr/); t%n = t%n + 1
  END SUBROUTINE h_put
  !
  LOGICAL FUNCTION h_get(t, key, id, isym, tr)
    TYPE(htab), INTENT(IN) :: t
    INTEGER(8), INTENT(IN) :: key
    INTEGER, INTENT(OUT) :: id, isym, tr
    INTEGER :: i
    h_get = .FALSE.
    IF (t%cap == 0) RETURN
    i = h_slot(t, key)
    IF (t%key(i) /= key) RETURN
    id = t%val(1, i); isym = t%val(2, i); tr = t%val(3, i)
    h_get = .TRUE.
  END FUNCTION h_get
  !
  SUBROUTINE h_images(t, xs, id)
    !! insert every image tr S xs of source id
    USE symm_base, ONLY : time_reversal
    TYPE(htab), INTENT(INOUT) :: t
    REAL(DP), INTENT(IN) :: xs(3)
    INTEGER, INTENT(IN) :: id
    INTEGER :: is, js, itr, tr, j, ipass
    INTEGER(8) :: key
    REAL(DP) :: kr(3)
    LOGICAL :: hasft
    ! RTSC_NSCF_SYMROT_PREFER_FT=1 (diagnostic): ops with a fractional translation are inserted first, so they are
    ! the ones used wherever they reach the image (exercises the phase factor)
    DO ipass = 1, 2
    DO itr = 1, 2
      tr = 3 - 2 * itr
      IF (tr == -1 .AND. .NOT. time_reversal) EXIT
      DO js = 1, nsr
        ! RTSC_NSCF_SYMROT_REVOPS=1 (diagnostic): reversed op order, so images with a non-trivial stabilizer are
        ! reached by another op (a pure gauge change of the rotated set)
        is = js
        IF (rev_ops) is = nsr + 1 - js
        hasft = ANY(ABS(ftr(:, is)) > 1.d-12)
        IF (prefer_ft) THEN
          IF ((ipass == 1) .NEQV. hasft) CYCLE
        ELSE
          IF (ipass == 2) CYCLE
        ENDIF
        DO j = 1, 3
          kr(j) = tr * (sr(j, 1, is) * xs(1) + sr(j, 2, is) * xs(2) + sr(j, 3, is) * xs(3))
        ENDDO
        IF (hkey(kr, key)) CALL h_put(t, key, id, is, tr)
      ENDDO
    ENDDO
    ENDDO
  END SUBROUTINE h_images
  !
  LOGICAL FUNCTION g0_of(xs, xt, isym, tr, g0)
    REAL(DP), INTENT(IN) :: xs(3), xt(3)
    INTEGER, INTENT(IN) :: isym, tr
    INTEGER, INTENT(OUT) :: g0(3)
    REAL(DP) :: d(3)
    INTEGER :: j
    DO j = 1, 3
      d(j) = xt(j) - tr * (sr(j, 1, isym) * xs(1) + sr(j, 2, isym) * xs(2) + sr(j, 3, isym) * xs(3))
    ENDDO
    g0 = NINT(d)
    g0_of = ALL(ABS(d - g0) < 1.d-6)
  END FUNCTION g0_of
  !
  !------------------------------------------------------------------------------------------
  SUBROUTINE symrot_plan(kdone, kskip)
    !------------------------------------------------------------------------------------------
    !! Called by c_bands_nscf_ph before any eigensolve. k-points whose orbit is in the cache are rotated
    !! now (kdone = .TRUE.). Of the others, one per orbit (the rep) is left to the eigensolver; the rest are
    !! marked kskip = .TRUE. and rotated by symrot_after_solve.
    USE klist,     ONLY : nks, ngk
    USE wvfct,     ONLY : et, nbnd, npwx
    USE fft_base,  ONLY : dfftp
    USE wavefunctions, ONLY : evc
    USE io_files,  ONLY : iunwfc, nwordwfc
    USE buffers,   ONLY : save_buffer
    USE control_flags, ONLY : io_level
    USE gvect,     ONLY : mill
    USE klist,     ONLY : igk_k
    USE io_global, ONLY : stdout
    USE symm_base, ONLY : s_cur => s, ft_cur => ft, nsym_cur => nsym
    LOGICAL, INTENT(INOUT) :: kdone(nks), kskip(nks)
    INTEGER :: ik, i0, isym, tr, g0(3), jk, i1, j1
    REAL(DP) :: xt(3), xs(3)
    INTEGER, ALLOCATABLE :: mrep(:,:)
    LOGICAL :: found, keyed, hit
    INTEGER(8) :: key, c0, c1, cr
    CHARACTER(LEN=16) :: sv
    INTEGER :: ios, miss(4), nprt
    !
    miss = 0; nprt = 0
    CALL start_clock('nscf_symrot')
    CALL SYSTEM_CLOCK(c0, cr)
    CALL h_reset(tl)
    IF (tc%cap == 0) THEN
      CALL h_reset(tc)
      CALL get_environment_variable('RTSC_NSCF_SYMROT_PREFER_FT', sv, STATUS=ios)
      prefer_ft = (ios == 0 .AND. TRIM(sv) == '1')
      nsr = nsym_cur
      sr(:, :, 1:nsr) = s_cur(:, :, 1:nsr)
      ftr(:, 1:nsr) = ft_cur(:, 1:nsr)
      CALL get_environment_variable('RTSC_NSCF_SYMROT_REVOPS', sv, STATUS=ios)
      rev_ops = (ios == 0 .AND. TRIM(sv) == '1')
    ENDIF
    IF (npos /= dfftp%nr1 * dfftp%nr2 * dfftp%nr3) THEN
      IF (ALLOCATED(pos)) DEALLOCATE(pos)
      npos = dfftp%nr1 * dfftp%nr2 * dfftp%nr3
      ALLOCATE(pos(npos))
      pos = 0
    ENDIF
    IF (nc > 0 .AND. nbc /= nbnd) CALL errore('symrot_plan', 'nbnd changed between q', 1)
    IF (ALLOCATED(kind_)) DEALLOCATE(kind_, ksrc, ksym, ktr, kg0)
    ALLOCATE(kind_(nks), ksrc(nks), ksym(nks), ktr(nks), kg0(3, nks))
    kind_ = 0; ksrc = 0; ksym = 0; ktr = 0; kg0 = 0
    nrot_c = 0; nrot_l = 0; nrep = 0; nfb = 0
    ALLOCATE(mrep(3, npwx))
    DO ik = 1, nks
      IF (kdone(ik)) CYCLE
      CALL kcryst(ik, xt)
      CALL pos_set(ik, 1)
      found = .FALSE.
      ! 1) the cache (hashed image lookup; linear scan if k is off the hash grid or the hashed map fails)
      keyed = hkey(xt, key)
      hit = .FALSE.
      IF (keyed) THEN
        IF (h_get(tc, key, i0, isym, tr)) THEN
          IF (g0_of(cxk(:, i0), xt, isym, tr, g0)) THEN
            hit = map_ok(cngw(i0), cmill(:, :, i0), ik, isym, tr, g0)
            IF (.NOT. hit) THEN
              IF (cngw(i0) /= ngk(ik)) THEN
                miss(3) = miss(3) + 1
              ELSE
                miss(4) = miss(4) + 1
              ENDIF
              IF (nprt < 3) THEN
                nprt = nprt + 1
                WRITE(stdout, '(5x,"symrot miss: k ",3F10.6," src ",3F10.6," op ",I3," tr ",I2," ngs ",I0," ngk ",I0)') &
                     xt, cxk(:, i0), isym, tr, cngw(i0), ngk(ik)
              ENDIF
            ENDIF
          ELSE
            miss(2) = miss(2) + 1
          ENDIF
        ELSEIF (nc > 0) THEN
          miss(1) = miss(1) + 1
        ENDIF
      ENDIF
      DO i1 = 1, nc
        IF (hit) THEN
          IF (i1 > 1) EXIT
        ELSE
          IF (keyed) EXIT     ! hashed: every image of every cached k is in tc
          i0 = i1
          IF (.NOT. find_op(cxk(:, i0), xt, isym, tr, g0)) CYCLE
          IF (.NOT. map_ok(cngw(i0), cmill(:, :, i0), ik, isym, tr, g0)) CYCLE
        ENDIF
        CALL rotate(cngw(i0), cmill(:, 1:cngw(i0), i0), cxk(:, i0), cwfc(:, :, i0), ik, isym, tr, g0, evc)
        et(1:nbnd, ik) = ce(1:nbnd, i0)
        IF (io_level > -1) CALL save_buffer(evc, nwordwfc, iunwfc, ik)
        kdone(ik) = .TRUE.
        kind_(ik) = 1; ksrc(ik) = i0
        nrot_c = nrot_c + 1
        found = .TRUE.
        EXIT
      ENDDO
      ! 2) a rep earlier in this list
      IF (.NOT. found) THEN
        hit = .FALSE.
        IF (keyed) THEN
          IF (h_get(tl, key, jk, isym, tr)) THEN
            CALL kcryst(jk, xs)
            IF (g0_of(xs, xt, isym, tr, g0)) THEN
              mrep(:, 1:ngk(jk)) = mill(:, igk_k(1:ngk(jk), jk))
              hit = map_ok(ngk(jk), mrep, ik, isym, tr, g0)
            ENDIF
          ENDIF
        ENDIF
        DO j1 = 1, ik - 1
          IF (hit) THEN
            IF (j1 > 1) EXIT
          ELSE
            IF (keyed) EXIT
            jk = j1
            IF (kind_(jk) /= 2) CYCLE
            CALL kcryst(jk, xs)
            IF (.NOT. find_op(xs, xt, isym, tr, g0)) CYCLE
            mrep(:, 1:ngk(jk)) = mill(:, igk_k(1:ngk(jk), jk))
            IF (.NOT. map_ok(ngk(jk), mrep, ik, isym, tr, g0)) CYCLE
          ENDIF
          kind_(ik) = 3; ksrc(ik) = jk; ksym(ik) = isym; ktr(ik) = tr; kg0(:, ik) = g0
          kskip(ik) = .TRUE.
          nrot_l = nrot_l + 1
          found = .TRUE.
          EXIT
        ENDDO
      ENDIF
      IF (.NOT. found) THEN
        kind_(ik) = 2
        nrep = nrep + 1
        CALL h_images(tl, xt, ik)
      ENDIF
      CALL pos_set(ik, 0)
    ENDDO
    DEALLOCATE(mrep)
    IF (ANY(miss > 0)) WRITE(stdout, '(5x,"rtsc nscf symrot: cache misses: no image ",I0,", G0 ",I0,", sphere size ",I0, &
         &", sphere map ",I0)') miss
    !$acc update device(et)
    CALL SYSTEM_CLOCK(c1)
    t_plan = t_plan + REAL(c1 - c0, DP) / REAL(cr, DP)
    CALL stop_clock('nscf_symrot')
  END SUBROUTINE symrot_plan
  !
  !------------------------------------------------------------------------------------------
  SUBROUTINE symrot_after_solve(kdone, kskip)
    !------------------------------------------------------------------------------------------
    !! after the eigensolver: cache the reps, rotate the k-points that wait for them
    USE klist,     ONLY : nks, ngk, igk_k
    USE wvfct,     ONLY : et, nbnd, npwx
    USE wavefunctions, ONLY : evc
    USE io_files,  ONLY : iunwfc, nwordwfc
    USE buffers,   ONLY : save_buffer, get_buffer
    USE control_flags, ONLY : io_level
    USE gvect,     ONLY : mill
    USE io_global, ONLY : stdout
    LOGICAL, INTENT(INOUT) :: kdone(nks), kskip(nks)
    INTEGER :: ik, jk, i0, n
    INTEGER, ALLOCATABLE :: slot(:)
    REAL(DP) :: xs(3)
    !
    INTEGER(8) :: c0, c1, cr
    CALL start_clock('nscf_symrot')
    CALL SYSTEM_CLOCK(c0, cr)
    IF (io_level <= -1) CALL errore('symrot_after_solve', 'RTSC_NSCF_SYMROT needs the wfc buffer (io_level > -1)', 1)
    ALLOCATE(slot(nks))
    slot = 0
    ! cache the reps that were solved
    n = COUNT(kind_ == 2 .AND. kdone)
    CALL grow(nc + n)
    DO ik = 1, nks
      IF (kind_(ik) /= 2 .OR. .NOT. kdone(ik)) CYCLE
      CALL get_buffer(evc, nwordwfc, iunwfc, ik)
      nc = nc + 1
      CALL kcryst(ik, cxk(:, nc))
      cngw(nc) = ngk(ik)
      cmill(:, 1:ngk(ik), nc) = mill(:, igk_k(1:ngk(ik), ik))
      cwfc(:, :, nc) = (0.d0, 0.d0)
      cwfc(1:ngk(ik), 1:nbnd, nc) = evc(1:ngk(ik), 1:nbnd)
      ce(1:nbnd, nc) = et(1:nbnd, ik)
      slot(ik) = nc
      CALL h_images(tc, cxk(:, nc), nc)
    ENDDO
    ! rotate the waiting k-points
    DO ik = 1, nks
      IF (kind_(ik) /= 3) CYCLE
      jk = ksrc(ik)
      kskip(ik) = .FALSE.
      i0 = slot(jk)
      IF (i0 == 0) THEN
        kind_(ik) = 0      ! its rep was not solved: diagonalize it itself
        nfb = nfb + 1
        CYCLE
      ENDIF
      CALL pos_set(ik, 1)
      CALL rotate(cngw(i0), cmill(:, 1:cngw(i0), i0), cxk(:, i0), cwfc(:, :, i0), ik, ksym(ik), ktr(ik), kg0(:, ik), evc)
      CALL pos_set(ik, 0)
      et(1:nbnd, ik) = ce(1:nbnd, i0)
      CALL save_buffer(evc, nwordwfc, iunwfc, ik)
      kdone(ik) = .TRUE.
      ksrc(ik) = i0
    ENDDO
    DEALLOCATE(slot)
    !$acc update device(et)
    tot_rot = tot_rot + nrot_c + nrot_l - nfb
    tot_dav = tot_dav + nrep + nfb
    CALL SYSTEM_CLOCK(c1)
    t_after = t_after + REAL(c1 - c0, DP) / REAL(cr, DP)
    WRITE(stdout, '(5x,"rtsc nscf symrot: ",I0," k rotated from the cache, ",I0," from reps of this list, ",I0, &
         &" reps diagonalized, ",I0," fallbacks; cache ",I0," k; time plan+rotate ",F8.3," s, cache+rotate ",F8.3, &
         &" s (cumulative); ",I0," rotations with a fractional translation (cumulative)")') &
         nrot_c, nrot_l - nfb, nrep, nfb, nc, t_plan, t_after, nft
    CALL stop_clock('nscf_symrot')
  END SUBROUTINE symrot_after_solve
  !
  SUBROUTINE grow(n)
    USE wvfct, ONLY : nbnd, npwx
    INTEGER, INTENT(IN) :: n
    REAL(DP), ALLOCATABLE :: t1(:,:), t2(:,:)
    INTEGER, ALLOCATABLE :: i1(:), i2(:,:,:)
    COMPLEX(DP), ALLOCATABLE :: w(:,:,:)
    INTEGER :: m, np
    IF (n <= ncmax .AND. npwx <= npwc) RETURN
    m = MAX(n, 2 * ncmax, 16)
    np = MAX(npwx, npwc)
    ALLOCATE(t1(3, m), t2(nbnd, m), i1(m), i2(3, np, m), w(np, nbnd, m))
    i2 = 0; w = (0.d0, 0.d0)
    IF (nc > 0) THEN
      t1(:, 1:nc) = cxk(:, 1:nc); t2(:, 1:nc) = ce(:, 1:nc); i1(1:nc) = cngw(1:nc)
      i2(:, 1:npwc, 1:nc) = cmill(:, :, 1:nc); w(1:npwc, :, 1:nc) = cwfc(:, :, 1:nc)
    ENDIF
    IF (ALLOCATED(cxk)) DEALLOCATE(cxk, ce, cngw, cmill, cwfc)
    CALL MOVE_ALLOC(t1, cxk); CALL MOVE_ALLOC(t2, ce); CALL MOVE_ALLOC(i1, cngw)
    CALL MOVE_ALLOC(i2, cmill); CALL MOVE_ALLOC(w, cwfc)
    ncmax = m; npwc = np; nbc = nbnd
  END SUBROUTINE grow
  !
  !------------------------------------------------------------------------------------------
  SUBROUTINE symrot_resid()
    !------------------------------------------------------------------------------------------
    !! RTSC_NSCF_SYMROT_CHECK=1: max_n |H psi_n - e_n psi_n| over rotated and over Davidson k-points
    USE klist,     ONLY : nks, ngk, igk_k, xk
    USE wvfct,     ONLY : et, nbnd, npwx, current_k
    USE wavefunctions, ONLY : evc
    USE io_files,  ONLY : iunwfc, nwordwfc
    USE buffers,   ONLY : get_buffer
    USE uspp,      ONLY : vkb, nkb
    USE uspp_init, ONLY : init_us_2
    USE io_global, ONLY : stdout
    USE becmod,    ONLY : becp, allocate_bec_type_acc, deallocate_bec_type_acc
    COMPLEX(DP), ALLOCATABLE :: hpsi(:,:)
    INTEGER :: ik, ib, npw
    REAL(DP) :: r, rr, rd, ro
    ALLOCATE(hpsi(npwx, nbnd))
    ! h_psi's nonlocal term uses the global becp (allocated by diag_bands in the eigensolver path)
    CALL allocate_bec_type_acc(nkb, nbnd, becp)
    !$acc enter data create(hpsi)
    rr = 0; rd = 0; ro = 0
    DO ik = 1, nks
      current_k = ik
      npw = ngk(ik)
      CALL g2_kin(ik)
      IF (nkb > 0) CALL init_us_2(npw, igk_k(1, ik), xk(1, ik), vkb, .true.)
      CALL get_buffer(evc, nwordwfc, iunwfc, ik)
      !$acc update device(evc)
      CALL h_psi(npwx, npw, nbnd, evc, hpsi)
      !$acc update host(hpsi)
      r = 0
      DO ib = 1, nbnd
        r = MAX(r, SQRT(SUM(ABS(hpsi(1:npw, ib) - et(ib, ik) * evc(1:npw, ib))**2)))
        ro = MAX(ro, ABS(SQRT(SUM(ABS(evc(1:npw, ib))**2)) - 1.d0))
      ENDDO
      IF (kind_(ik) == 1 .OR. kind_(ik) == 3) THEN
        rr = MAX(rr, r)
      ELSE
        rd = MAX(rd, r)
      ENDIF
    ENDDO
    !$acc exit data delete(hpsi)
    DEALLOCATE(hpsi)
    CALL deallocate_bec_type_acc(becp)
    rmax_rot = MAX(rmax_rot, rr); rmax_dav = MAX(rmax_dav, rd)
    WRITE(stdout, '(5x,"rtsc nscf symrot check: max |H psi - e psi| rotated ",ES10.3,", diagonalized ",ES10.3, &
         &"; max | |psi| - 1 | ",ES10.3)') rr, rd, ro
  END SUBROUTINE symrot_resid
  !
  SUBROUTINE symrot_report()
    USE io_global, ONLY : stdout
    WRITE(stdout, '(5x,"rtsc nscf symrot totals: ",I0," rotated, ",I0," diagonalized")') tot_rot, tot_dav
  END SUBROUTINE symrot_report
  !
END MODULE sth_nscf_symrot
