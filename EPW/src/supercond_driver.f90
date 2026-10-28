  !
  ! Copyright (C) 2023-2026 EPW-Collaboration
  ! Copyright (C) 2016-2023 EPW-Collaboration
  ! Copyright (C) 2010-2016 Samuel Ponce', Roxana Margine, Carla Verdi, Feliciano Giustino
  ! Copyright (C) 2007-2009 Roxana Margine
  !
  ! This file is distributed under the terms of the GNU General Public
  ! License. See the file `LICENSE' in the root directory of the
  ! present distribution, or http://www.gnu.org/copyleft.gpl.txt .
  !
  !----------------------------------------------------------------------
  MODULE supercond_driver
  !----------------------------------------------------------------------
  !!
  !! This module contains all the driver subroutines for superonductivity calculations.
  !! So far we only have the Migdal Eliashberg driver but other can be added in the future.
  !!
  IMPLICIT NONE
  !
  CONTAINS
    !
    !-----------------------------------------------------------------------
    SUBROUTINE eliashberg_eqs()
    !-----------------------------------------------------------------------
    !!
    !! This is the main driver for solving the Eliashberg equations
    !!
    USE io_global,         ONLY : stdout
    USE input,             ONLY : liso, fila2f, gap_edge, lreal, limag, laniso, &
                                  tc_linear, fbw, icoulomb, a2f_iso
    USE supercond_common,  ONLY : gap0
    USE supercond,         ONLY : eliashberg_init, estimate_tc_gap, find_a2f, &
                                  deallocate_eliashberg_elphon
    USE io_supercond,      ONLY : read_a2f, read_frequencies, read_eigenvalues, &
                                  read_ephmat, read_kqmap
    USE supercond_coul,    ONLY : read_eigen_cl, find_indices_ik_cl, &
                                  find_nbnd_offset
    USE supercond_iso,     ONLY : eliashberg_iso_iaxis, eliashberg_iso_raxis, &
                                  crit_temp_iso
    USE supercond_aniso,   ONLY : eliashberg_aniso_iaxis
    !
    IMPLICIT NONE
    !
    CALL start_clock('ELIASHBERG')
    !
    IF (liso) THEN
      IF (.NOT. a2f_iso) THEN
        WRITE(stdout, '(/5x, a)') REPEAT('=', 67)
        IF (fbw) THEN
          WRITE(stdout, '(5x, "Solve full-bandwidth isotropic Eliashberg equations")')
        ELSE
          WRITE(stdout, '(5x, "Solve isotropic Eliashberg equations")')
        ENDIF
        WRITE(stdout, '(5x, a/)') REPEAT('=', 67)
        CALL start_clock('el_init')
        CALL eliashberg_init()
        CALL stop_clock('el_init')
        IF (fila2f == ' ') THEN
          CALL start_clock('el_readfreq')
          CALL read_frequencies()
          CALL stop_clock('el_readfreq')
          CALL start_clock('el_readeig')
          CALL read_eigenvalues()
          CALL stop_clock('el_readeig')
          CALL start_clock('el_readkq')
          CALL read_kqmap()
          CALL stop_clock('el_readkq')
          CALL start_clock('el_readeph')
          CALL read_ephmat()
          CALL stop_clock('el_readeph')
          CALL start_clock('el_finda2f')
          CALL find_a2f()
          CALL stop_clock('el_finda2f')
          CALL deallocate_eliashberg_elphon()
        ENDIF
        !
        CALL start_clock('el_reada2f')
        CALL read_a2f()
        CALL stop_clock('el_reada2f')
        CALL estimate_tc_gap()
        IF (gap_edge > 0.d0) THEN
          gap0 = gap_edge
        ENDIF
        IF (lreal) CALL eliashberg_iso_raxis()
        IF (limag .AND. tc_linear) CALL crit_temp_iso()
        IF (limag .AND. .NOT. tc_linear) CALL eliashberg_iso_iaxis()
      ELSE
        !! Added by S. Mishra for image-parallelization isotropic a2f without storing g2
        IF (fbw) THEN
          WRITE(stdout, '(5x, "Solve adiabatic full-bandwidth isotropic Eliashberg equations (image para)")')
        ELSE 
          WRITE(stdout, '(5x, "Solve adiabatic FSR isotropic Eliashberg equations (image para)")')
        ENDIF
        WRITE(stdout, '(5x, a/)') REPEAT('=', 67)
        CALL start_clock('el_init')
        CALL eliashberg_init()
        CALL stop_clock('el_init')
        CALL start_clock('el_reada2f')
        CALL read_a2f()
        CALL stop_clock('el_reada2f')
        CALL estimate_tc_gap()
        IF (gap_edge > 0.d0) THEN
          gap0 = gap_edge
        ENDIF 
        !
        IF (lreal) CALL eliashberg_iso_raxis()
        IF (limag .AND. tc_linear) CALL crit_temp_iso()
        IF (limag .AND. .NOT. tc_linear) CALL eliashberg_iso_iaxis()
      ENDIF ! a2f_iso
    ENDIF
    !
    IF (laniso) THEN
      WRITE(stdout, '(/5x, a)') REPEAT('=', 67)
      IF (fbw) THEN
        WRITE(stdout, '(5x, "Solve full-bandwidth anisotropic Eliashberg equations")')
      ELSE
        WRITE(stdout, '(5x, "Solve anisotropic Eliashberg equations")')
      ENDIF
      WRITE(stdout, '(5x, a/)') REPEAT('=', 67)
      CALL start_clock('el_init')
      CALL eliashberg_init()
      CALL stop_clock('el_init')
      CALL start_clock('el_readfreq')
      CALL read_frequencies()
      CALL stop_clock('el_readfreq')
      CALL start_clock('el_readeig')
      CALL read_eigenvalues()
      CALL stop_clock('el_readeig')
      CALL start_clock('el_readkq')
      CALL read_kqmap()
      CALL stop_clock('el_readkq')
      CALL start_clock('el_readeph')
      CALL read_ephmat()
      CALL stop_clock('el_readeph')
      IF (icoulomb > 0) THEN
        CALL read_eigen_cl()
        CALL find_indices_ik_cl()
        CALL find_nbnd_offset()
      ENDIF
      CALL start_clock('el_finda2f')
      CALL find_a2f()
      CALL stop_clock('el_finda2f')
      CALL start_clock('el_reada2f')
      CALL read_a2f()
      CALL stop_clock('el_reada2f')
      CALL estimate_tc_gap()
      IF (gap_edge > 0.d0) THEN
        gap0 = gap_edge
      ENDIF
      CALL start_clock('el_aniso')
      CALL eliashberg_aniso_iaxis()
      CALL stop_clock('el_aniso')
    ENDIF
    !
    IF (.NOT. liso .AND. .NOT. laniso) THEN
      WRITE(stdout, '(/5x, a)') REPEAT('=', 67)
      WRITE(stdout, '(5x, "Calculate Eliashberg spectral function")')
      WRITE(stdout, '(5x, a/)') REPEAT('=', 67)
      CALL start_clock('el_init')
      CALL eliashberg_init()
      CALL stop_clock('el_init')
      CALL start_clock('el_readfreq')
      CALL read_frequencies()
      CALL stop_clock('el_readfreq')
      CALL start_clock('el_readeig')
      CALL read_eigenvalues()
      CALL stop_clock('el_readeig')
      CALL start_clock('el_readkq')
      CALL read_kqmap()
      CALL stop_clock('el_readkq')
      CALL start_clock('el_readeph')
      CALL read_ephmat()
      CALL stop_clock('el_readeph')
      CALL start_clock('el_finda2f')
      CALL find_a2f()
      CALL stop_clock('el_finda2f')
      CALL start_clock('el_reada2f')
      CALL read_a2f()
      CALL stop_clock('el_reada2f')
      CALL estimate_tc_gap()
      CALL deallocate_eliashberg_elphon()
    ENDIF
    !
    CALL stop_clock('ELIASHBERG')
    !
    RETURN
    !
    !-----------------------------------------------------------------------
    END SUBROUTINE eliashberg_eqs
    !-----------------------------------------------------------------------
  !-----------------------------------------------------------------------
  END MODULE supercond_driver
  !-----------------------------------------------------------------------
