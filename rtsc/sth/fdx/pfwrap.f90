! BIND(C) wrappers of sth_pfft / sth_pfft_s for the cuFFTDx harness (fdxbench.cu)
MODULE pfwrap
  USE cudafor
  USE iso_c_binding
  USE sth_pfft
  USE sth_pfft_s
  IMPLICIT NONE
CONTAINS
  SUBROUTINE pfw_setup(prec, nr, nrx, nksq, npwx, npwq, idxkq, chunk, ierr) BIND(C, name='pfw_setup')
    INTEGER(c_int), VALUE :: prec, nksq, npwx, chunk
    INTEGER(c_int) :: nr(3), nrx(3), npwq(nksq), idxkq(npwx, nksq), ierr
    IF (prec == 8) THEN
      CALL pfft_setup(nr, nrx, nksq, npwx, npwq, idxkq, chunk, ierr)
    ELSE
      CALL pfs_setup(nr, nrx, nksq, npwx, npwq, idxkq, chunk, ierr)
    ENDIF
  END SUBROUTINE pfw_setup
  SUBROUTINE pfw_info(nst, nxa, pc) BIND(C, name='pfw_info')
    INTEGER(c_int) :: nst, nxa, pc
    CALL pfft_info(nst, nxa, pc)
  END SUBROUTINE pfw_info
  SUBROUTINE pfw_local_d(x, y, list, nl, nslot, npert, npwx, nksq, nbnd, vrs, npwq, g2, etk, scale, strm) &
      BIND(C, name='pfw_local_d')
    INTEGER(c_int), VALUE :: nl, nslot, npert, npwx, nksq, nbnd
    COMPLEX(8), DEVICE :: x(npwx, *), y(npwx, *)
    INTEGER(c_int), DEVICE :: list(*), npwq(nksq)
    REAL(8), DEVICE :: vrs(*), g2(npwx, nksq), etk(nbnd, nksq)
    REAL(8), VALUE :: scale
    INTEGER(cuda_stream_kind), VALUE :: strm
    CALL pfft_local(x, y, list, nl, nslot, npert, npwx, nksq, nbnd, vrs, npwq, g2, etk, scale, strm)
  END SUBROUTINE pfw_local_d
  SUBROUTINE pfw_local_s(x, y, list, nl, nslot, npert, npwx, nksq, nbnd, vrs, npwq, g2, etk, scale, strm) &
      BIND(C, name='pfw_local_s')
    INTEGER(c_int), VALUE :: nl, nslot, npert, npwx, nksq, nbnd
    COMPLEX(4), DEVICE :: x(npwx, *), y(npwx, *)
    INTEGER(c_int), DEVICE :: list(*), npwq(nksq)
    REAL(4), DEVICE :: vrs(*), g2(npwx, nksq)
    REAL(8), DEVICE :: etk(nbnd, nksq)
    REAL(4), VALUE :: scale
    INTEGER(cuda_stream_kind), VALUE :: strm
    CALL pfs_local(x, y, list, nl, nslot, npert, npwx, nksq, nbnd, vrs, npwq, g2, etk, scale, strm)
  END SUBROUTINE pfw_local_s
END MODULE pfwrap
