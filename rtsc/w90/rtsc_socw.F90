! rtsc socw (groups/compute/SOCW-LOG.md; theory: groups/theory/returns/M7/W.md): RTSC_W90_DUMP=1 makes wann_main write
! the state it starts from (<seedname>.socw0, right after disentanglement and the initial projection) and the state
! it ends at (<seedname>.socw1) as unformatted stream files (native endianness) for the second-order-certified
! wannierise (SOC-W) python path. The routine only reads wannier90's arrays, so results are bitwise identical with
! the switch on or off (default: off, nothing is done).
! Layout: int32 magic 5050001, version 1, num_wann, num_kpts, num_bands, nntot, have_disentangled (0/1), tag;
! real64 real_lattice(3,3), recip_lattice(3,3), kpt_latt(3,nk); int32 nnlist(nk,nntot), nncell(3,nk,nntot);
! real64 wb(nntot), bk(3,nntot,nk), wbtot, omega_invariant; if disentangled: int32 ndimwin(nk), lwindow(nb,nk) (0/1),
! complex128 u_matrix_opt(nb,nw,nk), real64 eigval(nb,nk); then complex128 u_matrix(nw,nw,nk), m_matrix(nw,nw,nntot,nk).
module rtsc_w90_socw
  use w90_constants, only: dp
  implicit none
  private
  public :: rtsc_socw_dump
contains
  subroutine rtsc_socw_dump(tag)
    use w90_io, only: seedname, io_file_unit, stdout
    use w90_comms, only: on_root
    use w90_parameters, only: num_wann, num_kpts, num_bands, nntot, u_matrix, m_matrix, u_matrix_opt, &
      lwindow, ndimwin, eigval, nnlist, nncell, wb, bk, kpt_latt, real_lattice, recip_lattice, &
      omega_invariant, have_disentangled, wbtot
    integer, intent(in) :: tag
    character(len=16) :: ev
    character(len=1) :: ct
    integer :: ios, iu, idis
    integer, allocatable :: ilw(:, :)
    ev = ' '
    call get_environment_variable('RTSC_W90_DUMP', ev, status=ios)
    if (ios /= 0) return
    if (trim(ev) /= '1') return
    if (.not. on_root) return
    write (ct, '(i1)') tag
    iu = io_file_unit()
    open (unit=iu, file=trim(seedname)//'.socw'//ct, form='unformatted', access='stream', status='replace', &
          action='write')
    idis = 0
    if (have_disentangled) idis = 1
    write (iu) 5050001, 1, num_wann, num_kpts, num_bands, nntot, idis, tag
    write (iu) real_lattice, recip_lattice, kpt_latt
    write (iu) nnlist, nncell
    write (iu) wb, bk, wbtot, omega_invariant
    if (idis == 1) then
      allocate (ilw(num_bands, num_kpts))
      ilw = 0
      where (lwindow) ilw = 1
      write (iu) ndimwin, ilw
      deallocate (ilw)
      write (iu) u_matrix_opt, eigval
    endif
    write (iu) u_matrix, m_matrix
    close (iu)
    write (stdout, '(1x,a)') 'rtsc: RTSC_W90_DUMP wrote '//trim(seedname)//'.socw'//ct
  end subroutine rtsc_socw_dump
end module rtsc_w90_socw
