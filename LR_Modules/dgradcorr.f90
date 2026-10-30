!
! Copyright (C) 2001-2016 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!--------------------------------------------------------------------
subroutine dgradcorr (dfft, rho, grho, dvxc_rr, dvxc_sr, dvxc_ss, &
     dvxc_s, xq, drho, nspin, nspin0, g, dvxc)
  !--------------------------------------------------------------------
  !
  !  Add gradient correction contribution to 
  !  the response exchange-correlation potential dvxc.
  !  LSDA is allowed.         ADC (September 1999)
  !  Noncollinear is allowed. ADC (June 2007)
  !
  USE kinds,            ONLY : DP
  USE noncollin_module, ONLY : noncolin, domag
  USE gc_lr,            ONLY : gmag, vsgga, segni
  USE fft_types,        ONLY : fft_type_descriptor
  !
  IMPLICIT NONE
  !
  TYPE(fft_type_descriptor),INTENT(IN) :: dfft
  INTEGER, INTENT(IN) :: nspin, nspin0
  !
  REAL(DP), INTENT(IN) ::rho (dfft%nnr, nspin), grho (3, dfft%nnr, nspin0), &
       g (3, dfft%ngm), xq(3)       
  REAL(DP), INTENT(IN) :: &
       dvxc_rr(dfft%nnr, nspin0, nspin0), dvxc_sr (dfft%nnr, nspin0, nspin0), &
       dvxc_ss (dfft%nnr,nspin0, nspin0), dvxc_s (dfft%nnr, nspin0, nspin0)
  COMPLEX(DP), INTENT(IN) :: drho (dfft%nnr, nspin)
  COMPLEX(DP), INTENT(INOUT) :: dvxc (dfft%nnr, nspin)

  real(DP), parameter :: epsr = 1.0d-6, epsg = 1.0d-10
  real(DP) :: grho2, seg, seg0, amag
  complex(DP) :: s1, fact, term
  complex(DP) :: a (2, 2, 2), b (2, 2, 2, 2), c (2, 2, 2), &
                      ps (2, 2), ps1 (3, 2, 2), ps2 (3, 2, 2, 2)
  complex(DP), allocatable  :: gdrho (:,:,:), h (:,:,:), dh (:)
  complex(DP), allocatable  :: gdmag (:,:,:), dvxcsave(:,:), vgg(:,:)
  complex(DP), allocatable  :: drhoout(:,:)
  integer :: k, ipol, jpol, is, js, ks, ls

  IF ( nspin < nspin0 .OR. nspin < 1 .OR. nspin > 4 ) &
       CALL errore('dgradcorr', 'incorrect number of spin components',1)
  if (noncolin.and.domag) then
     allocate (gdmag(3, dfft%nnr, nspin))
     allocate (dvxcsave(dfft%nnr, nspin))
     allocate (vgg(dfft%nnr, nspin0))
     dvxcsave=dvxc
     dvxc=(0.0_dp,0.0_dp)
  endif
  allocate (drhoout( dfft%nnr, nspin0))
  allocate (gdrho( 3, dfft%nnr, nspin0))
  allocate (h( 3, dfft%nnr, nspin0))
  allocate (dh( dfft%nnr))
  
  h (:, :, :) = (0.d0, 0.d0)
  if (noncolin.and.domag) then
     do is = 1, nspin
        call fft_qgradient (dfft, drho(1,is), xq, g, gdmag (1, 1, is) )
     enddo
     DO is=1,nspin0
        IF (is==1) seg0=0.5_dp
        IF (is==2) seg0=-0.5_dp
        drhoout(:,is) = 0.5_dp*drho(:,1)
        DO ipol=1,3
           gdrho(ipol,:,is) = 0.5_dp*gdmag(ipol,:,1)
        ENDDO
        DO k=1,dfft%nnr
           seg=seg0*segni(k)
           amag=sqrt(rho(k,2)**2+rho(k,3)**2+rho(k,4)**2)
           IF (amag>1.d-12) THEN
              DO jpol=2,4
                 drhoout(k,is) = drhoout(k,is)+seg*rho(k,jpol)* &
                                                 drho(k,jpol)/amag
              END DO
              DO ipol=1,3
                 fact=(0.0_dp,0.0_dp)
                 DO jpol=2,4
                    fact=fact+rho(k,jpol)*drho(k,jpol)
                 END DO
                 DO jpol=2,4
                    gdrho(ipol,k,is) = gdrho(ipol,k,is)+ seg*( &
                        drho(k,jpol)*gmag(ipol,k,jpol)+ &
                        rho(k,jpol)*gdmag(ipol,k,jpol))/amag &
                        -seg*(rho(k,jpol)*gmag(ipol,k,jpol)*fact)/amag**3
                 END DO
              END DO
           END IF
        END DO
     END DO
  ELSE
     !
     DO is = 1, nspin0
        CALL fft_qgradient (dfft, drho(1,is), xq, g, gdrho (1, 1, is) )
        drhoout(:,is)=drho(:,is)
     ENDDO
     !
  ENDIF

  do k = 1, dfft%nnr
     grho2 = grho(1, k, 1)**2 + grho(2, k, 1)**2 + grho(3, k, 1)**2
     if (nspin == 1) then
        !
        !    LDA case
        !
        if (abs (rho (k, 1) ) > epsr .and. grho2 > epsg) then
           s1 = grho (1, k, 1) * gdrho (1, k, 1) + &
                grho (2, k, 1) * gdrho (2, k, 1) + &
                grho (3, k, 1) * gdrho (3, k, 1)
           !
           ! linear variation of the first term
           !
           dvxc (k, 1) = dvxc (k, 1) + dvxc_rr (k, 1, 1) * drho (k, 1) &
                + dvxc_sr (k, 1, 1) * s1
           do ipol = 1, 3
              h (ipol, k, 1) = (dvxc_sr(k, 1, 1) * drho(k, 1) + &
                                dvxc_ss(k, 1, 1) * s1 )*grho(ipol, k, 1) + &
                                dvxc_s (k, 1, 1) * gdrho (ipol, k, 1)
           enddo
        else
           do ipol = 1, 3
              h (ipol, k, 1) = (0.d0, 0.d0)
           enddo
        endif
     else
        !
        !    LSDA case
        !
        ps (:,:) = (0.d0, 0.d0)
        do is = 1, nspin0
           do js = 1, nspin0
              do ipol = 1, 3
                 ps1(ipol, is, js) = drhoout (k, is) * grho (ipol, k, js)
                 ps(is, js) = ps(is, js) + grho(ipol,k,is)*gdrho(ipol,k,js)
              enddo
              do ks = 1, nspin0
                 if (is == js .and. js == ks) then
                    a (is, js, ks) = dvxc_sr (k, is, is)
                    c (is, js, ks) = dvxc_sr (k, is, is)
                 else
                    if (is == 1) then
                       a (is, js, ks) = dvxc_sr (k, 1, 2)
                    else
                       a (is, js, ks) = dvxc_sr (k, 2, 1)
                    endif
                    if (js == 1) then
                       c (is, js, ks) = dvxc_sr (k, 1, 2)
                    else
                       c (is, js, ks) = dvxc_sr (k, 2, 1)
                    endif
                 endif
                 do ipol = 1, 3
                    ps2 (ipol, is, js, ks) = ps (is, js) * grho (ipol, k, ks)
                 enddo
                 do ls = 1, nspin0
                    if (is == js .and. js == ks .and. ks == ls) then
                       b (is, js, ks, ls) = dvxc_ss (k, is, is)
                    else
                       if (is == 1) then
                          b (is, js, ks, ls) = dvxc_ss (k, 1, 2)
                       else
                          b (is, js, ks, ls) = dvxc_ss (k, 2, 1)
                       endif
                    endif
                 enddo
              enddo
           enddo
        enddo
        do is = 1, nspin0
           do js = 1, nspin0
              dvxc (k, is) = dvxc (k, is) + dvxc_rr (k,is,js)*drhoout(k, js)
              do ipol = 1, 3
                 h (ipol, k, is) = h (ipol, k, is) + &
                      dvxc_s (k, is, js) * gdrho(ipol, k, js)
              enddo
              do ks = 1, nspin0
                 dvxc (k, is) = dvxc (k, is) + a (is, js, ks) * ps (js, ks)
                 do ipol = 1, 3
                    h (ipol, k, is) = h (ipol, k, is) + &
                         c (is, js, ks) * ps1 (ipol, js, ks)
                 enddo
                 do ls = 1, nspin0
                    do ipol = 1, 3
                       h (ipol, k, is) = h (ipol, k, is) + &
                            b (is, js, ks, ls) * ps2 (ipol, js, ks, ls)
                    enddo
                 enddo
              enddo
           enddo
        enddo
     endif
  enddo
  ! linear variation of the second term
  do is = 1, nspin0
     call fft_qgraddot (dfft, h (1, 1, is), xq, g, dh)
     do k = 1, dfft%nnr
        dvxc (k, is) = dvxc (k, is) - dh (k)
     enddo
  enddo
  IF (noncolin.AND.domag) THEN
     DO is=1,nspin0
        vgg(:,is)=dvxc(:,is)
     ENDDO
     dvxc=dvxcsave
     DO k=1,dfft%nnr
        dvxc(k,1)=dvxc(k,1)+0.5d0*(vgg(k,1)+vgg(k,2))
        amag=sqrt(rho(k,2)**2+rho(k,3)**2+rho(k,4)**2)
        IF (amag.GT.1.d-12) THEN
           DO is=2,4
              term=(0.0_dp,0.0_dp)
              DO jpol=2,4
                 term=term+rho(k,jpol)*drho(k,jpol)
              ENDDO
              term=term*rho(k,is)/amag**2
              dvxc(k,is)=dvxc(k,is)+0.5d0*segni(k)*((vgg(k,1)-vgg(k,2)) &
                                  *rho(k,is)+vsgga(k)*(drho(k,is)-term))/amag
           ENDDO
        ENDIF
     ENDDO
  ENDIF

  deallocate (dh)
  deallocate (h)
  deallocate (gdrho)
  deallocate (drhoout)
  if (noncolin.and.domag) then
     deallocate (gdmag)
     deallocate (dvxcsave)
     deallocate (vgg)
  endif

  return

end subroutine dgradcorr
!
!--------------------------------------------------------------------
subroutine dgradcorr_acc (dfft, nl, rho, grho, dvxc_rr, dvxc_sr, dvxc_ss, &
     dvxc_s, xq, drho, g, dvxc)
  !--------------------------------------------------------------------
  !! rtsc (RTSC_DVXC_GPU=1; groups/compute/SOLVE-LOG.md, wave 3B): dgradcorr on the device (OpenACC) for
  !! nspin = 1, collinear, not gamma-tricks. Same arithmetic as dgradcorr's nspin = 1 branch with
  !! fft_qgradient / fft_qgraddot inlined: drho and dvxc must be present on the device; the per-q GGA
  !! arrays are copied in. FFTs through fwfft/invfft on device data (cuFFT).
  !
  USE kinds,            ONLY : DP
  USE cell_base,        ONLY : tpiba
  USE fft_types,        ONLY : fft_type_descriptor
  USE fft_interfaces,   ONLY : fwfft, invfft
  IMPLICIT NONE
  TYPE(fft_type_descriptor), INTENT(IN) :: dfft
  INTEGER, INTENT(IN) :: nl(dfft%ngm)
  REAL(DP), INTENT(IN) :: rho(dfft%nnr), grho(3, dfft%nnr), dvxc_rr(dfft%nnr), dvxc_sr(dfft%nnr), &
       dvxc_ss(dfft%nnr), dvxc_s(dfft%nnr), xq(3), g(3, dfft%ngm)
  COMPLEX(DP), INTENT(IN) :: drho(dfft%nnr)
  COMPLEX(DP), INTENT(INOUT) :: dvxc(dfft%nnr)
  !
  real(DP), parameter :: epsr = 1.0d-6, epsg = 1.0d-10
  complex(DP), allocatable :: aux(:), dh(:), gd1(:), gd2(:), gd3(:), h1(:), h2(:), h3(:)
  real(DP) :: grho2, qx, tp
  complex(DP) :: s1, dr
  integer :: k, n, ipol, nnr, ngm
  !
  nnr = dfft%nnr
  ngm = dfft%ngm
  tp = tpiba
  allocate (aux(nnr), dh(nnr), gd1(nnr), gd2(nnr), gd3(nnr), h1(nnr), h2(nnr), h3(nnr))
  !$acc data present(drho, dvxc) present_or_copyin(g, nl) &
  !$acc&     copyin(rho, grho, dvxc_rr, dvxc_sr, dvxc_ss, dvxc_s) create(aux, dh, gd1, gd2, gd3, h1, h2, h3)
  !
  ! fft_qgradient: bring drho to G-space, multiply by i(q+G)_ipol, back to R-space, times tpiba
  !$acc parallel loop
  do k = 1, nnr
     aux(k) = drho(k)
  enddo
  !$acc host_data use_device(aux)
  CALL fwfft ('Rho', aux, dfft)
  !$acc end host_data
  do ipol = 1, 3
     qx = xq(ipol)
     !$acc parallel loop
     do k = 1, nnr
        dh(k) = (0.0_dp, 0.0_dp)
     enddo
     !$acc parallel loop
     do n = 1, ngm
        dh(nl(n)) = CMPLX(0.0_dp, qx + g(ipol, n), kind=DP) * aux(nl(n))
     enddo
     !$acc host_data use_device(dh)
     CALL invfft ('Rho', dh, dfft)
     !$acc end host_data
     IF (ipol == 1) THEN
        !$acc parallel loop
        do k = 1, nnr
           gd1(k) = dh(k) * tp
        enddo
     ELSEIF (ipol == 2) THEN
        !$acc parallel loop
        do k = 1, nnr
           gd2(k) = dh(k) * tp
        enddo
     ELSE
        !$acc parallel loop
        do k = 1, nnr
           gd3(k) = dh(k) * tp
        enddo
     ENDIF
  enddo
  !
  ! pointwise part (dgradcorr, nspin == 1)
  !$acc parallel loop private(grho2, s1, dr)
  do k = 1, nnr
     grho2 = grho(1, k)**2 + grho(2, k)**2 + grho(3, k)**2
     if (abs (rho (k) ) > epsr .and. grho2 > epsg) then
        s1 = grho (1, k) * gd1 (k) + &
             grho (2, k) * gd2 (k) + &
             grho (3, k) * gd3 (k)
        dr = drho (k)
        dvxc (k) = dvxc (k) + dvxc_rr (k) * dr &
             + dvxc_sr (k) * s1
        h1 (k) = (dvxc_sr(k) * dr + dvxc_ss(k) * s1 ) * grho(1, k) + dvxc_s (k) * gd1 (k)
        h2 (k) = (dvxc_sr(k) * dr + dvxc_ss(k) * s1 ) * grho(2, k) + dvxc_s (k) * gd2 (k)
        h3 (k) = (dvxc_sr(k) * dr + dvxc_ss(k) * s1 ) * grho(3, k) + dvxc_s (k) * gd3 (k)
     else
        h1 (k) = (0.d0, 0.d0)
        h2 (k) = (0.d0, 0.d0)
        h3 (k) = (0.d0, 0.d0)
     endif
  enddo
  !
  ! fft_qgraddot of h, subtracted from dvxc
  !$acc parallel loop
  do k = 1, nnr
     dh(k) = (0.0_dp, 0.0_dp)
  enddo
  do ipol = 1, 3
     qx = xq(ipol)
     IF (ipol == 1) THEN
        !$acc parallel loop
        do k = 1, nnr
           aux(k) = h1(k)
        enddo
     ELSEIF (ipol == 2) THEN
        !$acc parallel loop
        do k = 1, nnr
           aux(k) = h2(k)
        enddo
     ELSE
        !$acc parallel loop
        do k = 1, nnr
           aux(k) = h3(k)
        enddo
     ENDIF
     !$acc host_data use_device(aux)
     CALL fwfft ('Rho', aux, dfft)
     !$acc end host_data
     !$acc parallel loop
     do n = 1, ngm
        dh(nl(n)) = dh(nl(n)) + CMPLX(0.0_dp, qx + g(ipol, n), kind=DP) * aux(nl(n))
     enddo
  enddo
  !$acc host_data use_device(dh)
  CALL invfft ('Rho', dh, dfft)
  !$acc end host_data
  !$acc parallel loop
  do k = 1, nnr
     dvxc(k) = dvxc(k) - dh(k) * tp
  enddo
  !$acc end data
  deallocate (aux, dh, gd1, gd2, gd3, h1, h2, h3)
  !
  return
end subroutine dgradcorr_acc
