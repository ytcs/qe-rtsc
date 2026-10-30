set(sources
    adddvscf.f90
    adddvhubscf.f90
    addusdbec.f90
    addusdbec_nc.f90
    apply_dpot.f90
    apply_dpot_mod.f90
    apply_trev.f90
    non_scf_ph.f90
    c_bands_ph.f90
    addusldos.f90
    adddvepsi_us.f90
    cft_wave.f90
    cgsolve_all.f90
    cg_psi.f90
    ch_psi_all.f90
    Coul_cut_2D_ph.f90
    check_q_points_sym.f90
    dnsq_scf.f90
    dfpt_tetra_mod.f90
    dfpt_type.f90
    dynmat_sub.f90
    h_prec.f90
    h_prec_freq.f90
    incdrhoscf_mod.f90
    dv_of_drho.f90
    dgradcorr.f90
    dnonloccorr.f90
    dv_vdW_DF.f90
    dv_rVV10.f90
    newdq.f90
    orthogonalize.f90
    orthogonalize_omega.f90
    setup_nscf.f90
    set_dbecsum_nc.f90
    set_int3_nc.f90
    smallgq.f90
    lr_sm1_psi.f90
    lr_addusddens.f90
    lr_two_chem.f90
    twochem_postproc_dfpt.f90
    localdos.f90
    setup_dgc.f90
    compute_vsgga.f90
    setup_dmuxc.f90
    setup_nbnd_occ.f90
    setup_alpha_pv.f90
    set_small_group_of_q.f90
    sgam_lr.f90
    star_q.f90
    setqmod.f90
    lr_orthoUwfc.f90
    qdipol_cryst.f90
    mode_group.f90
    mix_pot.f90
    rtsc_mix.f90
    lrcom.f90
    qpoint.f90
    lr_dot.f90
    lr_dot_magnons.f90
    lr_nc_mag.f90
    lr_restart.f90
    lanczos_pseudohermitian.f90
    lanczos_nonhermitian.f90
    lanczos_pseudohermitian_c.f90
    lanczos_nonhermitian_c.f90
    set_kplusq_nc.f90
    lr_sym_mod.f90
    lr_sym_paw.f90
    sym_def.f90
    sym_dns.f90
    psymdvscf.f90
    symdvscf.f90
    sym_dmag.f90
    ccg_psi.f90
    ccgsolve_all.f90
    ch_psi_all_complex.f90
    sth_dump.f90
    sth_tail.f90
    sth_batched.f90
    sth_nscf_start.f90
    sth_nscf_batched.f90
    response_kernels.f90
    efermi_shift.f90
    dfpt_kernels.f90)
if(QE_ENABLE_CUDA)
    # rtsc: batched Sternheimer engine (CUDA Fortran, cuFFT, cuBLAS); used by sth_batched.f90
    list(PREPEND sources ${CMAKE_CURRENT_SOURCE_DIR}/../rtsc/sth/sth_pfft.f90
                         ${CMAKE_CURRENT_SOURCE_DIR}/../rtsc/sth/sth_pfft_s.f90
                         ${CMAKE_CURRENT_SOURCE_DIR}/../rtsc/sth/sth_engine.f90
                         ${CMAKE_CURRENT_SOURCE_DIR}/../rtsc/sth/nscf_engine.f90)
endif()
qe_enable_cuda_fortran("${sources}")
if(QE_ENABLE_CUDA)
    # rtsc: cuFFTDx local operator (rtsc/sth/fdx/sth_fftdx.cu; groups/compute/FFT-LOG.md). The real kernels need the
    # MathDx and CUTLASS headers (rtsc/sth/fdx/deps.sh); without them a stub whose fdx_setup reports "unsupported".
    set(RTSC_MATHDX_INC "" CACHE PATH "MathDx include dir (cufftdx.hpp) for the fftdx engine path")
    set(RTSC_CUTLASS_INC "" CACHE PATH "CUTLASS include dir for the fftdx engine path")
    if(RTSC_MATHDX_INC AND RTSC_CUTLASS_INC)
        find_program(RTSC_NVCC nvcc HINTS ${CUDAToolkit_BIN_DIR} REQUIRED)
        list(GET QE_GPU_ARCHS 0 _fdx_arch)
        string(REGEX REPLACE "[^0-9]" "" _fdx_sm "${_fdx_arch}")
        set(_fdx_src ${CMAKE_CURRENT_SOURCE_DIR}/../rtsc/sth/fdx/sth_fftdx.cu)
        set(_fdx_obj ${CMAKE_CURRENT_BINARY_DIR}/sth_fftdx.o)
        add_custom_command(OUTPUT ${_fdx_obj}
            COMMAND ${RTSC_NVCC} -std=c++17 -O3 -arch=sm_${_fdx_sm} -DFDX_SM=${_fdx_sm}0 -Xcompiler -fPIC
                    -I${RTSC_MATHDX_INC} -I${RTSC_CUTLASS_INC} -c ${_fdx_src} -o ${_fdx_obj}
            DEPENDS ${_fdx_src} VERBATIM)
        set_source_files_properties(${_fdx_obj} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
        list(APPEND sources ${_fdx_obj})
        message(STATUS "rtsc fftdx: cuFFTDx kernels for sm_${_fdx_sm}")
    else()
        list(APPEND sources ${CMAKE_CURRENT_SOURCE_DIR}/../rtsc/sth/fdx/sth_fftdx_stub.c)
        message(STATUS "rtsc fftdx: stub (set RTSC_MATHDX_INC and RTSC_CUTLASS_INC for the cuFFTDx kernels)")
    endif()
endif()
qe_add_library(qe_lr_modules ${sources})
target_link_libraries(qe_lr_modules
    PRIVATE
        qe_openacc_fortran
	qe_openmp_fortran
	qe_mpi_fortran
        qe_modules
        qe_pw
        qe_upflib
        qe_fftx
        qe_xclib
	qe_device_lapack)

if(QE_ENABLE_CUDA)
    target_link_libraries(qe_modules
        PRIVATE
            CUDA::cublas)
    target_link_libraries(qe_lr_modules
        PRIVATE
            CUDA::cufft
            CUDA::cublas
            stdc++)
endif()

###########################################################

qe_install_targets(qe_lr_modules)
