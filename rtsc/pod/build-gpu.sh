#!/usr/bin/env bash
# Build a QE tree on the RunPod NVHPC image: GPU (OpenACC + CUDA Fortran) and CPU (nvfortran) variants.
#   build-gpu.sh <src dir> <cuda cc, e.g. 80 for A100, 90 for H100>
set -euo pipefail
SRC=$(realpath "$1"); CC=${2:-80}
NV=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
export PATH=$NV/compilers/bin:$PATH
J=${J:-12}   # cgroup quota is ~12 CPUs; nproc reports the host
export PATH=$NV/comm_libs/mpi/bin:$PATH
command -v pkg-config >/dev/null || (apt-get update && apt-get install -y libfftw3-dev pkg-config)
common=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=nvc -DCMAKE_Fortran_COMPILER=mpif90
        -DQE_ENABLE_OPENMP=ON -DQE_ENABLE_SCALAPACK=OFF -DBLA_VENDOR=NVHPC
        -DQE_FFTW_VENDOR=Internal)   # system libfftw3_omp drags in libgomp, which aborts under nvfortran OpenMP
cmake -S "$SRC" -B "$SRC/build-gpu" "${common[@]}" -DQE_GPU_ARCHS=sm_$CC -DQE_GPU="openacc;cuda" \
      -DQE_ENABLE_MPI_GPU_AWARE=ON
cmake --build "$SRC/build-gpu" -j "$J" --target pw.x ph.x epw.x q2r.x matdyn.x
[ "${CPU:-1}" = 1 ] || exit 0   # CPU=0: GPU build only (dev pod)
cmake -S "$SRC" -B "$SRC/build-cpu" "${common[@]}"
cmake --build "$SRC/build-cpu" -j "$J" --target pw.x ph.x epw.x q2r.x matdyn.x
