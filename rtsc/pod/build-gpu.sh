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
# NVTX=1: QE timers become NVTX ranges (nsys). CUDA 13 ships NVTX3 headers only, so build a v2-ABI shim.
gpuflags=()
if [ "${NVTX:-0}" = 1 ]; then
  SHIM=/root/rtsc/nvtxshim
  if [ ! -f $SHIM/libnvToolsExt.so ]; then
    mkdir -p $SHIM && I=$NV/cuda/13.2/targets/x86_64-linux/include
    printf '#define NVTX_EXPORT_API\n#include <nvtx3/nvToolsExt.h>\n' > $SHIM/shim.c
    gcc -shared -fPIC -I$I $SHIM/shim.c -o $SHIM/libnvToolsExt.so -ldl
  fi
  export LIBRARY_PATH=$SHIM:${LIBRARY_PATH:-} LD_LIBRARY_PATH=$SHIM:${LD_LIBRARY_PATH:-}
  gpuflags+=(-DQE_ENABLE_PROFILE_NVTX=ON)
fi
# MDX/CUTLASS set (rtsc/sth/fdx/deps.sh): the engine's cuFFTDx path (RTSC_STH_FFTDX) is built, else a stub
if [ -n "${MDX:-}" ] && [ -n "${CUTLASS:-}" ]; then gpuflags+=(-DRTSC_MATHDX_INC=$MDX -DRTSC_CUTLASS_INC=$CUTLASS); fi
cmake -S "$SRC" -B "$SRC/build-gpu" "${common[@]}" "${gpuflags[@]}" -DQE_GPU_ARCHS=sm_$CC -DQE_GPU="openacc;cuda" \
      -DQE_ENABLE_MPI_GPU_AWARE=ON
cmake --build "$SRC/build-gpu" -j "$J" --target pw.x ph.x epw.x q2r.x matdyn.x
[ "${CPU:-1}" = 1 ] || exit 0   # CPU=0: GPU build only (dev pod)
cmake -S "$SRC" -B "$SRC/build-cpu" "${common[@]}"
cmake --build "$SRC/build-cpu" -j "$J" --target pw.x ph.x epw.x q2r.x matdyn.x
