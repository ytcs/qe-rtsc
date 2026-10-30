#!/usr/bin/env bash
# Build the cuFFTDx microbenchmark (FFT-LOG.md): rtsc/sth/fdx/build.sh [sm, e.g. 86 or 90] -> build/fdxbench
# Needs NVHPC 26.5 (nvfortran, nvcc), MathDx (pip nvidia-mathdx 25.6) and CUTLASS headers:
#   MDX=<.../nvidia/mathdx/include> CUTLASS=<cutlass/include>
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
SM=${1:-86}
NV=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
export PATH=$NV/cuda/13.2/bin:$NV/compilers/bin:$PATH
MDX=${MDX:-/root/mdx/lib/python3.12/site-packages/nvidia/mathdx/include}
CUTLASS=${CUTLASS:-/root/cutlass/include}
B=build${EMU:+-emu}; mkdir -p $B && cd $B
nvfortran -O3 -cuda -gpu=cc$SM -c ../../sth_pfft.f90 ../../sth_pfft_s.f90 ../pfwrap.f90
nvcc -std=c++17 -O3 -arch=sm_$SM -DFDX_SM=${SM}0 ${EMU:+-DFDX_EMU} -I$MDX -I$CUTLASS -Xptxas -v -c ../sth_fftdx.cu -o sth_fftdx.o 2> ptxas.txt || { cat ptxas.txt; exit 1; }
grep -E "spill" ptxas.txt | grep -v " 0 bytes spill stores, 0 bytes spill loads" || true
nvcc -std=c++17 -O3 -arch=sm_$SM ${EMU:+-DFDX_EMU} -I$NV/math_libs/13.2/include -c ../fdxbench.cu -o fdxbench.o
nvfortran -cuda -gpu=cc$SM -cudalib=cufft -c++libs -Mnomain fdxbench.o sth_fftdx.o pfwrap.o sth_pfft.o sth_pfft_s.o -o fdxbench
echo "built $(pwd)/fdxbench"
nvcc -O3 -arch=sm_$SM -I$NV/math_libs/13.2/include ../gemmbench.cu -o gemmbench -L$NV/math_libs/13.2/lib64 -Xlinker -rpath=$NV/math_libs/13.2/lib64 -lcublas && echo "built $(pwd)/gemmbench"
