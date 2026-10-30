#!/usr/bin/env bash
# MathDx (cuFFTDx) and CUTLASS headers for the fftdx engine path (FFT-LOG.md). Prints the two include dirs.
#   eval "$(rtsc/sth/fdx/deps.sh)"   -> MDX=..., CUTLASS=...
set -euo pipefail
D=${FDX_DEPS:-/root/rtsc/fdxdeps}
mkdir -p $D
if [ ! -f $D/mdx/lib/python3.12/site-packages/nvidia/mathdx/include/cufftdx.hpp ]; then
  python3 -m venv $D/mdx >&2 && $D/mdx/bin/pip install -q nvidia-mathdx==25.6.0 >&2
fi
[ -d $D/cutlass/include ] || git clone -q --depth 1 -b v3.9.2 https://github.com/NVIDIA/cutlass.git $D/cutlass >&2
echo "export MDX=$(ls -d $D/mdx/lib/python3*/site-packages/nvidia/mathdx/include) CUTLASS=$D/cutlass/include"
