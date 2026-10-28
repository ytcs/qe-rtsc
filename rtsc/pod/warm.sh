#!/usr/bin/env bash
# Warm start for a dev pod (RULES.md: default until the campaign closes).
# Expects the global volume rtsc-build-cache mounted at $CACHE (default /cache).
#   curl -sSfL https://raw.githubusercontent.com/ytcs/qe-rtsc/rtsc/rtsc/pod/warm.sh | bash
# Cache hit: restore source + build-gpu (NVTX on) + pseudo + shim, git pull, incremental rebuild.
# Cache miss: full setup + build, then save to the cache.
set -euo pipefail
CACHE=${CACHE:-/cache}; J=${J:-28}; CC=${CC:-80}; ARC=$CACHE/rtsc-dev-sm$CC.tar.zst
NV=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq zstd curl git libfftw3-dev pkg-config cmake ninja-build python3 python3-numpy >/dev/null
mkdir -p /root/rtsc && cd /root/rtsc
cat > env.sh <<EOI
NV=$NV
export PATH=\$NV/comm_libs/mpi/bin:\$NV/compilers/bin:\$NV/profilers/13.2/Nsight_Systems/bin:\$PATH
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export LIBRARY_PATH=/root/rtsc/nvtxshim:\${LIBRARY_PATH:-} LD_LIBRARY_PATH=/root/rtsc/nvtxshim:\${LD_LIBRARY_PATH:-}
EOI
if [ -f "$ARC" ]; then
  echo "warm: restoring $(du -h "$ARC" | cut -f1) from cache"
  tar -I 'zstd -d -T0' -xf "$ARC" -C /root/rtsc
  cd qe-rtsc && git fetch -q origin && git checkout -q "${BRANCH:-rtsc}" && git pull -q --ff-only origin "${BRANCH:-rtsc}" && cd ..
else
  echo "warm: cache miss, full setup"
  [ -d qe-rtsc ] || git clone -q -b "${BRANCH:-rtsc}" https://github.com/ytcs/qe-rtsc.git
  (cd qe-rtsc && git submodule update -q --init external/wannier90 external/mbd external/fox)
  qe-rtsc/rtsc/pseudo.sh
fi
source env.sh
NVTX=1 CPU=0 J=$J qe-rtsc/rtsc/pod/build-gpu.sh qe-rtsc $CC
[ -f "$ARC" ] || CC=$CC qe-rtsc/rtsc/pod/save.sh
echo WARM_OK
