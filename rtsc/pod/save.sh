#!/usr/bin/env bash
# Save the dev tree (source, build-gpu objects and binaries, pseudo, NVTX shim) to the build cache.
# Run at the end of a dev session so the next pod starts warm. Runs and data are not saved.
set -euo pipefail
CACHE=${CACHE:-/cache}; CC=${CC:-80}; ARC=$CACHE/rtsc-dev-sm$CC.tar.zst
cd /root/rtsc
tar -I 'zstd -3 -T0' -cf "$ARC.tmp" --exclude='qe-rtsc/build-cpu' qe-rtsc nvtxshim
mv "$ARC.tmp" "$ARC" && echo "saved $(du -h "$ARC" | cut -f1) to $ARC"
