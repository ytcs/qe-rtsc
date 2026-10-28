#!/usr/bin/env bash
# nsys profile of ph.x on one q-point (PH-DESIGN Step 0). Run scf unprofiled, then trace a window of ph.x.
#   profile.sh <inputs dir> <iq> <bin dir> <workdir> [delay s] [duration s] [np]
# Use a bin dir built with -DQE_ENABLE_PROFILE_NVTX=ON so QE's start_clock/stop_clock become NVTX ranges.
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
IN=$(realpath "$1"); IQ=$2; BIN=$(realpath "$3"); W=$4; DELAY=${5:-60}; DUR=${6:-60}; NP=${7:-1}
mkdir -p "$W"; W=$(realpath "$W"); cd "$W"
PSEUDO_DIR=$(realpath "${PSEUDO_DIR:-$HERE/../pseudo/sg15}")
cp "$IN"/scf.in "$IN"/ph.in .; sed -i "s|@PSEUDO@|$PSEUDO_DIR|" *.in
sed -i "s|^/|  start_q = $IQ\n  last_q = $IQ\n/|" ph.in
export OMP_NUM_THREADS=1
mpirun --bind-to none -np "$NP" "$BIN"/pw.x -in scf.in > scf.out
nsys profile -o ph-q$IQ --force-overwrite true -t cuda,nvtx,openacc,osrt --cuda-memory-usage=false \
     --delay "$DELAY" --duration "$DUR" mpirun --bind-to none -np "$NP" "$BIN"/ph.x -in ph.in > ph.out 2>&1 || true
nsys stats -q --format csv -r nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
     -o stats ph-q$IQ.nsys-rep > /dev/null
grep -E "iter #|av.it" ph.out | tail -3
command ls stats_*.csv
