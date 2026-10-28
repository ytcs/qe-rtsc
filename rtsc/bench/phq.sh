#!/usr/bin/env bash
# ph.x alone on a finished scf (dev runs; the scored pipeline is run-fork.sh).
#   phq.sh <scf dir> <run dir> <bin dir> [start_q last_q]
# Copies the scf inputs and <prefix>.save into <run dir>, optionally restricts ph.in to
# start_q..last_q, and runs 1 rank x OMP_NUM_THREADS (default 8) with RTSC_STH=1 RTSC_STRICT=1 and
# whatever RTSC_* the caller exports (e.g. RTSC_STH_MP=1). The ph.x wall time, measured inside
# flock /root/rtsc/TIMING.lock, goes to <run dir>/ph.time.
set -euo pipefail
S=$(realpath "$1"); W=$2; BIN=$(realpath "$3"); Q0=${4:-}; Q1=${5:-}
mkdir -p "$W"; W=$(realpath "$W"); cd "$W"
cp "$S"/*.in .
prefix=$(grep -oP "prefix\s*=\s*'\K[^']+" scf.in)
rm -rf "$prefix.save"; cp -r "$S/$prefix.save" .
if [ -n "$Q0" ]; then
  sed -i "s|^/|  start_q = $Q0\n  last_q = $Q1\n/|" ph.in
fi
export RTSC_STH=${RTSC_STH:-1} RTSC_STRICT=${RTSC_STRICT:-1} OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
flock /root/rtsc/TIMING.lock bash -c "t0=\$(date +%s.%N); mpirun --bind-to none -np 1 \
  \$(env | grep -o '^RTSC_[A-Z0-9_]*' | sed 's/^/-x /') -x OMP_NUM_THREADS $BIN/ph.x -in ph.in > ph.out; \
  t1=\$(date +%s.%N); awk -v a=\$t0 -v b=\$t1 'BEGIN{printf \"%.2f\n\", b-a}' > ph.time"
echo "$W: ph.x $(cat ph.time) s"
