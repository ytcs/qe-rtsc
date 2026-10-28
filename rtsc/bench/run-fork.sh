#!/usr/bin/env bash
# Scored end-to-end fork timing (MILESTONE-1): same stages as run.sh, fork configuration per stage.
#   run-fork.sh <inputs dir> <bin dir> <workdir> [omp threads for pw/ph] [epw ranks]
# pw/ph: 1 MPI rank x NT OpenMP threads on the GPU, RTSC_STH=1 (engine, GPU tail, batched nscf),
#        RTSC_STRICT=1 (an engine decline or unsupported case stops ph.x instead of running vanilla).
# epw:   NE ranks + MPS, -npool NE (RTSC_EPW_* defaults: fused fine interpolation, low-rank Eliashberg).
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
IN=$(realpath "$1"); BIN=$(realpath "$2"); W=$3; NT=${4:-8}; NE=${5:-12}
mkdir -p "$W"; W=$(realpath "$W"); cd "$W"
cp "$IN"/*.in .
PSEUDO_DIR=$(realpath "${PSEUDO_DIR:-$HERE/../pseudo/sg15}")
sed -i "s|@PSEUDO@|$PSEUDO_DIR|" *.in
export RTSC_STH=1 RTSC_STRICT=1   # strict: an engine decline stops ph.x (no silent vanilla path)
prefix=$(grep -oP "prefix\s*=\s*'\K[^']+" scf.in)
: > timings.tsv
stage() { local name=$1; shift; local t0=$(date +%s.%N); "$@"; local t1=$(date +%s.%N)
          awk -v n="$name" -v a="$t0" -v b="$t1" 'BEGIN{printf "%s\t%.2f\n", n, b-a}' | tee -a timings.tsv; }
T0=$(date +%s.%N)
stage scf  env OMP_NUM_THREADS=$NT mpirun --bind-to none -np 1 -x RTSC_STH -x RTSC_STRICT -x OMP_NUM_THREADS "$BIN"/pw.x -in scf.in > scf.out
stage ph   env OMP_NUM_THREADS=$NT mpirun --bind-to none -np 1 -x RTSC_STH -x RTSC_STRICT -x OMP_NUM_THREADS "$BIN"/ph.x -in ph.in > ph.out
stage collect bash -c "
  mkdir -p save; nq=\$(sed -n 2p $prefix.dyn0 | awk '{print \$1}')
  for i in \$(seq 1 \$nq); do
    cp $prefix.dyn\$i save/$prefix.dyn_q\$i
    if [ \$i -eq 1 ]; then cp _ph0/$prefix.dvscf1 save/$prefix.dvscf_q1; cp -r _ph0/$prefix.phsave save/
    else cp _ph0/$prefix.q_\$i/$prefix.dvscf1 save/$prefix.dvscf_q\$i; fi
  done"
stage nscf env OMP_NUM_THREADS=$NT mpirun --bind-to none -np 1 -x OMP_NUM_THREADS "$BIN"/pw.x -in nscf.in > nscf.out
stage epw  bash -c "nvidia-cuda-mps-control -d 2>/dev/null || true; OMP_NUM_THREADS=1 mpirun --bind-to none -np $NE -x OMP_NUM_THREADS $BIN/epw.x -npool $NE -in epw.in > epw.out; echo quit | nvidia-cuda-mps-control 2>/dev/null || true"
awk -v a="$T0" -v b="$(date +%s.%N)" 'BEGIN{printf "total\t%.2f\n", b-a}' | tee -a timings.tsv
