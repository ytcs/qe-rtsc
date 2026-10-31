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
# kernel branch (groups/compute/KERNEL-LOG.md): dV_bare psi built in the engine (class E, gated on the B1/B2 smoke
# chains), and wannier90's disentanglement over NE OpenMP threads on EPW's ionode (bitwise identical)
export RTSC_STH_DVB=${RTSC_STH_DVB:-1} RTSC_W90_NT=${RTSC_W90_NT:-$NE}
prefix=$(grep -oP "prefix\s*=\s*'\K[^']+" scf.in)
: > timings.tsv
stage() { local name=$1; shift; local t0=$(date +%s.%N); "$@"; local t1=$(date +%s.%N)
          awk -v n="$name" -v a="$t0" -v b="$t1" 'BEGIN{printf "%s\t%.2f\n", n, b-a}' | tee -a timings.tsv; }
T0=$(date +%s.%N)
# every RTSC_* switch of the caller (e.g. RTSC_STH_MP=1, mixed-precision engine) reaches pw.x / ph.x
XR=$(env | grep -o '^RTSC_[A-Z0-9_]*' | sed 's/^/-x /' | tr '\n' ' ')
# epw4 (groups/compute/EPW4-LOG.md), both default off, outputs bitwise identical:
#  RTSC_RUN_DIRECT=1  the 1-rank pw.x stages (scf, nscf) run without mpirun (Open MPI singleton; ≈ 1 s less start-up each)
#  RTSC_RUN_PREWARM=1 epw.x (and the MPS daemon) start when nscf starts; the ranks finish MPI_Init and their CUDA
#                     contexts, then wait for the go file (RTSC_EPW_GOFILE) that is touched when nscf ends
pw1() { if [ "${RTSC_RUN_DIRECT:-0}" = 1 ]; then env OMP_NUM_THREADS=$NT "$BIN"/pw.x "$@"
        else env OMP_NUM_THREADS=$NT mpirun --bind-to none -np 1 $XR -x OMP_NUM_THREADS "$BIN"/pw.x "$@"; fi; }
runepw() { nvidia-cuda-mps-control -d 2>/dev/null || true
  OMP_NUM_THREADS=1 mpirun --bind-to none -np $NE -x OMP_NUM_THREADS $XR $EXR "$BIN"/epw.x -npool $NE -in epw.in > epw.out
  echo quit | nvidia-cuda-mps-control 2>/dev/null || true; }
EXR=""
stage scf  pw1 -in scf.in > scf.out
stage ph   env OMP_NUM_THREADS=$NT mpirun --bind-to none -np 1 $XR -x OMP_NUM_THREADS "$BIN"/ph.x -in ph.in > ph.out
stage collect bash -c "
  mkdir -p save; nq=\$(sed -n 2p $prefix.dyn0 | awk '{print \$1}')
  for i in \$(seq 1 \$nq); do
    cp $prefix.dyn\$i save/$prefix.dyn_q\$i
    if [ \$i -eq 1 ]; then cp _ph0/$prefix.dvscf1 save/$prefix.dvscf_q1; cp -r _ph0/$prefix.phsave save/
    else cp _ph0/$prefix.q_\$i/$prefix.dvscf1 save/$prefix.dvscf_q\$i; fi
  done"
if [ "${RTSC_RUN_PREWARM:-0}" = 1 ]; then
  rm -f "$W/.epwgo"; export RTSC_EPW_GOFILE="$W/.epwgo"; EXR="-x RTSC_EPW_GOFILE"
  runepw & EPWPID=$!
  trap 'pkill -P $EPWPID 2>/dev/null; kill $EPWPID 2>/dev/null' EXIT   # a failed nscf must not leave epw.x waiting
  stage nscf pw1 -in nscf.in > nscf.out
  stage epw  bash -c "touch $W/.epwgo; while kill -0 $EPWPID 2>/dev/null; do sleep 0.02; done"
  wait $EPWPID; trap - EXIT
else
  stage nscf pw1 -in nscf.in > nscf.out
  stage epw  runepw
fi
awk -v a="$T0" -v b="$(date +%s.%N)" 'BEGIN{printf "total\t%.2f\n", b-a}' | tee -a timings.tsv
