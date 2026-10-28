#!/usr/bin/env bash
# Timed T_c pipeline (MILESTONE-1.md). Usage:
#   run.sh <inputs dir> <bin dir> <workdir> [mpi ranks] [omp threads] [pool flags]
# Writes <workdir>/timings.tsv: stage, seconds.
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
IN=$(realpath "$1"); BIN=$(realpath "$2"); W=$3; NP=${4:-1}; NT=${5:-1}; POOL=${6:-}
mkdir -p "$W"; W=$(realpath "$W"); cd "$W"
cp "$IN"/*.in .
PSEUDO_DIR=$(realpath "${PSEUDO_DIR:-$HERE/../pseudo/sg15}")
sed -i "s|@PSEUDO@|$PSEUDO_DIR|" *.in
export OMP_NUM_THREADS=$NT
MPI="mpirun --bind-to none -np $NP"
prefix=$(grep -oP "prefix\s*=\s*'\K[^']+" scf.in)
: > timings.tsv
stage() { local name=$1; shift; local t0=$(date +%s.%N); "$@"; local t1=$(date +%s.%N)
          awk -v n="$name" -v a="$t0" -v b="$t1" 'BEGIN{printf "%s\t%.2f\n", n, b-a}' | tee -a timings.tsv; }
T0=$(date +%s.%N)
stage scf  bash -c "$MPI $BIN/pw.x $POOL -in scf.in > scf.out"
stage ph   bash -c "$MPI $BIN/ph.x $POOL -in ph.in > ph.out"
stage collect bash -c "
  mkdir -p save; nq=\$(sed -n 2p $prefix.dyn0 | awk '{print \$1}')
  for i in \$(seq 1 \$nq); do
    cp $prefix.dyn\$i save/$prefix.dyn_q\$i
    if [ \$i -eq 1 ]; then cp _ph0/$prefix.dvscf1 save/$prefix.dvscf_q1; cp -r _ph0/$prefix.phsave save/
    else cp _ph0/$prefix.q_\$i/$prefix.dvscf1 save/$prefix.dvscf_q\$i; fi
  done"
stage nscf bash -c "$MPI $BIN/pw.x $POOL -in nscf.in > nscf.out"
stage epw  bash -c "$MPI $BIN/epw.x -npool $NP -in epw.in > epw.out"
awk -v a="$T0" -v b="$(date +%s.%N)" 'BEGIN{printf "total\t%.2f\n", b-a}' | tee -a timings.tsv
