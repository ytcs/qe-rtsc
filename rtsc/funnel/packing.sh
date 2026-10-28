#!/usr/bin/env bash
# F0 packing probe (FUNNEL.md section 4 (b)): two T3a-style jobs (B2 smoke, MP) concurrently under MPS vs serially.
#   packing.sh <bench inputs dir> <workroot> [NT, default 4] [NE epw ranks, default 4]
# Both arms use the same NT/NE, so CPU oversubscription does not masquerade as a GPU-packing loss.
# MPS is started once for the session (run-fork.sh would quit it at the end of its EPW stage).
set -uo pipefail
IN=$(realpath "$1"); W=$2; NT=${3:-4}; NE=${4:-4}
BIN=${BIN:-/root/rtsc/qe-rtsc/build-gpu/bin}
PSEUDO_DIR=${PSEUDO_DIR:-/root/rtsc/qe-rtsc/rtsc/pseudo/sg15}
source /root/rtsc/env.sh
mkdir -p "$W"; W=$(realpath "$W")
export RTSC_STH=1 RTSC_STRICT=1 RTSC_STH_MP=1
nvidia-cuda-mps-control -d 2>/dev/null || true
job() {  # job <dir>
  local d=$1; mkdir -p "$d"; cd "$d"; cp "$IN"/*.in .; sed -i "s|@PSEUDO@|$PSEUDO_DIR|" *.in
  local prefix; prefix=$(grep -oP "prefix\s*=\s*'\K[^']+" scf.in)
  local t0; t0=$(date +%s.%N)
  OMP_NUM_THREADS=$NT mpirun --bind-to none -np 1 -x RTSC_STH -x RTSC_STRICT -x RTSC_STH_MP -x OMP_NUM_THREADS "$BIN"/pw.x -in scf.in > scf.out
  OMP_NUM_THREADS=$NT mpirun --bind-to none -np 1 -x RTSC_STH -x RTSC_STRICT -x RTSC_STH_MP -x OMP_NUM_THREADS "$BIN"/ph.x -in ph.in > ph.out
  mkdir -p save; local nq; nq=$(sed -n 2p $prefix.dyn0 | awk '{print $1}')
  for i in $(seq 1 $nq); do cp $prefix.dyn$i save/$prefix.dyn_q$i
    if [ $i -eq 1 ]; then cp _ph0/$prefix.dvscf1 save/$prefix.dvscf_q1; cp -r _ph0/$prefix.phsave save/
    else cp _ph0/$prefix.q_$i/$prefix.dvscf1 save/$prefix.dvscf_q$i; fi; done
  OMP_NUM_THREADS=$NT mpirun --bind-to none -np 1 -x OMP_NUM_THREADS "$BIN"/pw.x -in nscf.in > nscf.out
  OMP_NUM_THREADS=1 mpirun --bind-to none -np $NE -x OMP_NUM_THREADS "$BIN"/epw.x -npool $NE -in epw.in > epw.out
  awk -v a="$t0" -v b="$(date +%s.%N)" -v d="$d" 'BEGIN{printf "%s\t%.2f\n", d, b-a}'
}
T0=$(date +%s.%N)
( job "$W/serial-A" ); ( job "$W/serial-B" )
T1=$(date +%s.%N)
( job "$W/conc-A" ) & ( job "$W/conc-B" ) & wait
T2=$(date +%s.%N)
awk -v a="$T0" -v b="$T1" -v c="$T2" 'BEGIN{printf "serial makespan\t%.2f\nconcurrent makespan\t%.2f\npacking gain\t%.3f\n", b-a, c-b, (b-a)/(c-b)}'
echo quit | nvidia-cuda-mps-control 2>/dev/null || true
