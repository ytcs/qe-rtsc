#!/usr/bin/env bash
# Vanilla configuration sweep on one ph.x q-point (MILESTONE-1: baseline = vanilla's best config).
#   sweep.sh <inputs dir> <iq> <outdir> "label|bindir|np|nt|pool flags|mps(0/1)" ...
# For each config: scf + ph.x restricted to q-point <iq> (start_q = last_q = iq). Prints label, scf s, ph s.
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
IN=$(realpath "$1"); IQ=$2; OUT=$3; shift 3
mkdir -p "$OUT"; OUT=$(realpath "$OUT")
PSEUDO_DIR=$(realpath "${PSEUDO_DIR:-$HERE/../pseudo/sg15}")
for c in "$@"; do
  IFS='|' read -r label bin np nt pool mps <<< "$c"
  W=$OUT/$label; rm -rf "$W"; mkdir -p "$W"; cd "$W"
  cp "$IN"/scf.in "$IN"/ph.in .; sed -i "s|@PSEUDO@|$PSEUDO_DIR|" *.in
  sed -i "s|^/|  start_q = $IQ\n  last_q = $IQ\n/|" ph.in
  export OMP_NUM_THREADS=$nt
  if [ "$mps" = 1 ]; then nvidia-cuda-mps-control -d; fi
  t0=$(date +%s.%N); mpirun --bind-to none -np "$np" "$bin"/pw.x $pool -in scf.in > scf.out
  t1=$(date +%s.%N); mpirun --bind-to none -np "$np" "$bin"/ph.x $pool -in ph.in > ph.out
  t2=$(date +%s.%N)
  if [ "$mps" = 1 ]; then echo quit | nvidia-cuda-mps-control; fi
  awk -v l="$label" -v a="$t0" -v b="$t1" -v c="$t2" 'BEGIN{printf "%s\t%.1f\t%.1f\n", l, b-a, c-b}' | tee -a "$OUT"/sweep.tsv
done
