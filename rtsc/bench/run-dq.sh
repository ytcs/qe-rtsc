#!/usr/bin/env bash
# Fork pipeline with dvscf interpolation (groups/compute/DQ-LOG.md): the stages of run-fork.sh, with
# ph.x split into a self-consistent DFPT on a coarse q-subgrid and a one-shot pass on the full grid.
#   run-dq.sh <inputs dir> <bin dir> <workdir> <nsub> [omp threads for pw/ph] [epw ranks]
#   scf    pw.x scf (as run-fork.sh)
#   ph1    ph.x, nq = nsub^3 (full SCF on the batched engine) in <workdir>/sub
#   q2r    dvscf_q2r.x on the subgrid dvscf -> <workdir>/sub/w_pot (no long-range part: metals)
#   ph2    ph.x on the full grid with RTSC_DQ=1: q /= Gamma one Sternheimer pass from the interpolated
#          dvscf (PH/rtsc_dq.f90), Gamma self-consistent
#   collect, nscf, epw as run-fork.sh
# FROM=<stage> skips the stages before it (dev reruns; not a timing).
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
IN=$(realpath "$1"); BIN=$(realpath "$2"); W=$3; NS=$4; NT=${5:-8}; NE=${6:-12}
mkdir -p "$W"; W=$(realpath "$W"); cd "$W"
FROM=${FROM:-scf}
if [ "$FROM" = scf ]; then
  cp "$IN"/*.in .
  PSEUDO_DIR=$(realpath "${PSEUDO_DIR:-$HERE/../pseudo/sg15}")
  sed -i "s|@PSEUDO@|$PSEUDO_DIR|" *.in
  : > timings.tsv
fi
export RTSC_STH=1 RTSC_STRICT=1
prefix=$(grep -oP "prefix\s*=\s*'\K[^']+" scf.in)
on=0
stage() { local name=$1; shift; [ "$name" = "$FROM" ] && on=1; [ $on = 1 ] || return 0
          local t0=$(date +%s.%N); "$@"; local t1=$(date +%s.%N)
          awk -v n="$name" -v a="$t0" -v b="$t1" 'BEGIN{printf "%s\t%.2f\n", n, b-a}' | tee -a timings.tsv; }
T0=$(date +%s.%N)
XR=$(env | grep -o '^RTSC_[A-Z0-9_]*' | sed 's/^/-x /' | tr '\n' ' ')
run1() { env OMP_NUM_THREADS=$NT mpirun --bind-to none -np 1 $XR -x OMP_NUM_THREADS "$@" < /dev/null; }
stage scf run1 "$BIN"/pw.x -in scf.in > scf.out
ph1() { rm -rf sub; mkdir sub; cp -r $prefix.save sub/
        sed "s/nq1 = [0-9]*, nq2 = [0-9]*, nq3 = [0-9]*/nq1 = $NS, nq2 = $NS, nq3 = $NS/" ph.in > sub/ph.in
        (cd sub && run1 "$BIN"/ph.x -in ph.in > ph.out); }
stage ph1 ph1
q2r() { cat > sub/dq2r.in <<EOF
&input
  prefix = '$prefix'
  outdir = './'
  fildyn = '$prefix.dyn'
  fildvscf = 'dvscf'
  wpot_dir = './w_pot/'
  do_long_range = .false.
  do_charge_neutral = .false.
/
EOF
        (cd sub && RTSC_DQ=1 OMP_NUM_THREADS=$NT mpirun --bind-to none -np 1 -x RTSC_DQ -x OMP_NUM_THREADS "$BIN"/dvscf_q2r.x \
           -in dq2r.in > dq2r.out < /dev/null); }
stage q2r q2r
export RTSC_DQ=1 RTSC_DQ_WPOT=$W/sub/w_pot/
XR=$(env | grep -o '^RTSC_[A-Z0-9_]*' | sed 's/^/-x /' | tr '\n' ' ')
stage ph2 run1 "$BIN"/ph.x -in ph.in > ph.out
stage collect bash -c "
  rm -rf save; mkdir -p save; nq=\$(sed -n 2p $prefix.dyn0 | awk '{print \$1}')
  for i in \$(seq 1 \$nq); do
    cp $prefix.dyn\$i save/$prefix.dyn_q\$i
    if [ \$i -eq 1 ]; then cp _ph0/$prefix.dvscf1 save/$prefix.dvscf_q1; cp -r _ph0/$prefix.phsave save/
    else cp _ph0/$prefix.q_\$i/$prefix.dvscf1 save/$prefix.dvscf_q\$i; fi
  done"
stage nscf env OMP_NUM_THREADS=$NT mpirun --bind-to none -np 1 -x OMP_NUM_THREADS "$BIN"/pw.x -in nscf.in > nscf.out < /dev/null
stage epw  bash -c "nvidia-cuda-mps-control -d 2>/dev/null || true; OMP_NUM_THREADS=1 mpirun --bind-to none -np $NE -x OMP_NUM_THREADS $BIN/epw.x -npool $NE -in epw.in > epw.out < /dev/null; echo quit | nvidia-cuda-mps-control 2>/dev/null || true"
awk -v a="$T0" -v b="$(date +%s.%N)" 'BEGIN{printf "total\t%.2f\n", b-a}' | tee -a timings.tsv
