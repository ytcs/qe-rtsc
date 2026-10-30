#!/usr/bin/env bash
# M3-3 Phase 0 queue (groups/compute/M3-3-LOG.md pre-registration). Waits for $WAIT_FOR's DONE marker.
#   m33.sh <work dir> <stages: A | AB> <systems, e.g. "Si B1 B2">
set -uo pipefail
mkdir -p "$1"; W=$(realpath "$1"); cd "$W"; STAGES=$2; SYSTEMS=$3
M=$(dirname "$(realpath "$0")")
source /root/rtsc/env.sh
BIN=/root/rtsc/qe-rtsc/build-gpu/bin; PS=/root/rtsc/qe-rtsc/rtsc/pseudo/sg15
PY=/root/venv-m33/bin/python
export W90=/root/w90/src/wannier90.x P2W=$BIN/pw2wannier90.x MPIRUN="mpirun --bind-to none -np 1"
mark() { echo "$1 $(date -u +%H:%M:%S)" | tee -a queue.log; }
pwx() { (cd "$1" && OMP_NUM_THREADS=8 RTSC_STH=0 mpirun --bind-to none -np 1 -x OMP_NUM_THREADS -x RTSC_STH $BIN/pw.x -in "$2" > "${2%.in}.out" 2>&1); }
if [ -n "${WAIT_FOR:-}" ]; then while ! grep -q DONE "$WAIT_FOR" 2>/dev/null; do sleep 20; done; fi
mark M33_START
for X in $SYSTEMS; do
  D=$W/$X; mkdir -p $D; ln -sfn $PS $D/upf
  if [ ! -f $D/dft.done ]; then
    mark "DFT_START_$X"
    python3 $M/prep.py $X $D $PS >> queue.log 2>&1
    pwx $D scf.in
    [ -f $D/scf_ph.in ] && { mkdir -p $D/phrun; pwx $D scf_ph.in; }
    (cd $D && mpirun --bind-to none -np 1 $BIN/pp.x -in pp.in > pp.out 2>&1)
    (cd $D && OMP_NUM_THREADS=8 RTSC_STH=1 RTSC_STRICT=1 mpirun --bind-to none -np 1 -x OMP_NUM_THREADS -x RTSC_STH -x RTSC_STRICT \
       $BIN/ph.x -in ph.in > ph.out 2>&1)
    f=$(find $D -name "x.dvscf1" | head -1); cp "$f" $D/dvscf
    for s in nf nc; do mkdir -p $D/$s; rm -rf $D/$s/x.save; cp -r $D/x.save $D/$s/; done
    mv $D/x.save $D/scf_save; mkdir -p $D/scf; ln -sfn $D/scf_save $D/scf/x.save
    pwx $D nscf_f.in; pwx $D nscf_c.in
    EF=$($PY -c "import sys; sys.path.insert(0,'$M'); import pwtool as P; print(P.read_xml('$D/scf/x.save')['ef']*P.RYEV)")
    NC=$($PY -c "print(eval(open('$D/cfg.txt').read())['nc'])")
    for d in 2 4 6; do (cd $D && $PY $M/wan.py nc x $NC $EF $d W$d > W$d.log 2>&1) || echo "wan $X $d failed" >> queue.log; done
    (cd $D && $PY $M/wan.py nc x $NC $EF 2 Wpoor poor > Wpoor.log 2>&1) || echo "wan $X poor failed" >> queue.log
    touch $D/dft.done
    mark "DFT_END_$X"
  fi
  # q in crystal coordinates from the dyn file (cart 2pi/alat)
  Q=$($PY - $D <<'EOP'
import sys, glob, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0] if '/' in sys.argv[0] else '.')
d = sys.argv[1]
f = sorted(glob.glob(f"{d}/**/x.dyn*", recursive=True))
f = [x for x in f if not x.endswith("dyn0")][0]
t = open(f).read()
q = np.array([float(x) for x in t.split("q = (")[1].split(")")[0].split()])
sys.path.insert(0, "/root/rtsc/qe-rtsc/rtsc/funnel/m33")
import pwtool as P
x = P.read_xml(f"{d}/scf/x.save")
qc = q @ (x["at"] / x["alat"]).T
print(" ".join(f"{v:.10f}" for v in qc))
EOP
)
  NC=$($PY -c "print(eval(open('$D/cfg.txt').read())['nc'])"); NF=$($PY -c "print(eval(open('$D/cfg.txt').read())['nf'])")
  WS=""; for d in 2 4 6 poor; do [ -f $D/W$d/w_u.mat ] && WS="$WS,W$d=$D/W$d"; done; WS=${WS#,}
  echo "$X q_crys $Q Nc $NC Nf $NF W $WS" >> queue.log
  mark "A_START_$X"; $PY $M/run.py $D A $NC $NF $Q "$WS" $D/stageA.json > $D/stageA.log 2>&1; mark "A_END_$X"
  if [ "$STAGES" = AB ]; then
    mark "B_START_$X"; M33_XP=cupy $PY $M/run.py $D B $NC $NF $Q "$WS" $D/stageB.json > $D/stageB.log 2>&1; mark "B_END_$X"
  fi
done
mark M33_DONE
