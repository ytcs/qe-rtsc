#!/usr/bin/env bash
# P1 queue (FUNNEL.md section 8 P1; FUNNEL-LOG "F0c" pre-registration 2026-10-30 09:50). Waits for $WAIT_FOR's DONE marker.
#   p1.sh <in dir: p1-base.json (H3S MgB2 Cu Nb Pb Al jobs), mg2irh6-st.json> <work dir> [F0c t3a-jobs.json for Mg2IrH6]
# 1. NLCC gate: Mg2IrH6 SPMS scf, ph.x q1-q2 with the T3a engine (RTSC_STH=1 RTSC_STH_MP=1) and with RTSC_STH=0; frequencies
#    within max(0.5 %, 1 cm^-1), Gamma acoustic excluded. A failure stops P1 (decision: stay on SG15).
# 2. SG15 arm extras (Nb, Pb, Al), generator G1-G5+S1.  3. SPMS arm (7 controls), same generator.
set -uo pipefail
IN=$(realpath "$1"); mkdir -p "$2"; W=$(realpath "$2"); cd "$W"
F=$(dirname "$(realpath "$0")")
source /root/rtsc/env.sh
unset RTSC_STH_MP
export BIN=${BIN:-/root/rtsc/qe-rtsc/build-gpu/bin}
SG=/root/rtsc/qe-rtsc/rtsc/pseudo/sg15; SP=/root/pseudo/spms
mark() { echo "$1 $(date -u +%H:%M:%S)" | tee -a queue.log; }
if [ -n "${WAIT_FOR:-}" ]; then while ! grep -q DONE "$WAIT_FOR" 2>/dev/null; do sleep 20; done; fi
mark P1_START
python3 - "$IN/p1-base.json" "${3:-/root/f0c/run/t3a-jobs.json}" <<'EOP' >> queue.log 2>&1
import json, sys
b = json.load(open(sys.argv[1])); f = json.load(open(sys.argv[2]))
json.dump({k: b[k] for k in ("Nb", "Pb", "Al")}, open("p1-sg15.json", "w"))
sp = dict(b); sp["Mg2IrH6"] = f["Mg2IrH6"]      # the same T3a input structure as the F0c SG15 run
json.dump(sp, open("p1-spms.json", "w")); print("P1 jobs", list(sp))
EOP
# 1. NLCC gate
mark GATE_START
PSET=spms PSEUDO_DIR=$SP python3 "$F/p1gate.py" prep "$IN/mg2irh6-st.json" gate/scf >> queue.log 2>&1
(cd gate/scf && RTSC_STH=1 RTSC_STRICT=1 RTSC_STH_MP=1 OMP_NUM_THREADS=8 mpirun --bind-to none -np 1 -x RTSC_STH -x RTSC_STRICT \
   -x RTSC_STH_MP -x OMP_NUM_THREADS $BIN/pw.x -in scf.in > scf.out 2>&1)
RTSC_STH_MP=1 bash /root/rtsc/qe-rtsc/rtsc/bench/phq.sh gate/scf gate/engine $BIN 1 2 >> queue.log 2>&1
RTSC_STH=0 RTSC_STRICT=0 bash /root/rtsc/qe-rtsc/rtsc/bench/phq.sh gate/scf gate/vanilla $BIN 1 2 >> queue.log 2>&1
python3 "$F/p1gate.py" compare gate/engine gate/vanilla gate/gate.json >> queue.log 2>&1
mark GATE_END
if ! python3 -c "import json,sys; sys.exit(0 if json.load(open('gate/gate.json'))['pass_gate'] else 1)"; then
  mark P1_GATE_FAIL; mark P1_DONE; exit 0
fi
# 2. SG15 arm extras, 3. SPMS arm
run_arm() {   # $1 pset, $2 pseudo dir, $3 jobs, $4 ids
  for id in $4; do
    mark "T3A_START_$1_$id"
    PSET=$1 PSEUDO_DIR=$2 timeout -k 20 9000 python3 "$F/t3a.py" "$3" $id t3a-$1 8 8 >> t3a-$1.log 2>&1 || echo "$id rc=$?" >> t3a-$1.log
    mark "T3A_END_$1_$id"
  done
}
run_arm sg15 $SG p1-sg15.json "Al Pb Nb"
run_arm spms $SP p1-spms.json "Al Pb Cu H3S MgB2 Nb Mg2IrH6"
mark P1_DONE
