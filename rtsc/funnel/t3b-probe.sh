#!/usr/bin/env bash
# F0 T3b probe (FUNNEL.md section 5 step 4): mid-grid DFPT with MP + one anisotropic ME solve at 0.7 T_target.
#   t3b-probe.sh <workroot>
# Mg2IrH6 (B2 inputs): scf k 8^3, q 3^3, Wannier coarse k 6^3, fine 12^3 / 6^3, T = 0.7 * 77 = 53.9 K.
# MgB2 (B1 inputs):    scf k 16^3, q 4^3, Wannier coarse k 4^3, fine 24^3 / 12^3, T = 0.7 * 39 = 27.3 K.
# mu* = 0.10 (T3 convention). Device high-water per q is printed by ph.x (rtsc mem lines): decision (v).
set -uo pipefail
R=/root/rtsc/qe-rtsc/rtsc; W=$(realpath -m "$1"); mkdir -p "$W"
source /root/rtsc/env.sh
export RTSC_STH_MP=1
prep() {  # prep <B> <name> <kscf> <nq> <nk> <nkf> <nqf> <T>
  local B=$1 N=$2 K=$3 Q=$4 NK=$5 KF=$6 QF=$7 T=$8 D=$W/$2-in
  mkdir -p $D; cp $R/bench/$B/smoke/*.in $D/
  sed -i -E "s/^[0-9]+ [0-9]+ [0-9]+ 0 0 0/$K $K $K 0 0 0/" $D/scf.in
  sed -i -E "s/nq1 = [0-9]+, nq2 = [0-9]+, nq3 = [0-9]+/nq1 = $Q, nq2 = $Q, nq3 = $Q/" $D/ph.in $D/epw.in
  python3 - "$D/nscf.in" "$NK" <<'EOI'
import sys
p, n = sys.argv[1], int(sys.argv[2])
s = open(p).read(); s = s[:s.index("K_POINTS crystal")]
pts = [(i/n, j/n, k/n) for i in range(n) for j in range(n) for k in range(n)]
s += f"K_POINTS crystal\n{len(pts)}\n" + "".join(f" {a:.10f} {b:.10f} {c:.10f} {1/len(pts):.6e}\n" for a, b, c in pts)
open(p, "w").write(s)
EOI
  sed -i -E "s/nk1 = [0-9]+, nk2 = [0-9]+, nk3 = [0-9]+/nk1 = $NK, nk2 = $NK, nk3 = $NK/; \
             s/nkf1 = [0-9]+, nkf2 = [0-9]+, nkf3 = [0-9]+/nkf1 = $KF, nkf2 = $KF, nkf3 = $KF/; \
             s/nqf1 = [0-9]+, nqf2 = [0-9]+, nqf3 = [0-9]+/nqf1 = $QF, nqf2 = $QF, nqf3 = $QF/; \
             s/temps = .*/temps = $T/; s/nstemp = .*/nstemp = 1/; s/muc = .*/muc = 0.10/" $D/epw.in
}
prep B2 Mg2IrH6 8 3 6 12 6 53.9
prep B1 MgB2 16 4 4 24 12 27.3
for N in Mg2IrH6 MgB2; do
  echo "== $N start $(date -u +%T)"
  flock /root/rtsc/TIMING.lock $R/bench/run-fork.sh $W/$N-in /root/rtsc/qe-rtsc/build-gpu/bin $W/$N 8 8 > $W/$N.log 2>&1
  echo "== $N end $(date -u +%T) rc=$?"
done
echo T3B_DONE
