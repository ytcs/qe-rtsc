#!/usr/bin/env bash
# F0c queue (FUNNEL.md section 8 rules G1-G5, S1, routing; FUNNEL-LOG "F0c" pre-registration 2026-10-30 09:50).
# Same path as F0b (f0b.sh), generator G1-G5+S1 (PSET sg15). Waits for $WAIT_FOR (a queue log) to show its DONE marker.
#   f0c.sh <in dir: t1-structs.json, t3a-jobs.json> <work dir>
# Mg2IrH6 end to end: T1a (both MLIPs; MLIP hull vs the Mg-Ir-H references) -> T1b (two-model rule)
# -> T2 (PBE + PBEsol at the MatterSim geometry; ZPE-source rule; the access gate is recorded and, as for every
# physics control, bypassed) -> T3a if it reaches T3a. Then T3a (generator G1G2) on H3S, MgB2, Cu, LiAl2Rh;
# MgB3C3 last under a 30-min cap (section 7: run only if <= 30 min; a timeout is recorded as cost-skipped).
# Stage markers go to <work>/queue.log; F0C_DONE ends it.
set -uo pipefail
IN=$(realpath "$1"); mkdir -p "$2"; W=$(realpath "$2"); cd "$W"
F=$(dirname "$(realpath "$0")")
source /root/rtsc/env.sh
unset RTSC_STH_MP
export BIN=${BIN:-/root/rtsc/qe-rtsc/build-gpu/bin} PSEUDO_DIR=${PSEUDO_DIR:-/root/rtsc/qe-rtsc/rtsc/pseudo/sg15}
MS=/root/venv-ms/bin/python; MACE=/root/venv-mace/bin/python
mark() { echo "$1 $(date -u +%H:%M:%S)" | tee -a queue.log; }
export PSET=sg15
if [ -n "${WAIT_FOR:-}" ]; then while ! grep -q DONE "$WAIT_FOR" 2>/dev/null; do sleep 20; done; fi
mark F0C_START

# 1. T1a: MatterSim v1 5M and MACE-MPA-0 relax (concurrently, as in F0); E_hull vs the Mg-Ir-H references
mark T1A_START
OMP_NUM_THREADS=4 $MS "$F/t1.py" relax mattersim "$IN/t1-structs.json" t1-ms.jsonl > t1-ms.log 2>&1 &
OMP_NUM_THREADS=4 $MACE "$F/t1.py" relax mace "$IN/t1-structs.json" t1-mace.jsonl > t1-mace.log 2>&1 &
wait
REFS=H2,MgH2,Mg,Ir,Mg2IrH5,JVASP-165447,JVASP-21730,JVASP-129234,JVASP-180750
$MS "$F/analyze.py" hull1 t1-ms.jsonl t1-mace.jsonl Mg2IrH6 $REFS t1a-hull.json >> queue.log 2>&1
mark T1A_END

# 2. T1b: both models, each on its own relaxed geometry, Mg2IrH6 and every solid of its reaction channels
PH=Mg2IrH6,MgH2,Mg,Ir,Mg2IrH5
mark T1B_START
OMP_NUM_THREADS=4 $MS "$F/t1.py" phonon mattersim t1-ms.jsonl $PH t1b-ms.jsonl > t1b-ms.log 2>&1 &
OMP_NUM_THREADS=4 $MACE "$F/t1.py" phonon mace t1-mace.jsonl $PH t1b-mace.jsonl > t1b-mace.log 2>&1 &
wait
$MS "$F/analyze.py" t1b2 t1b-ms.jsonl t1b-mace.jsonl t1b2.json >> queue.log 2>&1
mark T1B_END

# 3. T2: PBE + PBEsol single points at the MatterSim geometry (every phase of the reaction, H2 included)
mark T2_START
T2_BOTH=1 python3 "$F/t2.py" t1-ms.jsonl Mg2IrH6,MgH2,Mg,Ir,Mg2IrH5,H2 t2 t2.jsonl 8 > t2.log 2>&1
mark T2_END

# 4. ZPE-source rule: DFT Gamma-ZPE for each phase whose parent cell is unstable (< -30 cm^-1) in either MLIP
mark ZPE_START
UNST=$($MS - "$F" <<'EOP'
import json, sys
sys.path.insert(0, sys.argv[1])
from jobs import sym
ld = lambda p: {json.loads(l)["id"]: json.loads(l) for l in open(p) if l.strip()}
a, b, s = ld("t1b-ms.jsonl"), ld("t1b-mace.jsonl"), ld("t1-ms.jsonl")
u = [k for k in a if a[k].get("wmin0", -1e9) < -30.0 or b.get(k, {}).get("wmin0", -1e9) < -30.0]
json.dump({k: dict(structure=sym(s[k]["structure"])) for k in u}, open("zpe-jobs.json", "w"))
print(",".join(u))
EOP
)
echo "unstable in either MLIP (parent cell): ${UNST:-none}" | tee -a queue.log
[ -n "$UNST" ] && python3 "$F/zpe.py" zpe-jobs.json "$UNST" zpe dft-zpe.jsonl 8 >> queue.log 2>&1
$MS "$F/analyze.py" access2 t1-ms.jsonl t1-mace.jsonl t1b-ms.jsonl t1b-mace.jsonl t2.jsonl dft-zpe.jsonl Mg2IrH6 \
    access2.json > access2.log 2>&1
mark ZPE_END

# 5. Does Mg2IrH6 reach T3a? (T1a hull + T1a gas line, T1b two-model rule, T2 metal and non-magnetic;
#    the access gate is bypassed and flagged for this physics control, FUNNEL.md section 2.1)
$MS - "$F" "$IN/t3a-jobs.json" <<'EOP' >> queue.log 2>&1
import json, sys
sys.path.insert(0, sys.argv[1])
from jobs import sym
h = json.load(open("t1a-hull.json")); b = json.load(open("t1b2.json"))["Mg2IrH6"]; acc = json.load(open("access2.json"))
t2 = [json.loads(l) for l in open("t2.jsonl") if l.strip()]
t2p = [r for r in t2 if r["id"] == "Mg2IrH6" and r["xc"] == "pbe"][0]
s = {json.loads(l)["id"]: json.loads(l) for l in open("t1-ms.jsonl") if l.strip()}
path = dict(T1a_hull=h["pass_T1a_hull"], T1a_gas=acc["T1a_line_1.1"] == "pass", T1b=b["pass_T1b"],
            t1b_split=b["t1b_split"], T2_metal=t2p["N_EF"] >= 0.05, T2_nonmag=abs(t2p["mag"]) <= 0.1,
            access_gate=acc["T2_gate"], access_bypassed_flag=True)
path["reaches_T3a"] = all(path[k] for k in ("T1a_hull", "T1a_gas", "T1b", "T2_metal", "T2_nonmag"))
json.dump(path, open("mg2irh6-path.json", "w"), indent=1)
print("Mg2IrH6 path", json.dumps(path))
jobs = json.load(open(sys.argv[2]))
if path["reaches_T3a"]:
    jobs = dict(Mg2IrH6=dict(structure=sym(s["Mg2IrH6"]["structure"]), Ttarget=77.0, role="physics+access",
                             expect="keep"), **jobs)
json.dump(jobs, open("t3a-jobs.json", "w"))
print("T3a jobs", list(jobs))
EOP

# 6. T3a, generator G1-G5+S1 with routing (FP32-MP engine as in F0; EPW 8 ranks). Outer safety cap 2.5 h per control
#    (S1 rerun child cap 45 min and T3b-lite child cap 60 min are inside t3a.py).
for id in Mg2IrH6 H3S MgB2 Cu MgB3C3 LiAl2Rh; do
  python3 -c "import json,sys; sys.exit(sys.argv[1] not in json.load(open('t3a-jobs.json')))" $id || { mark "T3A_SKIP_$id"; continue; }
  cap=9000
  mark "T3A_START_$id"
  timeout -k 20 $cap python3 "$F/t3a.py" t3a-jobs.json $id t3a 8 8 >> t3a.log 2>&1 || echo "$id rc=$? (124 = timeout)" >> t3a.log
  mark "T3A_END_$id"
done
mark F0C_DONE
