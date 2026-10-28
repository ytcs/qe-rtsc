#!/usr/bin/env bash
# Standalone Wannier check of B2 (no ph.x): scf -> nscf on an NK^3 full list -> W90 variants,
# each scored against DFT bands on the interpolation path (cmp.py).
#   run.sh <bin dir> <wannier90.x> <workdir> <NK> <np> "name|nw|froz|projs" ...
set -euo pipefail
HERE=$(dirname "$(realpath "$0")"); BENCH=$HERE/../..
BIN=$(realpath "$1"); W90=$(realpath "$2"); W=$3; NK=$4; NP=$5; shift 5
mkdir -p "$W"; W=$(realpath "$W"); cd "$W"
PSEUDO_DIR=$(realpath "${PSEUDO_DIR:-$BENCH/../pseudo/sg15}")
export OMP_NUM_THREADS=1 NK
M="mpirun --bind-to none -np $NP"
python3 - "$BENCH" "$NK" "$PSEUDO_DIR" <<'PY'
import sys; sys.path.insert(0, sys.argv[1]); import mkbench as m
s = m.SYSTEMS["B2"]; n = int(sys.argv[2]); p = sys.argv[3]
open("scf.in", "w").write(m.pw(s, "scf", "K_POINTS automatic\n12 12 12 0 0 0\n").replace("@PSEUDO@", p))
open("nscf.in", "w").write(m.pw(s, "nscf", m.kfull(n), "\n    nbnd = 40").replace("@PSEUDO@", p))
PY
cp "$HERE"/pp.in .
[ -f scf.out ] && grep -q "JOB DONE" scf.out || $M "$BIN"/pw.x -nk 4 -in scf.in > scf.out
rm -rf bands && mkdir bands && cp -r Mg2IrH6.save Mg2IrH6.xml bands/
[ -f nscf.out ] && grep -q "JOB DONE" nscf.out || $M "$BIN"/pw.x -nk 4 -in nscf.in > nscf.out
for v in "$@"; do
  name=${v%%|*}; mkdir -p "$name"
  python3 "$HERE"/mkwin.py "$v" > "$name"/mg2irh6.win
  ( cd "$name" && ln -sf ../Mg2IrH6.save . && "$W90" -pp mg2irh6 &&
    $M "$BIN"/pw2wannier90.x -in ../pp.in > pp.out && "$W90" mg2irh6 ) || echo "FAIL $name"
done
# DFT bands on exactly the Wannier interpolation path, from the scf charge density
first=${1%%|*}; nk=$(head -1 "$first"/mg2irh6_band.kpt)
sed -e "s/calculation = 'nscf'/calculation = 'bands'/" -e '/^K_POINTS/,$d' nscf.in > bands/bands.in
{ echo "K_POINTS crystal"; echo "$nk"; tail -n +2 "$first"/mg2irh6_band.kpt; } >> bands/bands.in
( cd bands && $M "$BIN"/pw.x -nk 4 -in bands.in > bands.out )
python3 - <<'PY'
import numpy as np, xml.etree.ElementTree as ET
r = ET.parse("bands/Mg2IrH6.save/data-file-schema.xml").getroot()
np.save("dft_path.npy", np.array([[float(x) for x in k.find("eigenvalues").text.split()]
                                  for k in r.iter("ks_energies")]) * 27.211386245988)
PY
EF=$(grep -oP "the Fermi energy is\s+\K\S+" nscf.out)
for v in "$@"; do python3 "$HERE"/cmp.py "${v%%|*}" "$EF"; done | tee scores.txt
