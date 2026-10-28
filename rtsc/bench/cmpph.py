#!/usr/bin/env python3
"""cmpph.py REF TEST [PREFIX] -- compare two ph.x run directories (ldisp, fildvscf set).

Per q: max |delta omega| (cm-1) over the modes of <prefix>.dyn<i> (all "freq" lines), the largest
relative deviation max(|d|/|w|) among modes above 1 cm-1, and the dvscf relative L2
|dv_test - dv_ref| / |dv_ref| (whole _ph0/<prefix>.dvscf1 or _ph0/<prefix>.q_<i>/<prefix>.dvscf1
file read as complex128). Gate (MILESTONE-1, class N): every mode within max(0.5 %, 1 cm-1);
the target is 0.2 cm-1; dvscf <= 5e-6.
"""
import glob
import os
import re
import sys

import numpy as np

ref, tst = sys.argv[1], sys.argv[2]
prefix = sys.argv[3] if len(sys.argv) > 3 else None
if prefix is None:
    dyn0 = glob.glob(os.path.join(ref, "*.dyn0"))[0]
    prefix = os.path.basename(dyn0)[:-5]


def freqs(d, i):
    with open(os.path.join(d, f"{prefix}.dyn{i}")) as fh:
        return np.array([float(x) for x in re.findall(r"freq.*=\s*\S+\s*\[THz\]\s*=\s*(\S+)\s*\[cm-1\]", fh.read())])


def dvscf(d, i):
    p = os.path.join(d, "_ph0", f"{prefix}.dvscf1") if i == 1 else \
        os.path.join(d, "_ph0", f"{prefix}.q_{i}", f"{prefix}.dvscf1")
    return np.fromfile(p, dtype=np.complex128) if os.path.exists(p) else None


with open(os.path.join(ref, f"{prefix}.dyn0")) as fh:
    fh.readline()
    nq = int(fh.readline().split()[0])
ok = True
wmax = 0.0
print(f"{'q':>3s} {'max|dw| cm-1':>13s} {'max rel':>9s} {'dvscf relL2':>12s}")
for i in range(1, nq + 1):
    try:
        a, b = freqs(ref, i), freqs(tst, i)
    except FileNotFoundError:
        print(f"{i:3d}  missing")
        ok = False
        continue
    d = np.abs(a - b)
    big = np.abs(a) > 1.0
    rel = (d[big] / np.abs(a[big])).max() if big.any() else 0.0
    gate = np.all(d <= np.maximum(0.005 * np.abs(a), 1.0))
    va, vb = dvscf(ref, i), dvscf(tst, i)
    dv = np.linalg.norm(vb - va) / np.linalg.norm(va) if va is not None and vb is not None and va.size == vb.size \
        else float("nan")
    ok &= bool(gate) and (dv <= 5e-6 or np.isnan(dv))
    wmax = max(wmax, d.max())
    print(f"{i:3d} {d.max():13.4f} {rel:9.2e} {dv:12.3e}  {'PASS' if gate and not dv > 5e-6 else 'FAIL'}")
print(f"overall max |dw| {wmax:.4f} cm-1 (target 0.2): {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
