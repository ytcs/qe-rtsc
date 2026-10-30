#!/usr/bin/env python3
"""cmpgate.py REF TEST [PREFIX] -- MILESTONE-1 frequency gate (as amended 2026-10-28) on two ph.x run dirs.

Compares every q for which both directories hold <prefix>.dyn<i> (so single-q runs work). Per q:
max |delta omega| (cm-1) over the gated modes, the largest relative deviation among gated modes above
1 cm-1, and the dvscf relative L2 (reported only). Gated: every mode within max(0.5 %, 1 cm-1), except
the three acoustic modes at q = Gamma (the three lowest |omega| there), reported as a diagnostic.
Also reported (not pass/fail): max |delta omega| against the 0.2 cm-1 target.
"""
import glob
import os
import re
import sys

import numpy as np

ref, tst = sys.argv[1], sys.argv[2]
prefix = sys.argv[3] if len(sys.argv) > 3 else None
if prefix is None:
    prefix = os.path.basename(glob.glob(os.path.join(ref, "*.dyn*"))[0]).split(".dyn")[0]


def dyn(d, i):
    p = os.path.join(d, f"{prefix}.dyn{i}")
    if not os.path.exists(p):
        return None, None
    txt = open(p).read()
    q = re.search(r"q = \(\s*(\S+)\s+(\S+)\s+(\S+)", txt)
    xq = np.array([float(x) for x in q.groups()]) if q else None
    w = np.array([float(x) for x in re.findall(r"freq.*=\s*\S+\s*\[THz\]\s*=\s*(\S+)\s*\[cm-1\]", txt)])
    return xq, w


def dvscf(d, i):
    for p in (os.path.join(d, "_ph0", f"{prefix}.q_{i}", f"{prefix}.dvscf1"),
              os.path.join(d, "_ph0", f"{prefix}.dvscf1") if i == 1 else ""):
        if p and os.path.exists(p):
            return np.fromfile(p, dtype=np.complex128)
    return None


ok = True
wmax = 0.0
nq = 0
print(f"{'q':>3s} {'max|dw| cm-1':>13s} {'max rel':>9s} {'dvscf relL2':>12s}  gate   (Gamma acoustic |dw|)")
for i in range(1, 200):
    xa, a = dyn(ref, i)
    xb, b = dyn(tst, i)
    if a is None or b is None:
        continue
    nq += 1
    d = np.abs(a - b)
    mask = np.ones(len(a), bool)
    extra = ""
    if xa is not None and np.allclose(xa, 0.0):
        ac = np.argsort(np.abs(a))[:3]
        mask[ac] = False
        extra = "  (" + " ".join(f"{x:.3f}" for x in d[ac]) + ")"
    g = d[mask]
    big = mask & (np.abs(a) > 1.0)
    rel = (d[big] / np.abs(a[big])).max() if big.any() else 0.0
    gate = bool(np.all(g <= np.maximum(0.005 * np.abs(a[mask]), 1.0)))
    va, vb = dvscf(ref, i), dvscf(tst, i)
    dv = np.linalg.norm(vb - va) / np.linalg.norm(va) if va is not None and vb is not None and va.size == vb.size \
        else float("nan")
    ok &= gate
    wmax = max(wmax, g.max() if g.size else 0.0)
    print(f"{i:3d} {g.max():13.4f} {rel:9.2e} {dv:12.3e}  {'PASS' if gate else 'FAIL'}{extra}")
print(f"{nq} q compared; max |dw| (gated modes) {wmax:.4f} cm-1 (target 0.2 {'met' if wmax <= 0.2 else 'missed'}); "
      f"gate {'PASS' if ok and nq else 'FAIL'}")
sys.exit(0 if ok and nq else 1)
