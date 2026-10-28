#!/usr/bin/env python3
"""gate.py REF TEST -- MILESTONE-1 accuracy gate on two EPW run directories.

Checks lambda and w_log (1 %), Allen-Dynes Tc (max(2 %, 0.5 K)), and at every temperature the
mean leading-edge gap (mean of delta(0) over the Fermi-surface states in *.imag_aniso_gap_FS_<T>,
2 %) and the two peak positions of the gap distribution (3 %, MgB2
two-gap structure; peaks of the smoothed per-state gap density, see peaks()). Also prints the epmatwp difference when both runs wrote one.
"""
import glob
import os
import re
import sys

import numpy as np

ref, tst = sys.argv[1], sys.argv[2]
ok = True


def grab(d, pat):
    with open(os.path.join(d, "epw.out")) as fh:
        m = re.findall(pat, fh.read())
    return float(m[-1]) if m else float("nan")


def row(name, a, b, tol_rel, tol_abs=0.0):
    global ok
    d = abs(b - a)
    p = d <= max(tol_rel * abs(a), tol_abs)
    ok &= p
    rel = d / abs(a) if a else 0.0
    verdict = "PASS" if p else "FAIL"
    print(f"{name:36s} ref {a:12.6f} test {b:12.6f}  rel {rel:9.2e}  {verdict}")


def peaks(gaps, width=1.0, sep=3.0):
    """Two highest maxima, at least `sep` meV apart, of the Gaussian-smoothed (`width` meV)
    density of the per-state leading-edge gaps. The raw *.imag_aniso_gap0_* histogram is too
    spiky on these grids: its argmax flips between near-equal bins from run to run."""
    x = np.arange(gaps.min() - 3 * width, gaps.max() + 3 * width, 0.01)
    y = np.exp(-0.5 * ((x[:, None] - gaps[None, :]) / width) ** 2).sum(axis=1)
    loc = [i for i in range(1, len(y) - 1) if y[i] >= y[i - 1] and y[i] > y[i + 1]]
    loc.sort(key=lambda i: -y[i])
    sel = []
    for i in loc:
        if all(abs(x[i] - x[j]) > sep for j in sel):
            sel.append(i)
        if len(sel) == 2:
            break
    return sorted(x[i] for i in sel)


row("lambda", grab(ref, r"coupling strength =\s*([\d.]+)"),
    grab(tst, r"coupling strength =\s*([\d.]+)"), 0.01)
row("w_log [meV]", grab(ref, r"Estimated w_log =\s*([\d.]+)"),
    grab(tst, r"Estimated w_log =\s*([\d.]+)"), 0.01)
pat = r"Allen-Dynes modified McMillan expression =\s*([\d.]+)"
row("Tc Allen-Dynes [K]", grab(ref, pat), grab(tst, pat), 0.02, 0.5)
for f in sorted(glob.glob(os.path.join(ref, "*.imag_aniso_gap_FS_*"))):
    T = f.rsplit("_", 1)[1]
    g = os.path.join(tst, os.path.basename(f))
    if not os.path.exists(g):
        print("missing", g)
        ok = False
        continue
    a = np.loadtxt(f, comments="#")[:, -1]
    b = np.loadtxt(g, comments="#")[:, -1]
    row(f"mean leading-edge gap {T} K [meV]", a.mean(), b.mean(), 0.02)
    pa, pb = peaks(a), peaks(b)
    if len(pa) == 2 and len(pb) == 2:
        for i in range(2):
            row(f"  gap peak {i + 1} at {T} K [meV]", pa[i], pb[i], 0.03)
    else:
        print(f"  peaks at {T} K: ref {pa} test {pb} (not bimodal)")
        ok = False
ew = [os.path.join(d, n) for d in (ref, tst) for n in os.listdir(d) if n.endswith(".epmatwp")]
if len(ew) == 2:
    a = np.fromfile(ew[0], dtype=np.complex128)
    b = np.fromfile(ew[1], dtype=np.complex128)
    if a.size == b.size and a.size:
        print(f"epmatwp: max|d| {np.abs(a - b).max():.2e}  "
              f"rel L2 {np.linalg.norm(a - b) / np.linalg.norm(a):.2e}")
print("GATE", "PASS" if ok else "FAIL")
