#!/usr/bin/env python3
"""ephcmp.py REF TEST -- compare the ephmat files (<prefix>.ephmat/ephmatN, egnv, freq, ikmap) of
two EPW run directories written with the same number of pools.

ephmatN is a stream file: two int32 (pool id, fermicount) followed by the raw float64 g^2 list.
Prints, per file, whether the bytes are identical, else max |d| / max |ref| and the number of
entries whose relative difference (to max |ref|) exceeds 1e-10.
"""
import glob
import os
import sys

import numpy as np


def ephdir(d):
    c = glob.glob(os.path.join(d, "*.ephmat"))
    if len(c) != 1:
        sys.exit(f"no unique *.ephmat in {d}")
    return c[0]


ref, tst = ephdir(sys.argv[1]), ephdir(sys.argv[2])
worst = 0.0
for f in sorted(os.listdir(ref)):
    a_p, b_p = os.path.join(ref, f), os.path.join(tst, f)
    if not os.path.exists(b_p):
        print(f"{f:10s} missing in test")
        worst = np.inf
        continue
    a, b = open(a_p, "rb").read(), open(b_p, "rb").read()
    if a == b:
        print(f"{f:10s} identical ({len(a)} bytes)")
        continue
    if len(a) != len(b):
        print(f"{f:10s} DIFFERENT SIZE {len(a)} vs {len(b)}")
        worst = np.inf
        continue
    if f.startswith("ephmat"):
        h = 8
        ha, hb = np.frombuffer(a[:h], np.int32), np.frombuffer(b[:h], np.int32)
        x, y = np.frombuffer(a[h:], np.float64), np.frombuffer(b[h:], np.float64)
        hdr = "" if (ha == hb).all() else f" header {ha} vs {hb}"
    else:
        x, y = np.frombuffer(a[: len(a) // 8 * 8], np.float64), np.frombuffer(b[: len(b) // 8 * 8], np.float64)
        hdr = " (raw float64 view)"
    d = np.abs(x - y)
    s = np.abs(x).max() if x.size else 1.0
    r = d.max() / s if x.size else 0.0
    worst = max(worst, r)
    print(f"{f:10s} n={x.size:10d} max|d|/max|ref| {r:9.2e}  n(>1e-10) {int((d > 1e-10 * s).sum())}{hdr}")
print(f"WORST {worst:.2e}")
